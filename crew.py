"""
crew.py
This is where I wire everything together into the actual pipeline.
There are 4 agents and 4 tasks that run one after the other for each invoice.
The extractor cleans the data, the validator runs all the checks using its tools,
the resolver decides what to do with the results, and the reporter writes the final report.
Only the validator gets the tools - the others just use the LLM to think through things.
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from config.config import ACTIVE_MODEL, ACTIVE_FAST_MODEL, ACTIVE_TEMPERATURE, LLM_PROVIDER

from tools.check_tools import (
    check_invoice_format,
    check_duplicate_invoice,
    check_gstin_format,
    check_gst_rate_consistency,
    check_line_item_arithmetic,
    check_subtotal_matches_line_items,
    check_tds_applicability,
    check_tds_section_and_rate,
    check_po_amount_tolerance,
    check_approved_vendor,
)


@CrewBase
class ComplianceValidatorCrew:
    """
    The main crew class. CrewBase handles loading agents.yaml and tasks.yaml
    automatically so I don't have to wire up the config files manually.
    """

    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    def __init__(self):
        print(f"[crew] Using LLM provider: {LLM_PROVIDER} | model: {ACTIVE_MODEL} | fast: {ACTIVE_FAST_MODEL}")
        self.llm      = LLM(model=ACTIVE_MODEL,      temperature=ACTIVE_TEMPERATURE)
        self.fast_llm = LLM(model=ACTIVE_FAST_MODEL, temperature=ACTIVE_TEMPERATURE)

    # ============================================================
    # agents
    # each agent loads its personality and goal from agents.yaml
    # I turned off delegation so each agent just does its own job
    # ============================================================

    @agent
    def extractor_agent(self) -> Agent:
        """Just reads the invoice and normalizes it. No tools needed for this."""
        return Agent(
            config=self.agents_config["extractor_agent"],
            llm=self.fast_llm,
            allow_delegation=False,
            verbose=True
        )

    @agent
    def validator_agent(self) -> Agent:
        """This agent runs all 10 checks by calling the tools. It uses fast_llm for
        deciding which tool to call next but the TDS tools call the full model on their own."""
        return Agent(
            config=self.agents_config["validator_agent"],
            llm=self.fast_llm,
            allow_delegation=False,
            verbose=True,
            tools=[
                check_invoice_format,
                check_duplicate_invoice,
                check_gstin_format,
                check_gst_rate_consistency,
                check_line_item_arithmetic,
                check_subtotal_matches_line_items,
                check_tds_applicability,
                check_tds_section_and_rate,
                check_po_amount_tolerance,
                check_approved_vendor,
            ]
        )

    @agent
    def resolver_agent(self) -> Agent:
        """Looks at all the check results and makes the final call. No tools needed."""
        return Agent(
            config=self.agents_config["resolver_agent"],
            llm=self.fast_llm,
            allow_delegation=False,
            verbose=True
        )

    @agent
    def reporter_agent(self) -> Agent:
        """Takes everything and formats it into the output JSON schema. No tools needed."""
        return Agent(
            config=self.agents_config["reporter_agent"],
            llm=self.fast_llm,
            allow_delegation=False,
            verbose=True
        )

    # ============================================================
    # tasks
    # each task loads its instructions from tasks.yaml
    # in sequential mode each task automatically gets the output from the previous one as context
    # ============================================================

    @task
    def extract_task(self) -> Task:
        """Normalize the raw invoice into a clean consistent structure."""
        return Task(
            config=self.tasks_config["extract_task"],
            agent=self.extractor_agent()
        )

    @task
    def validate_task(self) -> Task:
        """Run all 10 compliance checks using the available tools."""
        return Task(
            config=self.tasks_config["validate_task"],
            agent=self.validator_agent()
        )

    @task
    def resolve_task(self) -> Task:
        """Look at the check results and decide what to do with this invoice."""
        return Task(
            config=self.tasks_config["resolve_task"],
            agent=self.resolver_agent()
        )

    @task
    def report_task(self) -> Task:
        """Write the final compliance report in the right format."""
        return Task(
            config=self.tasks_config["report_task"],
            agent=self.reporter_agent()
        )

    # ============================================================
    # crew
    # putting everything together into a sequential pipeline
    # sequential means they run one after another, not in parallel
    # ============================================================

    @crew
    def crew(self) -> Crew:
        """The full pipeline."""
        return Crew(
            agents=self.agents,   # auto-collected from @agent methods
            tasks=self.tasks,     # auto-collected from @task methods
            process=Process.sequential,
            verbose=True
        )
