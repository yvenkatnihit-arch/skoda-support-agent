"""Every tool, registered once with where it may be used. To add a tool: write it, then add one register() line.

    register(tool, agents=["locator"])          # only these agents get it
    register(tool, shared=True)                 # every agent gets it (read-only tools only)
    register(tool, agents=["lead"], action=True)  # changes something: the customer must Confirm/Cancel first
    register(tool, ..., action=True, precheck=fn)  # fn(args) -> what to ask first, or None; runs before Confirm

Agents get their tools from here (AgentSpec.tools), so an agent's own file never lists tools. The catalog
(app/agents/__init__.py) checks at startup that every agent named here exists and that action tools belong
only to the action agents (which run one at a time, after the parallel group).
"""

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.tools.data_tools import (
    calculate_on_road_price,
    calculate_range,
    get_service_cost,
    get_service_schedule,
    get_variant_details,
)
from app.tools.geo_tools import find_nearest_dealers, geocode_location
from app.tools.lead_tools import book_test_drive, contact_problem, save_lead
from app.tools.retrieval import search_brochures, search_service_docs
from app.tools.ticket_tools import create_ticket, get_ticket_status, ticket_problem


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ToolEntry:
    tool: BaseTool
    agents: tuple[str, ...]
    shared: bool
    action: bool
    # Action tools: args -> what the agent must still ask the customer, or None. The approval step runs it before
    # Confirm, so the customer is never asked to confirm something the tool would reject or that is incomplete.
    precheck: Callable[[dict], str | None] | None = None


TOOL_REGISTRY: dict[str, ToolEntry] = {}


def register(tool: BaseTool, *, agents: list[str] | tuple[str, ...] = (), shared: bool = False,
             action: bool = False, precheck: Callable[[dict], str | None] | None = None) -> None:
    if tool.name in TOOL_REGISTRY:
        raise RegistryError(f"tool '{tool.name}' is registered twice")
    if shared and action:
        raise RegistryError(f"tool '{tool.name}': an action tool can't be shared (only one agent may pause)")
    if not shared and not agents:
        raise RegistryError(f"tool '{tool.name}': give agents=[...] or shared=True")
    TOOL_REGISTRY[tool.name] = ToolEntry(tool, tuple(agents), shared, action, precheck)


def precheck_problem(tool_name: str, args: dict) -> str | None:
    entry = TOOL_REGISTRY.get(tool_name)
    return entry.precheck(args) if entry and entry.precheck else None


def _contact_check(args: dict) -> str | None:
    return contact_problem(args.get("name"), args.get("phone"), args.get("city"))


def tools_for(agent: str) -> list[BaseTool]:
    """The agent's tools in registration order: its own plus the shared ones."""
    return [e.tool for e in TOOL_REGISTRY.values() if e.shared or agent in e.agents]


# ---- the tools ---------------------------------------------------------------------
register(search_brochures, agents=["car_info"])
register(get_variant_details, agents=["quotation"])
register(calculate_on_road_price, agents=["quotation"])
register(calculate_range, agents=["quotation"])
register(get_service_cost, agents=["quotation"])
register(geocode_location, agents=["locator", "support"])
register(find_nearest_dealers, agents=["locator", "support"])   # support: the centre's contact
register(search_service_docs, agents=["service", "support"])    # support: roadside assistance, warranty, booking
register(get_service_schedule, agents=["service"])
register(save_lead, agents=["lead"], action=True, precheck=_contact_check)
register(book_test_drive, agents=["lead"], action=True, precheck=_contact_check)
register(create_ticket, agents=["support"], action=True, precheck=ticket_problem)
register(get_ticket_status, agents=["support"])

ALL_TOOLS = [e.tool for e in TOOL_REGISTRY.values()]

# Action tools: the graph pauses for the customer's Confirm / Cancel before running them.
TOOLS_REQUIRING_APPROVAL = {name for name, e in TOOL_REGISTRY.items() if e.action}
