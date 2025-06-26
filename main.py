from typing_extensions import TypedDict
from typing import Annotated
from src.agents import *
from langgraph.graph import END, START, StateGraph
import os
from dotenv import load_dotenv

load_dotenv()

added_edges = set()


class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    graph_state: Annotated[dict, lambda d1, d2: {**d1, **d2}]


# add all the nodes
builder_flow_1 = StateGraph(State)
builder_flow_1.add_node("RequirementsAnalysisAgent", high_to_lowlevel_requirements)
builder_flow_1.add_node("ComponentsAnalysisAgent", components_analysis)
builder_flow_1.add_node("ExternalInteligenceAgent", external_inteligence)
builder_flow_1.add_node("BreakFlow", break_flow)


builder_flow_2 = StateGraph(State)
builder_flow_2.add_node("RelationshipAgent", relationship_extraction)
builder_flow_2.add_node("ZoneComponentAnalysisAgent", zone_components_analysis)
builder_flow_2.add_node("ColorMapInteligenceAgent", map_type_with_color)
builder_flow_2.add_node("LayoutGenerationAgent", generate_layout)
builder_flow_2.add_node("OverLapsFixAgent", check_and_fix_zone_overlaps_layout)
builder_flow_2.add_node("RenderingAgent", render_layout)


def components_from_llm(requirement, temperature, domain):
    print("debug :::::: components_from_llm")
    initial_state = load_initial_state(requirement, temperature, domain)
    builder_flow_1.add_edge(START, "RequirementsAnalysisAgent")
    # if "RequirementsAnalysisAgent" in builder_flow_1.nodes:  # Check if node exists
    #     # Ensure edges are not added twice in your logic
    #     pass
    if ("RequirementsAnalysisAgent", "decide_mood") not in added_edges:
        builder_flow_1.add_conditional_edges("RequirementsAnalysisAgent", decide_mood)
        added_edges.add(("RequirementsAnalysisAgent", "decide_mood"))
    # builder_flow_1.add_conditional_edges("RequirementsAnalysisAgent", decide_mood)
    builder_flow_1.add_edge("ComponentsAnalysisAgent", "ExternalInteligenceAgent")
    builder_flow_1.add_edge("ExternalInteligenceAgent", END)
    builder_flow_1.add_edge("BreakFlow", END)

    # # Initialize a set to track added conditional edges
    # added_edges = getattr(builder_flow_1, "added_edges", set())

    # # Check before adding the edge
    # if ("RequirementsAnalysisAgent", "decide_mood") not in added_edges:
    #     builder_flow_1.add_conditional_edges("RequirementsAnalysisAgent", decide_mood)
    #     added_edges.add(("RequirementsAnalysisAgent", "decide_mood"))  # Mark as added

    # # Store the updated set back to builder (if modifying builder directly is needed)
    # builder_flow_1.added_edges = added_edges
    # # builder.add_conditional_edges("RequirementsAnalysisAgent",decide_mood)

    graph = builder_flow_1.compile()
    state = graph.invoke(initial_state)
    # print(state)
    return state


def load_initial_state(requirement, temperature, domain):
    initial_state: json = {
        "graph_state": {
            "user_requirements": requirement,
            "api_endpoint": os.environ.get("API_ENDPOINT"),
            "api_key": os.environ.get("API_KEY"),
            "temperature": temperature,
            "domain":domain
        }
    }
    return initial_state


requirements = """HMI layout for a car dashboard all the options along with Speedometer and a tachometer."""

initial_state = {
    "graph_state": {
        "user_requirements": requirements,
        "api_endpoint": os.environ.get("API_ENDPOINT"),
        "api_key": os.environ.get("API_KEY"),
    }
}

# components_from_llm(initial_state)


def layout_generation(final_components, temperature, domain, requirement):
    print("debug :::: layout_generation")
    state = load_final_state(final_components, temperature, domain, requirement)
    builder_flow_2.add_edge(START, "RelationshipAgent")
    builder_flow_2.add_edge("RelationshipAgent", "ZoneComponentAnalysisAgent")
    builder_flow_2.add_edge("ZoneComponentAnalysisAgent", "ColorMapInteligenceAgent")
    builder_flow_2.add_edge("ColorMapInteligenceAgent", "LayoutGenerationAgent")
    builder_flow_2.add_edge("LayoutGenerationAgent", "OverLapsFixAgent")
    builder_flow_2.add_edge("OverLapsFixAgent", END)
    # builder_flow_2.add_edge("RenderingAgent", END)
    graph = builder_flow_2.compile()
    try:
        state = graph.invoke(state)
        return state
    except Exception as e:
        return {"error": str(e)}


# final_components = [{'components': [{'name': 'Speedometer', 'type': 'gauge', 'shape': 'circular', 'category': 'informational'}, {'name': 'Tachometer', 'type': 'gauge', 'shape': 'circular', 'category': 'informational'}, {'name': 'Climate Control Panel', 'type': 'panel', 'shape': 'rectangular', 'category': 'interactive'}, {'name': 'Media Control Panel', 'type': 'panel', 'shape': 'rectangular', 'category': 'interactive'}, {'name': 'Color', 'type': 'attribute', 'shape': 'rectangular', 'category': 'informational'}, {'name': 'Contrast', 'type': 'attribute', 'shape': 'rectangular', 'category': 'informational'}, {'name': 'Safety Standards', 'type': 'attribute', 'shape': 'rectangular', 'category': 'informational'}, {'name': 'Buttons', 'type': 'control', 'shape': 'various', 'category': 'interactive'}, {'name': 'Multi-touch Gestures', 'type': 'attribute', 'shape': 'rectangular', 'category': 'informational'}, {'name': 'Haptic Feedback', 'type': 'attribute', 'shape': 'rectangular', 'category': 'informational'}]}]


def load_final_state(final_components, temperature, domain, requirements):
    state = {
        "graph_state": {
            "user_requirements": requirements,
            "final_components": final_components,
            "api_endpoint": os.environ.get("API_ENDPOINT"),
            "api_key": os.environ.get("API_KEY"),
            "temperature": temperature,
            "domain": domain,
        }
    }
    return state


# state = {
#     "graph_state": {
#         "final_components": final_components,
#         "api_endpoint": os.environ.get("API_ENDPOINT"),
#         "api_key": os.environ.get("API_KEY")
#     }
# }

# layout_generation(state)


def system_control_models(requirement, temperature, domain):
    system_flow = StateGraph(State)
    system_flow.add_node("RequirementsAnalysisAgent", high_to_lowlevel_requirements)
    system_flow.add_node("ComponentsAnalysisAgent", components_analysis)
    system_flow.add_node("ExternalInteligenceAgent", external_inteligence)
    system_flow.add_node("BreakFlow", break_flow)
    system_flow.add_node("RelationshipAgent", relationship_extraction)
    system_flow.add_node("ZoneComponentAnalysisAgent", zone_components_analysis)
    system_flow.add_node("ColorMapInteligenceAgent", map_type_with_color)
    system_flow.add_node("LayoutGenerationAgent", generate_layout)
    system_flow.add_node("OverLapsFixAgent", check_and_fix_zone_overlaps_layout)
    system_flow.add_node("RenderingAgent", render_layout)

    system_flow.add_edge(START, "RequirementsAnalysisAgent")
    system_flow.add_conditional_edges("RequirementsAnalysisAgent", decide_mood)
    system_flow.add_edge("ComponentsAnalysisAgent", "ExternalInteligenceAgent")
    system_flow.add_edge("ExternalInteligenceAgent", "RelationshipAgent")
    system_flow.add_edge("BreakFlow", END)
    system_flow.add_edge("RelationshipAgent", "ZoneComponentAnalysisAgent")
    system_flow.add_edge("ZoneComponentAnalysisAgent", "ColorMapInteligenceAgent")
    system_flow.add_edge("ColorMapInteligenceAgent", "LayoutGenerationAgent")
    system_flow.add_edge("LayoutGenerationAgent", "OverLapsFixAgent")
    system_flow.add_edge("OverLapsFixAgent", "RenderingAgent")
    system_flow.add_edge("RenderingAgent", END)

    graph = system_flow.compile()

    state = {
        "graph_state": {
            "user_requirements": requirement,
            "api_endpoint": os.environ.get("API_ENDPOINT"),
            "api_key": os.environ.get("API_KEY"),
            "temperature": temperature,
            "domain": domain,
        }
    }
    state = graph.invoke(state)
    return "success"

# def load_initial_state(requirement, temperature):
#     initial_state: json = {
#         "graph_state": {
#             "user_requirements": requirement,
#             "api_endpoint": os.environ.get("API_ENDPOINT"),
#             "api_key": os.environ.get("API_KEY"),
#             "temperature": temperature,
#             "domain": "Automotive HMI Infotainment layout generation",
#         }
#     }
#     return initial_state
