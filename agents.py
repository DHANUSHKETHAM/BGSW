from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import StructuredTool
from typing import Literal, List
from src.widgetPlacer import WidgetPlacer
from src.customLLM import CustomLLM
import src.prompts as pt

# from src.AzureOpenAI import AzureChatClient
from load_model_bert import bert_json_layout
from langchain.chat_models import AzureChatOpenAI
from PIL import Image
from src.helpers import *
from src.prompts import *
import pandas as pd
import tkinter as tk
import ast
import math
import json
import re
import sys
import copy


def generate_prompt_template(domain: str, agent_prompt: str) -> str:
    return (
        f"You are an expert UX/UI designer specializing in {domain} domain, "
        f"with a strong focus on creating clean, intuitive, and effective low-fidelity wireframes."
        + agent_prompt
    )


def planner_agent(state, agent_prompt):
    prompt_template = pt.planner_agent_prompt

    prompt = PromptTemplate.from_template(template=prompt_template)
    formated_prompt: str = prompt.format(
        domain=state["graph_state"]["domain"], agent_prompt=agent_prompt
    )

    llm = CustomLLM(
        api_endpoint=state["graph_state"]["api_endpoint"],
        api_key=state["graph_state"]["api_key"],
        prompt=prompt,
        temperature=state["graph_state"]["temperature"],
    )

    response = llm.invoke(formated_prompt)
    print(response)
    return response


def high_to_lowlevel_requirements(state):
    prompt_template = generate_prompt_template(
        state["graph_state"]["domain"], pt.high_to_lowlevel_requirements_prompt
    )
    # prompt_template: str = planner_agent(state, prompt_template)

    prompt = PromptTemplate.from_template(template=prompt_template)
    formated_prompt: str = prompt.format(
        requirements=state["graph_state"].get("user_requirements", "")
    )

    # llm = AzureChatOpenAI(
    #     azure_deployment="mydep",  # Name of the deployment in Azure OpenAI Studio
    #     azure_endpoint="https://mini-gpt4.openai.azure.com",
    #     api_key="35cWRcmzKF50fCA7CdU4vWJuE2XhwiiCGFRgMa5fVsCXKAahUbqFJQQJ99ALACYeBjFXJ3w3AAABACOGYxYV",
    #     api_version="2023-05-15",  # Replace with your version if different
    #     temperature=state["graph_state"].get("temperature", 0),
    # )
    llm = CustomLLM(
        api_endpoint=state["graph_state"]["api_endpoint"],
        api_key=state["graph_state"]["api_key"],
        prompt=prompt,
        temperature=state["graph_state"]["temperature"],
    )

    tools = []  # Add necessary tools here if needed

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    response = agent.run(formated_prompt)
    print("reqirement:", response)
    return {"graph_state": {**state["graph_state"], "requirements": response}}


def bert_requirements(state):
    prompt_template = generate_prompt_template(
        state["graph_state"]["domain"], pt.bert_requirements_prompt
    )
    prompt = PromptTemplate.from_template(template=prompt_template)
    formated_prompt: str = prompt.format(
        requirements=state["graph_state"].get("user_requirements", "")
    )

    llm = CustomLLM(
        api_endpoint=state["graph_state"]["api_endpoint"],
        api_key=state["graph_state"]["api_key"],
        prompt=prompt,
        temperature=state["graph_state"]["temperature"],
    )

    tools = []  # Add necessary tools here if needed

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    response = agent.run(formated_prompt)
    # print(response)
    return {"graph_state": {**state["graph_state"], "requirements": response}}


def decide_mood(state) -> Literal["ComponentsAnalysisAgent", "BreakFlow"]:
    unrelated_phrases = [
        "unrelated",
        "unclear",
        "not clear",
        "not related",
        "irrelevant",
        "no relation",
        "no connection",
        "off-topic",
        "out of context",
        "tangential",
        "extraneous",
        "ambiguous",
        "vague",
        "indistinct",
        "hazy",
        "obscure",
        "muddled",
        "perplexing",
        "immaterial",
        "insignificant",
        "not relevant",
        "not pertinent",
        "off the mark",
        "nonessential",
        "inconsequential",
        "out of place",
        "mismatched",
        "inconsistent",
        "divergent",
        "dissimilar",
        "unrelated matter",
        "not aligned",
        "not associated",
        "not linked",
        "not tied",
    ]

    for phrase in unrelated_phrases:
        if phrase in state["graph_state"]["requirements"]:
            print("Flow broken due to unclear requirements.")
            # print(phrase)
            return "BreakFlow"
        else:
            return "ComponentsAnalysisAgent"


def break_flow(state):
    print("Flow terminated due to unclear or unrelated requirements.")
    return state


def components_analysis(state):
    # Define individual component schema
    class Component(BaseModel):
        name: str = Field(description="Name of the component")
        type: str = Field(
            description="Type of the component (e.g., button, gauge, slider)"
        )
        shape: str = Field(description="Component shape (e.g., circular, rectangular)")
        category: Literal["Interactive", "Informational", "Navigation"] = Field(
            description="Must be one of the predefined category: Interactive, Informational,Navigation"
        )

    # Define a wrapper schema to handle multiple components
    class ComponentsList(BaseModel):
        components: list[Component]

    # Initialize parser
    parser = PydanticOutputParser(pydantic_object=ComponentsList)

    # Define prompt template
    prompt_template = PromptTemplate(
        template=generate_prompt_template(
            state["graph_state"]["domain"], pt.components_analysis_prompt
        ),
        input_variables=["requirements"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Format prompt
    formatted_prompt = prompt_template.format(
        requirements=state["graph_state"].get("requirements", "")
    )

    tools = []  # Add the tool to the list

    # Initialize the language model
    llm = CustomLLM(
        api_endpoint=state["graph_state"]["api_endpoint"],
        api_key=state["graph_state"]["api_key"],
        prompt=formatted_prompt,
        temperature=state["graph_state"]["temperature"],
    )

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    # Run the agent
    response = agent.run(formatted_prompt)
    response = extract_json_objects(response)

    # Iterate through the components and update the shape if it is None
    for component_group in response:
        for component in component_group["components"]:
            shape = component.get("shape")

            if isinstance(shape, list):
                shape = shape[0] if shape else ""
            if isinstance(shape, str) and shape.lower() in [
                "none",
                "",
                "null",
                "mixed",
                "n/a",
                "various",
                "not applicable",
                "Varied",
            ]:  # Check if shape is None
                component["shape"] = "rectangular"  # Update shape to 'rectangular

    print("component_analysis:", response)

    return {"graph_state": {**state["graph_state"], "components_analysis": response}}


def external_inteligence(state):
    class Component(BaseModel):
        name: str = Field(description="Name of the component")
        type: str = Field(
            description="Type of the component (e.g., button, gauge, slider)"
        )
        shape: str = Field(description="Component shape (e.g., circular, rectangular)")
        category: Literal["Interactive", "Informational", "Navigation"] = Field(
            description="Must be one of the predefined category: Interactive, Informational,Navigation"
        )

    # Define a wrapper schema to handle multiple components
    class ComponentsList(BaseModel):
        components: list[Component]

    # Initialize parser
    parser = PydanticOutputParser(pydantic_object=ComponentsList)

    # Read CSV dataset
    csv_path = r"data\Dataset_csv.csv"
    df = pd.read_csv(csv_path)

    # Extract component list
    component_list = df["component"].unique().tolist()[50]

    # Define prompt template
    prompt_template = PromptTemplate(
        template=generate_prompt_template(
            state["graph_state"]["domain"], pt.external_inteligence_prompt
        ),
        input_variables=["requirements", "component_list"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Format prompt
    formatted_prompt = prompt_template.format(
        requirements=state["graph_state"].get("requirements", ""),
        component_list=component_list,
    )

    # Define a simple tool
    def component_lookup_tool(query: str):
        """Search for a component in the dataset."""
        if isinstance(query, list):
            query = " ".join(map(str, query))
        elif not isinstance(query, str):
            query = str(query)
        matching_components = [
            comp
            for comp in component_list
            if isinstance(comp, str) and query.lower() in comp.lower()
        ]
        return (
            matching_components
            if matching_components
            else "No matching component found."
        )

    component_tool = Tool(
        name="Component Lookup",
        func=component_lookup_tool,
        description="Looks up components based on a query.",
    )

    tools = [component_tool]  # Add the tool to the list

    # Initialize the language model
    llm = CustomLLM(
        api_endpoint=state["graph_state"]["api_endpoint"],
        api_key=state["graph_state"]["api_key"],
        prompt=formatted_prompt,
        temperature=state["graph_state"]["temperature"],
    )

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    # Run the agent
    response = extract_json_objects(agent.run(formatted_prompt))

    return {"graph_state": {**state["graph_state"], "components_from_csv": response}}


def unique_components(state):
    # components_analysis,components_from_csv
    llm_components = state["graph_state"].get("components_analysis", "")
    csv_components = state["graph_state"].get("components_from_csv", "")

    if isinstance(llm_components, str):
        llm_components = json.loads(llm_components)

    if isinstance(csv_components, str):
        csv_components = json.loads(csv_components)  # Convert to JSON object

    if isinstance(llm_components, list) and len(llm_components) > 0:
        llm_components = llm_components[0].get(
            "components", []
        )  # Use .get() to avoid KeyError
    else:
        llm_components = []  # Default to empty list if L1 is empty

    if isinstance(csv_components, list) and len(csv_components) > 0:
        csv_components = csv_components[0].get("components", [])
    else:
        csv_components = []

    combined_components = llm_components + csv_components

    unique_components = {}
    for component in combined_components:
        key = component["name"].lower()
        unique_components[key] = component

    final_list = list(unique_components.values())
    return {"graph_state": {**state["graph_state"], "final_components": final_list}}


def relationship_extraction(state):
    print("final_components::::", state["graph_state"].get("final_components", ""))
    print(state)
    # Define individual component schema
    class Component(BaseModel):
        name: str = Field(description="Name of the component")
        type: str = Field(
            description="Type of the component (e.g., button, gauge, slider)"
        )
        shape: str = Field(description="Component shape (e.g., circular, rectangular)")
        category: str = Field(
            description="Category: Interactive, Informational, or Navigation"
        )
        related_to: List[str] = Field(description="list of components names")

    # Define a wrapper schema to handle multiple components
    class ComponentsList(BaseModel):
        components: list[Component]

    # Initialize parser
    parser = PydanticOutputParser(pydantic_object=ComponentsList)

    # Define prompt template
    prompt_template = PromptTemplate(
        template=generate_prompt_template(
            state["graph_state"]["domain"], pt.relationship_extraction_prompt
        ),
        input_variables=["final_components"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Format prompt
    formatted_prompt = prompt_template.format(
        final_components=state["graph_state"].get("final_components", "")
    )

    tools = []  # Add the tool to the list

    # Initialize the language model
    llm = CustomLLM(
        api_endpoint=state["graph_state"]["api_endpoint"],
        api_key=state["graph_state"]["api_key"],
        prompt=formatted_prompt,
        temperature=state["graph_state"]["temperature"],
    )

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    # Run the agent
    response = agent.run(formatted_prompt)
    response = extract_json_objects(response)

    print("relationship ::", response)

    return {"graph_state": {**state["graph_state"], "relationship_analysis": response}}


def zone_components_analysis(state):
    # Define individual component schema
    class Component(BaseModel):
        name: str = Field(description="Name of the component")
        type: str = Field(description="Type of the component")
        shape: str = Field(description="Component shape")
        category: str = Field(description="Category of component")
        related_to: List[str] = Field(description="list of components names")
        zones: Literal[
            "header_zone",
            "navigation_zone",
            "main_content_zone",
            "notification_zone",
            "auxiliary_zone",
            "footer_zone",
        ] = Field(description="Must be one of the predefined zones")

    # Define a wrapper schema to handle multiple components
    class ComponentsList(BaseModel):
        components: list[Component]

    # Initialize parser
    parser = PydanticOutputParser(pydantic_object=ComponentsList)

    prompt_template = generate_prompt_template(
        state["graph_state"]["domain"], pt.zone_components_analysis_prompt
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["relationship_analysis"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    formated_prompt: str = prompt.format(
        relationship_analysis=state["graph_state"].get("relationship_analysis", "")
    )

    llm = CustomLLM(
        api_endpoint=state["graph_state"]["api_endpoint"],
        api_key=state["graph_state"]["api_key"],
        prompt=prompt,
        temperature=state["graph_state"]["temperature"],
    )

    tools = []  # Add necessary tools here if needed

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
    )
    response = extract_json_objects(agent.run(formated_prompt))

    if isinstance(response, list) and all(
        isinstance(item, dict)
        and {"name", "type", "shape", "category", "related_to", "zones"}.issubset(
            item.keys()
        )
        for item in response
    ):
        formatted_output = [
            {
                "components": [
                    {
                        "name": item["name"],
                        "type": item["type"],
                        "shape": item["shape"],
                        "category": item["category"],
                        "related_to": item["related_to"],
                        "zones": item["zones"],
                    }
                    for item in response
                ]
            }
        ]
        json.dumps(formatted_output, indent=2)
        response = formatted_output

    print("zone_components_analysis:", response)

    return {
        "graph_state": {**state["graph_state"], "zone_components_analysis": response}
    }


zone_location = [
    {
        "name": "header_zone",
        "zone_layout": "Single-Row Layout",
        "size": {"width": 180, "height": 15},
        "position": {"x": 0, "y": 85},
    },
    {
        "name": "navigation_zone",
        "zone_layout": "Single-Column Layout",
        "size": {"width": 18, "height": 75},
        "position": {"x": 0, "y": 10},
    },
    {
        "name": "main_content_zone",
        "zone_layout": "Multi-Column Layout with bootstrap",
        "size": {"width": 131.4, "height": 75},
        "position": {"x": 18, "y": 10},
    },
    {
        "name": "notification_zone",
        "zone_layout": "Single-Column Layout",
        "size": {"width": 12.6, "height": 75},
        "position": {"x": 167.4, "y": 10},
    },
    {
        "name": "auxiliary_zone",
        "zone_layout": "Single-Column Layout",
        "size": {"width": 18, "height": 75},
        "position": {"x": 149.4, "y": 10},
    },
    {
        "name": "footer_zone",
        "zone_layout": "Single-Row Layout",
        "size": {"width": 180, "height": 10},
        "position": {"x": 0, "y": 0},
    },
]


def map_type_with_color(state):
    # csv_file_path = "data\widgets.csv"
    csv_file_path = "C:/Users/EDX1COB/Downloads/widgets.csv"
    try:
        # Load CSV data into a DataFrame
        df = pd.read_csv(csv_file_path)
        df["types"] = df["types"].apply(str.lower)

        # Convert JSON string to Python dictionary
        data = state["graph_state"]["zone_components_analysis"][0]
        # data = state['graph_state']['zone_components_analysis']

        # Prepare the output format
        output = {"components": []}

        # Iterate over components and update with color
        for component in data["components"]:
            if "type" not in component:
                print(f"Warning: Missing 'type' in component: {component}")
                continue
            c_type = component["type"]

            # Find matching color from DataFrame
            match = df[df["types"] == c_type]
            color = match.iloc[0]["color"] if not match.empty else "white"

            # Append formatted component to output
            output["components"].append(
                {
                    "name": component["name"],
                    "type": component["type"],
                    "shape": component["shape"],
                    "category": component["category"],
                    "related_to": component["related_to"],
                    "zones": component["zones"],
                    "color": color,
                }
            )

        # print("color:",output)
        return {
            "graph_state": {
                **state["graph_state"],
                "colour_components_analysis": json.dumps(output, indent=4),
            }
        }

    except Exception as e:
        print(f"Unexpected error in map_type_with_color: {e}")
        return state  # Return the original state if an unexpected error occurs


def generate_layout(state):
    try:

        class Position(BaseModel):
            x: int = Field(description="X-coordinate (integer only)")
            y: int = Field(description="Y-coordinate (integer only)")

        class Size(BaseModel):
            width: int = Field(description="Width of the zone")
            height: int = Field(description="Height of the zone")

        # Component Model
        class Component(BaseModel):
            name: str = Field(description="Name of the component")
            type: str = Field(
                description="Type of the component (e.g., button, gauge, slider)"
            )
            shape: str = Field(
                description="Component shape (e.g., circular, rectangular)"
            )
            category: str = Field(
                description="Category: Interactive, Informational, or Navigation"
            )
            related_to: List[str] = Field(description="list of components names")
            color: str = Field(description="Color of the component")
            size: Size = Field(
                description="Size of the component with width and height"
            )
            position: Position = Field(
                description="Position of the component with x and y coordinates"
            )

        # Zone Model
        class Zone(BaseModel):
            zone: str = Field(description="Name of the zone")
            zone_size: Size = Field(
                description="Size of the zone with width and height"
            )
            position: Position = Field(
                description="Position of the zone with x and y coordinates"
            )
            components: List[Component] = Field(
                description="List of components within this zone"
            )

        # Layout Model
        class Layout(BaseModel):
            layout: List[Zone] = Field(
                description="List of zones containing components"
            )

        parser = PydanticOutputParser(pydantic_object=Layout)

        try:
            component_list = json.loads(
                state["graph_state"]["colour_components_analysis"]
            )
        except KeyError as e:
            raise ValueError(f"Missing key in graph_state: {e}")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse 'colour_components_analysis' JSON.")

        main_content_components, header_footer_components, other_components = (
            get_sorted_components(component_list)
        )

        main_content_zone, header_footer_zone, other_zones = get_sorted_zones(
            zone_location
        )

        prompt_templates = {
            "prompt_template_1": {
                "template": generate_prompt_template(
                    state["graph_state"]["domain"], pt.generate_layout_prompt_1
                ),
                "variables": {
                    "other_components": other_components,
                    "other_zones": other_zones,
                },
            },
            "prompt_template_2": {
                "template": generate_prompt_template(
                    state["graph_state"]["domain"], pt.generate_layout_prompt_2
                ),
                "variables": {
                    "header_footer_components": header_footer_components,
                    "header_footer_zone": header_footer_zone,
                },
            },
            "prompt_template_3": {
                "template": generate_prompt_template(
                    state["graph_state"]["domain"], pt.generate_layout_prompt_3
                ),
                "variables": {
                    "main_content_components": main_content_components,
                    "main_content_zone": main_content_zone,
                },
            },
        }

        def run_prompt(template_name):
            try:
                prompt = PromptTemplate(
                    template=template_name["template"],
                    input_variables=list(template_name["variables"].keys()),
                    partial_variables={
                        "format_instructions": parser.get_format_instructions()
                    },
                )

                formatted_prompt = prompt.format(**template_name["variables"])

                llm = CustomLLM(
                    api_endpoint=state["graph_state"]["api_endpoint"],
                    api_key=state["graph_state"]["api_key"],
                    prompt=prompt,
                    temperature=state["graph_state"]["temperature"],
                )
                problem_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

                class CalculatorInput(BaseModel):
                    expression: str = Field(description="A math expression to evaluate")

                # Define the math tool function
                def math_tool_func(expression: str) -> str:
                    """
                    Custom function to evaluate a math expression using the LLMMathChain.
                    """
                    print(f"Raw LLM Output: {expression}")  # Print raw output

                    # Clean the markdown formatting
                    cleaned_expression = re.sub(
                        r"^```(?:text)?\s*([\s\S]+?)\s*```$", r"\1", expression
                    ).strip()

                    print(
                        f"Evaluating expression: {cleaned_expression}"
                    )  # Debugging output

                    # Try evaluating the expression using Python's eval (you can replace this with the intended chain)
                    try:
                        result = eval(cleaned_expression)
                        return str(result)
                    except Exception as e:
                        return f"Error evaluating expression: {e}"

                Calculator = StructuredTool.from_function(
                    name="Calculator",
                    func=math_tool_func,
                    description="""Useful for when you need to answer questions about math. 
                            This tool is only for math questions and nothing else. 
                            Only input math expressions.""",
                    args_schema=CalculatorInput,
                )
                tools = [Calculator]

                agent = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                    handle_parsing_errors=True,
                    verbose=True,
                )

                result = agent.invoke({"input": formatted_prompt})
                print("result['output'] :::::::", result["output"])
                try:
                    return result["output"]
                # Ensure the output is parsed as JSON
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse JSON output: {result['output']}")
            except Exception as e:
                raise RuntimeError(f"Error during LLM invocation: {e}")

        results = []
        for template in [
            prompt_templates["prompt_template_1"],
            prompt_templates["prompt_template_2"],
            prompt_templates["prompt_template_3"],
        ]:
            try:
                results.append(run_prompt(template))
            except Exception as e:
                raise RuntimeError(f"Error processing template {template}: {e}")

        # print("results[0] :: ",ast.literal_eval(extract_data_brackets(results[0])))
        # print("results[1] :: ",ast.literal_eval(extract_data_brackets(results[1])))
        # print("results[2] :: ",ast.literal_eval(extract_data_brackets(results[2])))

        # print("results[2] :: before", results[2])

        if not results[2]:
            raise ValueError("Error in parsing main content zone response")

        # Extract container size
        main_content_zone = next(
            zone for zone in zone_location if zone["name"] == "main_content_zone"
        )
        container_width = main_content_zone["size"]["width"]
        container_height = main_content_zone["size"]["height"]

        # Extract widget sizes with names
        try:
            widgets = [
                (comp["name"], (comp["size"]["width"], comp["size"]["height"]))
                for comp in results[2]["layout"][0]["components"]
            ]
        except KeyError as e:
            raise ValueError(f"Missing key in component layout: {e}")

        # Run widget placement
        placer = WidgetPlacer(
            int(container_width), int(container_height), widgets, min_spacing=1
        )
        placements = placer.run()

        for placement in placements:
            name, x, y, width, height = placement
            for component in results[2]["layout"][0]["components"]:
                if component["name"] == name:
                    component["position"]["x"] = x
                    component["position"]["y"] = y
                    component["size"]["width"] = width
                    component["size"]["height"] = height

        # print("results[2] :: after", results[2])
        response = append_all_lists(
            ast.literal_eval(extract_data_brackets(results[0])),
            ast.literal_eval(extract_data_brackets(results[1])),
            ast.literal_eval(extract_data_brackets(results[2])),
        )
        # print("response :::", response)

        response = {"layout": response}
        # print("response :::", response)

        def generate_repositioned_layouts(input_layout):
            """Generate two different layouts by repositioning components within each zone."""
            try:

                def reposition_components(zone):
                    zone_width, zone_height = (
                        zone["zone_size"]["width"],
                        zone["zone_size"]["height"],
                    )
                    components = zone["components"]
                    num_components = len(components)

                    if num_components == 0:
                        return zone

                    # Determine maximum component width and height
                    max_comp_width = min(comp["size"]["width"] for comp in components)
                    max_comp_height = min(comp["size"]["height"] for comp in components)

                    # Ensure at least 1 row
                    rows = max(
                        1, int(zone_height // (max_comp_height * 1.2))
                    )  # Add 20% spacing
                    cols = max(
                        1, (num_components + rows - 1) // rows
                    )  # Avoid zero-division error

                    x_spacing = max(
                        1, (zone_width - (cols * max_comp_width)) // (cols + 1)
                    )
                    y_spacing = max(
                        1, (zone_height - (rows * max_comp_height)) // (rows + 1)
                    )

                    x_offset, y_offset = x_spacing, y_spacing
                    count = 0

                    for r in range(rows):
                        for c in range(cols):
                            if count >= num_components:
                                break
                            comp = components[count]
                            comp["position"]["x"] = x_offset
                            comp["position"]["y"] = y_offset
                            x_offset += max_comp_width + x_spacing
                            count += 1
                        x_offset = x_spacing
                        y_offset += max_comp_height + y_spacing

                    return zone

                layout1, layout2 = (
                    copy.deepcopy(input_layout),
                    copy.deepcopy(input_layout),
                )
                for layout in (layout1, layout2):
                    for zone in layout["layout"]:
                        reposition_components(zone)

                return layout1, layout2
            except Exception as e:
                raise RuntimeError(f"Error in repositioning components: {e}")

        try:
            layout1, layout2 = generate_repositioned_layouts(response)
        except RuntimeError as e:
            raise ValueError(f"Error generating repositioned layouts: {e}")

        # print("layout1 ::", response)
        # print("layout2 ::", layout1)
        # print("layout3 ::", layout1)

        # bert_state = {
        #    "graph_state": {
        #        "user_requirements": state["graph_state"]["user_requirements"],
        #        "api_endpoint": state["graph_state"]["api_endpoint"],
        #        "api_key": state["graph_state"]["api_key"],
        #        "temperature": 0,
        #    }
        # }

        # layout_3 = bert_json_layout(bert_state)

        response = [
            {"layout": response["layout"]},
            {"layout": layout1["layout"]},
            # {"layout": layout_3},
        ]
        print("final_response ::", response)

        # prompt = PromptTemplate(template=ReAct_Prompt,
        #                     input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        #                     partial_variables={"format_instructions": parser.get_format_instructions()})

        # llm = CustomLLM(api_endpoint=state['graph_state']["api_endpoint"], api_key=state['graph_state']["api_key"])

        # problem_chain = LLMMathChain.from_llm(llm=llm)
        # math_tool = Tool.from_function(name="Calculator", func=problem_chain.run, description="""Useful for math-related queries.""")
        # tools = [math_tool]
        # tool_names = ", ".join([tool.name for tool in tools])

        # agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        # agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)

        # agent_call_1 = f""" You are an expert in designing low-fidelity infotainment HMI layouts.
        #             Your task is to intelligently position components within designated zone sizes while adhering to zone layouts.
        #             Follow these structured steps to ensure optimal placement of components:

        #             Input Data:
        #                 Components: {other_components} – A list of components that need to be positioned within specific zones.
        #                 Zone Information: {other_zones} – Includes the width, height, and layout details of each zone.

        #             Execution Steps:
        #                 1) Zone-Wise Component Listing:
        #                     - Identify all components assigned to each zone.
        #                 2) Component Positioning Within Zones:
        #                     - Analyze the zone’s size (width, height).
        #                     - Arrange components inside the zone based on their category and related_to attributes.
        #                     - Ensure all components within a zone have a uniform width, matching the zone width.
        #                 3) Repeat for All Zones:
        #                     - Ensure all zones follow the same structured approach.
        #         ### IMPORTANT:  Return ONLY a valid JSON response, without explanations or comments. ###
        #         Your goal is to generate a structured JSON output that effectively positions components with proper sizes and position, while adhering to the designated zone layouts."""

        # agent_call_2 = f""" You are an expert in designing low-fidelity infotainment HMI layouts.
        #             Your task is to intelligently position components within designated zone sizes while adhering to zone layouts.
        #             Follow these structured steps to ensure optimal placement of components:

        #             Input Data:
        #                 Components: {header_footer_components} – A list of components that need to be positioned within specific zones.
        #                 Zone Information: {header_footer_zone} – Includes the width, height, and layout details of each zone.

        #             Execution Steps:
        #                 1) Zone-Wise Component Listing:
        #                     - Identify all components assigned to each zone.
        #                 2) Component Positioning Within Zones:
        #                     - Analyze the zone’s size (width, height).
        #                     - Arrange components inside the zone based on their category and related_to attributes.
        #                     - Ensure all components within a zone have a uniform width, matching the zone height.
        #                 3) Repeat for All Zones:
        #                     - Ensure all zones follow the same structured approach.
        #         ### IMPORTANT:  Return ONLY a valid JSON response, without explanations or comments. ###
        #         Your goal is to generate a structured JSON output that effectively positions components with proper sizes and position, while adhering to the designated zone layouts."""

        # agent_call_3 = f""" You are an expert in designing low-fidelity infotainment HMI (Human-Machine Interface) layouts.
        #         Your task is to intelligently position components {main_content_components} within designated zones while adhering to zone layouts.

        #         Your process involves:
        #         Analyzing {main_content_zone} data sizes, and zone layouts data. consider only "main_content_zone"

        #         Rules for Component Placement in Main Content Zone only:
        #             - Ensure components dynamically adjust their size based on available zone sizes for clear visibility.
        #             - Arrange components to fit within the layout without overlapping with each other.
        #             - Optimize the layout for usability, balancing efficient space utilization with an intuitive arrangement.
        #         ### IMPORTANT:  Return ONLY a valid JSON response, without explanations or comments. ###
        #         Your goal is to generate a structured JSON output that effectively positions components with proper sizes and position, while adhering to the designated zone layouts."""

        # # async def invoke_agent(agent_call):
        # #     result = await asyncio.to_thread(agent_executor.invoke, {"input": agent_call, "tools": tool_names, "agent_scratchpad": ""})
        # #     return extract_json_objects(result['output'])

        # # async def main():
        # #     results = await asyncio.gather(
        # #         invoke_agent(agent_call_1),
        # #         invoke_agent(agent_call_2),
        # #         invoke_agent(agent_call_3)
        # #     )
        # #     result_1, result_2, result_3 = results
        # #     print(result_1, result_2, result_3)
        # #     combine_result = combine_layouts(result_1, result_2, result_3)
        # #     print("combine_result:", combine_result)

        # # response =asyncio.run(main())

        # result_1=agent_executor.invoke({"input": agent_call_1, "tools": tool_names, "agent_scratchpad": ""})
        # print(result_1['output'])
        # result_1=extract_json_from_text(result_1['output'])
        # print(result_1)

        # time.sleep(5)

        # result_2=agent_executor.invoke({"input": agent_call_2, "tools": tool_names, "agent_scratchpad": ""})
        # print(result_2['output'])
        # result_2=extract_json_from_text(result_2['output'])
        # print(result_2)

        # time.sleep(5)

        # result_3=agent_executor.invoke({"input": agent_call_3, "tools": tool_names, "agent_scratchpad": ""})
        # print(result_3['output'])
        # result_3=extract_json_from_text(result_3['output'])
        # print(result_3)

        # time.sleep(5)

        # response=combine_layouts(result_1,result_2,result_3)
        # print("combine_result :", response)

        return {
            "graph_state": {
                **state["graph_state"],
                "layouts": response,
            }
        }
    except Exception as e:
        print(f"Error during layout generation: {e}")
        return {"error": str(e)}


def check_and_fix_zone_overlaps_layout(state):
    fixed_layouts = {}
    layouts = state["graph_state"].get("layouts", [])

    for i, layout_dict in enumerate(layouts):  # Loop through layouts list
        layout_key = f"layout{i + 1}"
        layout_data = next(iter(layout_dict.values()), [])  # Extract the layout list

        if isinstance(layout_data, str):
            try:
                layout_data = json.loads(layout_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for {layout_key}: {e}")
                continue

        for zone in layout_data:
            zone_x, zone_y = zone["position"]["x"], zone["position"]["y"]
            zone_width, zone_height = (
                zone["zone_size"]["width"],
                zone["zone_size"]["height"],
            )
            zone_type = zone.get("zone_type", "")

            for component in zone.get("components", []):
                comp_x, comp_y = component["position"]["x"], component["position"]["y"]
                comp_width, comp_height = (
                    component["size"]["width"],
                    component["size"]["height"],
                )

                # Adjust width for specific zones
                if zone_type in [
                    "navigation_zone",
                    "notification_zone",
                    "auxiliary_zone",
                    "side_zone",
                ]:
                    component["size"]["width"] = zone_width
                    comp_width = (
                        zone_width  # Ensure absolute position calculations align
                    )

                # Adjust height for specific zones
                if zone_type in ["footer_zone", "header_zone"]:
                    component["size"]["height"] = zone_height
                    comp_height = (
                        zone_height  # Ensure absolute position calculations align
                    )

                # Ensure components in side zones have the same width
                if zone_type == "side_zone":
                    component["size"]["width"] = zone_width
                    comp_width = zone_width

                # Calculate absolute positions
                absolute_x, absolute_y = zone_x + comp_x, zone_y + comp_y
                component_right, component_bottom = (
                    absolute_x + comp_width,
                    absolute_y + comp_height,
                )

                # Calculate zone boundaries
                zone_right, zone_bottom = zone_x + zone_width, zone_y + zone_height

                # Check boundary violations and adjust size proportionally
                if component_right > zone_right or component_bottom > zone_bottom:
                    excess_width = max(0, component_right - zone_right)
                    excess_height = max(0, component_bottom - zone_bottom)

                    if excess_width > 0:
                        component["size"]["width"] -= excess_width
                    if excess_height > 0:
                        component["size"]["height"] -= excess_height
        # print(layout_data)
        fixed_layouts[layout_key] = layout_data

        bert_state = {
            "graph_state": {
                "user_requirements": state["graph_state"]["user_requirements"],
                "api_endpoint": state["graph_state"]["api_endpoint"],
                "api_key": state["graph_state"]["api_key"],
                "temperature": 0,
            }
        }
        print("bert_input", bert_state)
        print(state)
        layout_3 = bert_json_layout(bert_state)

    final_layouts = [
        {"layout": fixed_layouts["layout2"]},
        {"layout": fixed_layouts["layout1"]},
        {"layout": layout_3},
    ]

    # print("final_response::",fixed_layouts)

    return {"graph_state": {**state["graph_state"], "layouts_fixed": final_layouts}}


def render_layout(state, scale_factor=5):
    layout_data = state["graph_state"]["layouts_fixed"]

    def transform_layout(input_data):
        return [input_data]

    layout_data = transform_layout(layout_data)
    print("layouts in canvas :", layout_data)

    # Original screen dimensions
    original_screen_width = 180
    original_screen_height = 100

    # Scaled screen dimensions
    screen_width = original_screen_width * scale_factor
    screen_height = original_screen_height * scale_factor

    # Zones configuration (original dimensions)
    zones = zone_location

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Screen Layout Design (Scaled)")
    canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="white")
    canvas.pack()

    def validate_color(color):
        """Returns a valid Tkinter color or a default if invalid."""
        valid_colors = {
            "black",
            "white",
            "gray",
            "red",
            "blue",
            "green",
            "yellow",
            "cyan",
            "magenta",
            "darkgray",
            "darkred",
            "darkblue",
            "darkgreen",
        }
        return color if color in valid_colors else "gray"

    # Function to draw a zone
    def draw_zone(zone):
        x = zone["position"]["x"] * scale_factor
        y = (
            original_screen_height - zone["position"]["y"] - zone["size"]["height"]
        ) * scale_factor  # Flip y-axis
        width = zone["size"]["width"] * scale_factor
        height = zone["size"]["height"] * scale_factor
        name = zone["name"]

        canvas.create_rectangle(x, y, x + width, y + height, outline="gray")

    def save_canvas():
        # Save canvas as PostScript file
        canvas.postscript(
            file=r"layouts_images\canvas_output.ps",
            colormode="color",
            pagewidth=canvas.winfo_width() * 1.5,
            pageheight=canvas.winfo_height() * 1.5,
        )
        # Convert PostScript to PNG
        img = Image.open(r"layouts_images\canvas_output.ps")

        # img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        img.save(r"layouts_images\canvas_output.png", "png")
        sys.stdout.write(r"layouts_images\canvas_output.png")
        root.quit()

    # Function to draw components in zones
    def draw_components(layout):
        for item in layout["layout"]:
            zone_x = item["position"]["x"] * scale_factor
            zone_y = (
                original_screen_height
                - item["position"]["y"]
                - item["zone_size"]["height"]
            ) * scale_factor

            for component in item.get("components", []):
                comp_x = zone_x + (component["position"]["x"] * scale_factor)
                comp_y = zone_y + (component["position"]["y"] * scale_factor)
                comp_width = component["size"]["width"] * scale_factor
                comp_height = component["size"]["height"] * scale_factor

                component_type = component.get("type").strip().lower()
                text = component.get("name", "")
                comp_color = validate_color(component.get("color", "").strip().lower())

                # Check the type of the component first
                if component_type == "text":
                    canvas.create_rectangle(
                        comp_x,
                        comp_y,
                        comp_x + comp_width,
                        comp_y + comp_height,
                        outline="black",
                        width=2,
                    )
                elif component_type == "image":
                    img_path = r"C:\Users\JVQ2KOR\Downloads\Layout Generation\AI_Agent_layout\data\av5c8336583e291842624.png"
                    if img_path:
                        img = tk.PhotoImage(file=img_path)
                        subsample_x = max(1, img.width() // comp_width)
                        subsample_y = max(1, img.height() // comp_height)
                        img = img.subsample(int(subsample_x), int(subsample_y))
                        canvas.create_rectangle(
                            comp_x,
                            comp_y,
                            comp_x + comp_width,
                            comp_y + comp_height,
                            outline="black",
                        )
                        canvas.create_image(
                            comp_x + comp_width / 2,
                            comp_y + comp_height / 2,
                            anchor="center",
                            image=img,
                        )
                        canvas.image = img
                elif component_type in ["arc"]:
                    canvas.create_arc(
                        comp_x,
                        comp_y,
                        comp_x + comp_width,
                        comp_y + comp_height,
                        start=component.get("start_angle", 0),
                        extent=component.get("extent", 90),
                        fill=comp_color,
                    )
                elif component_type in ["warning"]:
                    canvas.create_polygon(
                        comp_x,
                        comp_y + comp_height,
                        comp_x + comp_width / 2,
                        comp_y,
                        comp_x + comp_width,
                        comp_y + comp_height,
                        outline="red",
                        fill=comp_color,
                    )
                else:
                    shape = component.get("shape").strip().lower()

                    if shape in [
                        "rectangular",
                        "bar",
                        "slider",
                        "toggle",
                        "key-based",
                        "rectangle",
                    ]:
                        canvas.create_rectangle(
                            comp_x,
                            comp_y,
                            comp_x + comp_width,
                            comp_y + comp_height,
                            outline="black",
                            fill=comp_color,
                        )
                    elif shape in [
                        "circular",
                        "round",
                        "oval",
                        "ellipse",
                        "semi-circular",
                        "radial",
                    ]:
                        canvas.create_oval(
                            comp_x,
                            comp_y,
                            comp_x + comp_width,
                            comp_y + comp_height,
                            outline="red",
                            fill=comp_color,
                        )
                    elif shape in ["triangle", "triangular"]:
                        canvas.create_polygon(
                            comp_x,
                            comp_y + comp_height,
                            comp_x + comp_width / 2,
                            comp_y,
                            comp_x + comp_width,
                            comp_y + comp_height,
                            outline="red",
                            fill=comp_color,
                        )
                    elif shape in ["line", "vertical", "horizontal", "linear"]:
                        canvas.create_oval(
                            comp_x,
                            comp_y,
                            comp_x + comp_width,
                            comp_y + comp_height,
                            outline="red",
                            fill=comp_color,
                        )
                        center_x = comp_x + comp_width / 2
                        center_y = comp_y + comp_height / 2
                        radius = min(comp_width, comp_height) / 2
                        end_x = center_x + radius * math.cos(math.radians(30))
                        end_y = center_y - radius * math.sin(math.radians(30))
                        canvas.create_line(
                            center_x, center_y, end_x, end_y, fill="black", width=2
                        )
                    elif shape in ["rounded-rectangle", "rounded rectangle", "rounded"]:
                        canvas.create_rectangle(
                            comp_x,
                            comp_y,
                            comp_x + comp_width,
                            comp_y + comp_height,
                            outline="black",
                            width=2,
                            fill=comp_color,
                        )
                    elif shape == "waveform-based":
                        for i in range(0, int(comp_width), 10):
                            canvas.create_line(
                                comp_x + i,
                                comp_y + (comp_height / 2),
                                comp_x + i + 5,
                                comp_y,
                                fill="black",
                            )
                            canvas.create_line(
                                comp_x + i + 5,
                                comp_y,
                                comp_x + i + 10,
                                comp_y + (comp_height / 2),
                                fill="black",
                            )
                    else:
                        canvas.create_rectangle(
                            comp_x,
                            comp_y,
                            comp_x + comp_width,
                            comp_y + comp_height,
                            outline="black",
                            fill="gray",
                        )

                # Adjust text inside the shape
                wrapped_text = "\n".join(
                    [
                        text[i : i + int(comp_width / 6)]
                        for i in range(0, len(text), max(1, int(comp_width / 6)))
                    ]
                )
                canvas.create_text(
                    comp_x + comp_width / 2,
                    comp_y + comp_height / 2,
                    text=wrapped_text,
                    font=("Arial", 10),
                    width=comp_width,
                    anchor="center",
                )

    # Draw all zones
    for zone in zones:
        draw_zone(zone)

    # Draw components
    for layout in layout_data:
        draw_components(layout)

    # save_canvas_as_image(layout)
    root.after(10, root.withdraw)
    root.after(500, save_canvas)

    # Run the Tkinter event loop
    root.mainloop()
