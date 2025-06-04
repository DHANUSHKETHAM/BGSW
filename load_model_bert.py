import torch
import json
import pandas as pd
import numpy as np
import random
from transformers import RobertaTokenizer, RobertaModel
from torch import nn
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from src.customLLM import CustomLLM
from langchain.agents import initialize_agent, AgentType
import re
import random
import numpy as np
load_dotenv()


class CustomRobertaModel(nn.Module):
    def __init__(
        self,
        num_shape_labels,
        num_dimension_labels,
        num_component_labels,
        num_zone_labels,
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.shape_classifier = nn.Linear(self.roberta.config.hidden_size, num_shape_labels)
        self.dimension_classifier = nn.Linear(self.roberta.config.hidden_size, num_dimension_labels)
        self.component_classifier = nn.Linear(self.roberta.config.hidden_size, num_component_labels)
        self.zone_classifier = nn.Linear(self.roberta.config.hidden_size, num_zone_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        return {
            "shape_logits": self.shape_classifier(hidden_state),
            "dimension_logits": self.dimension_classifier(hidden_state),
            "component_logits": self.component_classifier(hidden_state),
            "zone_logits": self.zone_classifier(hidden_state),
        }


import random

class WidgetPlacer:
    def __init__(self, canvas_width=680, canvas_height=378):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.placed_widgets = []

    def resize_widget_to_fit(self, width, height):
        max_width, max_height = min(300, self.canvas_width), min(300, self.canvas_height)
        return min(width, max_width), min(height, max_height)

    def greedy_placement(self, width, height, component_name, zone=None, spacing=10):
        # Use entire canvas if no zone
        if zone:
            zx, zy = zone['x'], zone['y']
            zw, zh = zone['width'], zone['height']
        else:
            zx, zy = 0, 0
            zw, zh = self.canvas_width, self.canvas_height

        # Try original size first
        for attempt in range(20):
            max_x = zx + zw - width
            max_y = zy + zh - height
            if max_x < zx or max_y < zy:
                break

            x = random.randint(zx, max_x)
            y = random.randint(zy, max_y)

            if not any(self.is_overlapping(x, y, width, height, *rect[:4], spacing) for rect in self.placed_widgets):
                self.placed_widgets.append((x, y, width, height, component_name))
                return x, y, width, height

        # Try scaled versions if placement failed
        for scale_attempt in range(1, 11):
            scale_factor = 1 - 0.05 * scale_attempt
            new_w = max(50, int(width * scale_factor))
            new_h = max(30, int(height * scale_factor))
            max_x = zx + zw - new_w
            max_y = zy + zh - new_h

            if max_x < zx or max_y < zy:
                continue

            x = random.randint(zx, max_x)
            y = random.randint(zy, max_y)

            if not any(self.is_overlapping(x, y, new_w, new_h, *rect[:4], spacing) for rect in self.placed_widgets):
                self.placed_widgets.append((x, y, new_w, new_h, component_name))
                return x, y, new_w, new_h

        return None, None, None, None

    def is_overlapping(self, x1, y1, w1, h1, x2, y2, w2, h2, spacing=10):
        return not (
            x1 + w1 + spacing <= x2 or
            x2 + w2 + spacing <= x1 or
            y1 + h1 + spacing <= y2 or
            y2 + h2 + spacing <= y1
        )


def format_requirements(requirements_text):
    matches = re.findall(r'\d+\.\s*(.+)', requirements_text)
    formatted_text = "; ".join(matches) + "." if matches else requirements_text
    return formatted_text.strip()


def bert_requirements(state):
    prompt_template = """
Your primary responsibility is to analyze the given requirements. Break down and organize the requirements widget-wise, assigning relevant requirements to each widget. Then, consolidate and refine the requirements under each widget, ensuring clarity and eliminating redundancy.
Output format:Provide a list of final, consolidated requirements, each separated by a semicolon (;).
Input format:requirements :{requirements}
    """
    prompt = PromptTemplate.from_template(template=prompt_template)
    formatted_prompt = prompt.format(
        requirements=state["graph_state"].get("user_requirements", "")
    )

    llm = CustomLLM(
        api_endpoint=state["graph_state"]["api_endpoint"],
        api_key=state["graph_state"]["api_key"],
        prompt=prompt,
        temperature=state["graph_state"]["temperature"],
    )

    agent = initialize_agent(
        tools=[],
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    response = agent.run(formatted_prompt)
    formatted_response = format_requirements(response)
    print("Formatted Requirements:", formatted_response)
    return formatted_response  # return clean text


# Load dataset
csv_path = "C:/Users/EDX1COB/Downloads/Final_zones.csv"
df = pd.read_csv(csv_path, encoding="latin1")
df.columns = df.columns.str.strip().str.replace("[^A-Za-z0-9_]+", "", regex=True)

# Label mappings
zone_label2id = {label: idx for idx, label in enumerate(df["probablezone"].unique())}
zone_id2label = {idx: label for label, idx in zone_label2id.items()}
shape_label2id = {label: idx for idx, label in enumerate(df["Shape"].unique())}
dimension_label2id = {label: idx for idx, label in enumerate(df["Dimension"].unique())}
component_label2id = {label: idx for idx, label in enumerate(df["Component"].unique())}
shape_id2label = {idx: label for label, idx in shape_label2id.items()}
dimension_id2label = {idx: label for label, idx in dimension_label2id.items()}
component_id2label = {idx: label for label, idx in component_label2id.items()}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "C:/roberta/robert_retrained.pth"
model = CustomRobertaModel(
    len(shape_label2id),
    len(dimension_label2id),
    len(component_label2id),
    len(zone_label2id),
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Tokenizer and placer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
widget_placer = WidgetPlacer()


def predict_and_place(requirement_text, model, widget_placer):
    inputs = tokenizer(
        requirement_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = model(**{key: val.to(device) for key, val in inputs.items()})

    shape = shape_id2label[torch.argmax(outputs["shape_logits"]).item()]
    dimension = dimension_id2label[torch.argmax(outputs["dimension_logits"]).item()]
    component = component_id2label[torch.argmax(outputs["component_logits"]).item()]
    probable_zone = zone_id2label[torch.argmax(outputs["zone_logits"]).item()]

    if "x" in str(dimension):
        width, height = map(int, dimension.split("x"))
    else:
        width, height = 100, 100

    width, height = widget_placer.resize_widget_to_fit(width, height)
    x, y, final_width, final_height = widget_placer.greedy_placement(width, height, component)

    return {
        "zone": probable_zone,
        "zone_size": {"width": width or 0, "height": height or 0},
        "position": {"x": x or 0, "y": y or 0},
        "components": [
            {
                "name": component,
                "shape": shape,
                "size": {"width": final_width or 0, "height": final_height or 0},
                "position": {"x": x or 0, "y": y or 0},
            }
        ],
    }

zone_location_pixels = [
    {
        "name": "header_zone",
        "zone_layout": "Single-Row Layout",
        "size": {"width": 680, "height": 57},
        "position": {"x": 0, "y": 321},
    },
    {
        "name": "navigation_zone",
        "zone_layout": "Single-Column Layout",
        "size": {"width": 68, "height": 284},
        "position": {"x": 0, "y": 38},
    },
    {
        "name": "main_content_zone",
        "zone_layout": "Multi-Column Layout with bootstrap",
        "size": {"width": 564, "height": 284},
        "position": {"x": 68, "y": 38},
    },
    {
        "name": "notification_zone",
        "zone_layout": "Single-Column Layout",
        "size": {"width": 48, "height": 284},
        "position": {"x": 632, "y": 38},
    },
    {
        "name": "footer_zone",
        "zone_layout": "Single-Row Layout",
        "size": {"width": 680, "height": 38},
        "position": {"x": 0, "y": 0},
    },
]

# Convert to dict for easier lookup
zone_fixed_config = {zone["name"]: zone for zone in zone_location_pixels}



import random

import random

def generate_json_output(requirements, model, widget_placer):
    predicted_layout = [predict_and_place(req, model, widget_placer) for req in requirements]
    zone_groups = {}

    # Group components by zones
    for entry in predicted_layout:
        zone = entry['zone']
        fixed_zone = zone_fixed_config.get(zone)
        if not fixed_zone:
            continue

        if zone not in zone_groups:
            zone_groups[zone] = {
                "zone_size": fixed_zone["size"],
                "position": fixed_zone["position"],
                "zone_type": zone,
                "components": []
            }

        zone_groups[zone]["components"].extend(entry["components"])

    rearranged_layout = []

    for zone, details in zone_groups.items():
        zone_width = details["zone_size"]["width"]
        zone_height = details["zone_size"]["height"]
        zone_x = details["position"]["x"]
        zone_y = details["position"]["y"]
        components = details["components"]
        spacing = 10

        final_placed_components = []

        def check_overlap(x, y, w, h):
            for other in final_placed_components:
                ox, oy = other["position"]["x"], other["position"]["y"]
                ow, oh = other["size"]["width"], other["size"]["height"]
                if not (x + w + spacing <= ox or
                        ox + ow + spacing <= x or
                        y + h + spacing <= oy or
                        oy + oh + spacing <= y):
                    return True
            return False

        for comp in components:
            raw_width = max(comp["size"].get("width", 100), 50)
            raw_height = max(comp["size"].get("height", 100), 30)
            placed = False

            # Try random placement and check overlap
            for _ in range(30):
                max_x = zone_x + zone_width - raw_width
                max_y = zone_y + zone_height - raw_height
                if max_x < zone_x or max_y < zone_y:
                    break

                rand_x = random.randint(zone_x, max_x)
                rand_y = random.randint(zone_y, max_y)

                norm_x = rand_x // 5
                norm_y = rand_y // 5
                norm_w = raw_width // 5
                norm_h = raw_height // 5

                if not check_overlap(norm_x, norm_y, norm_w, norm_h):
                    comp["size"] = {"width": norm_w, "height": norm_h}
                    comp["position"] = {"x": norm_x, "y": norm_y}
                    final_placed_components.append(comp)
                    placed = True
                    break

            # If not placed, retry scaled down
            if not placed:
                for scale_attempt in range(1, 11):
                    scale_factor = 1 - 0.05 * scale_attempt
                    new_width = max(50, int(raw_width * scale_factor))
                    new_height = max(30, int(raw_height * scale_factor))
                    max_x = zone_x + zone_width - new_width
                    max_y = zone_y + zone_height - new_height
                    if max_x < zone_x or max_y < zone_y:
                        continue

                    rand_x = random.randint(zone_x, max_x)
                    rand_y = random.randint(zone_y, max_y)

                    norm_x = rand_x // 5
                    norm_y = rand_y // 5
                    norm_w = new_width // 5
                    norm_h = new_height // 5

                    if not check_overlap(norm_x, norm_y, norm_w, norm_h):
                        comp["size"] = {"width": norm_w, "height": norm_h}
                        comp["position"] = {"x": norm_x, "y": norm_y}
                        final_placed_components.append(comp)
                        placed = True
                        break

            # Final fallback: try fixed position and nudge if needed
            if not placed:
                fallback_width = raw_width
                fallback_height = raw_height
                max_x = zone_x + zone_width - fallback_width
                max_y = zone_y + zone_height - fallback_height
                rand_x = random.randint(zone_x, max_x) if max_x >= zone_x else zone_x
                rand_y = random.randint(zone_y, max_y) if max_y >= zone_y else zone_y

                norm_x = rand_x // 5
                norm_y = rand_y // 5
                norm_w = fallback_width // 5
                norm_h = fallback_height // 5

                for dx in range(-10, 11, 5):
                    for dy in range(-10, 11, 5):
                        try_x = norm_x + dx
                        try_y = norm_y + dy
                        if try_x < 0 or try_y < 0:
                            continue
                        if not check_overlap(try_x, try_y, norm_w, norm_h):
                            comp["size"] = {"width": norm_w, "height": norm_h}
                            comp["position"] = {"x": try_x, "y": try_y}
                            final_placed_components.append(comp)
                            placed = True
                            break
                    if placed:
                        break

        rearranged_layout.append({
            "zone": zone,
            "zone_size": details["zone_size"],
            "position": details["position"],
            "components": final_placed_components
        })

    print('Arranged layout', rearranged_layout)
    return rearranged_layout



def process_requirements_and_predict(initial_state, model, widget_placer):
    formatted_requirements = bert_requirements(initial_state)
    requirements_list = [
        req.strip()
        for req in formatted_requirements.split(";")
        if req.strip()
    ]
    return generate_json_output(requirements_list, model, widget_placer)


def bert_json_layout(initial_state):
     
    layout_json = process_requirements_and_predict(initial_state, model, widget_placer)

    return layout_json


# Example usage
initial_state = {
    "graph_state": {
        "user_requirements": "the layout shall have G-Force",
        "api_endpoint": "https://ews-esz-emea.api.bosch.com/it/application/mixtralshared/prod/v1/chat/completions",
        "api_key": "115f37d4-77c9-48f6-a374-edece0803c34",
        "temperature": 0,
    }
}

layout_json = bert_json_layout(initial_state)
print(json.dumps(layout_json, indent=2))
