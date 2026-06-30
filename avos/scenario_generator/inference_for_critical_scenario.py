import argparse
import base64
import json
import numpy as np
import os
import pickle
import random
import re
import torch

from PIL import Image

from config_paths import set_optimized_path
from inference.common import (
    build_generated_scenario,
    infer_scenes,
    load_config,
    load_model,
    make_dataloader,
    reconstruct_generated_scenario,
    require_cached_samples,
    resolve_device,
    save_scenario,
)
from inference.defaults import (
    config_path_value,
    config_value,
    default_config_path,
    default_infer_config_path,
    default_weight_path,
    load_infer_config,
)

ROAD_TYPE_OPTIONS = {
    0: "other",
    1: "intersection",
    2: "intersection approach",
    3: "road segment",
}

EGO_LANE_TYPE_OPTIONS = {
    0: "dead end",
    1: "U-turn",
    2: "U-turn + left",
    3: "U-turn + left + straight",
    4: "U-turn + left + straight + right",
    5: "left",
    6: "left + straight",
    7: "straight",
    8: "left + right",
    9: "left + straight + right",
    10: "straight + right",
    11: "right",
    12: "split right",
}

SIGNAL_STATE_OPTIONS = {
    0: "all green",
    1: "all red",
    2: "all yellow",
    3: "all flashing",
    4: "all unknown",
}


def input_option(title, options):
    print(f"{title} options:")
    for code, label in options.items():
        print(f"  {code}: {label}")
    value = int(input(f"Enter {title} code: "))
    if value not in options:
        raise ValueError(f"{title} must be one of {sorted(options)}.")
    return value


def input_distance_range():
    print("Distance range code options:")
    print("  0: unknown/no signal")
    print("  1: 0-10m")
    print("  2: 10-20m")
    print("  3: 20-30m")
    print("  n: (n-1)*10 to n*10m")
    return int(input("Enter Distance range code: "))


def distance_range_label(dist_range):
    if dist_range == 0:
        return "unknown/no signal"
    return f"{dist_range * 10 - 10}-{dist_range * 10} meters"


def get_map_code():
    print('\n')

    road_type = input_option("Road type", ROAD_TYPE_OPTIONS)
    num_lanes = int(input("The Number of Total Lanes (Motor + Non-Motor): "))
    if type(num_lanes) != int:
        raise TypeError("Input error. Please enter Arabic numerals (e.g., 2)")
    ego_lane_type = input_option("Ego lane type", EGO_LANE_TYPE_OPTIONS)
    global ego_lane_id
    ego_lane_id = input(f"Ego lane ID (from 1 to {num_lanes}): ")
    bike_lane_id = input(f"Bike lane ID (from 1 to {num_lanes} or None): ")
    if ego_lane_id == bike_lane_id:
        raise ValueError("Ego lane id cannot be equal to bike lane id")

    dist_range = input_distance_range()

    state_color = input_option("Ego light state", SIGNAL_STATE_OPTIONS)

    print(f"\nTotal lanes: {num_lanes}, "
          f"Ego lane type: {ego_lane_type} ({EGO_LANE_TYPE_OPTIONS[ego_lane_type]}), "
          f"Ego lane ID: {ego_lane_id}, "
          f"Bike lane ID: {bike_lane_id}, "
          f"Ego car to intersection distance range: {distance_range_label(dist_range)}, "
          f"Ego light state: {SIGNAL_STATE_OPTIONS[state_color]}, "
          f"Road type: {road_type} ({ROAD_TYPE_OPTIONS[road_type]})")

    lanes = [2] * num_lanes
    lanes[int(ego_lane_id)-1] = 1
    if bike_lane_id != 'None':
        lanes[int(bike_lane_id)-1] = 3

    map_code = [lanes + [dist_range] + [2] * 5 + [state_color] * 5 + [road_type], [ego_lane_type]]

    return map_code


def map_matching(map_codes, map_code_):
    matched_scenario_ids = []
    for scenario_id in map_codes:
        map_code_1 = map_codes[scenario_id][0]
        map_code_2 = map_codes[scenario_id][1]
        if map_code_1 == map_code_[0] and map_code_2[int(ego_lane_id)-1] == map_code_[1][0]:
            matched_scenario_ids.append(scenario_id)

    return matched_scenario_ids


def load_map_codes(map_code_path, subset_name):
    candidates = [
        os.path.join(map_code_path, 'map_codes.pkl'),
        os.path.join(map_code_path, subset_name, 'map_codes.pkl'),
    ]
    map_code_file = next((path for path in candidates if os.path.exists(path)), candidates[0])
    with open(map_code_file, 'rb') as file:
        return pickle.load(file)


def agent_input_dir(args, scenario_id):
    return os.path.join(args.input_path3, args.subset_name, scenario_id)


def ensure_llm_image(args, scenario_id):
    output_dir = agent_input_dir(args, scenario_id)
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, 'llm_image.png')
    if not os.path.exists(image_path):
        Image.new("RGB", (768, 768), color=(128, 128, 128)).save(image_path)
    return image_path


def save_default_agent_inputs(data, args, scenario_id):
    output_dir = agent_input_dir(args, scenario_id)
    os.makedirs(output_dir, exist_ok=True)
    ensure_llm_image(args, scenario_id)

    save_payload = {
        "scenario_id": scenario_id,
        "timestamp": str(data['extra_information']['timestamp']),
        "agents": [
            {
                "type": "ego",
                "x": 0.0,
                "y": 0.0,
                "heading_degrees": 0.0,
                "heading_radians": 0.0,
            }
        ],
    }
    with open(os.path.join(output_dir, 'edited_agents.json'), 'w', encoding='utf-8') as file:
        json.dump(save_payload, file, indent=4)

    return save_payload["agents"]


def load_agent_inputs(args, scenario_id):
    agent_json = os.path.join(agent_input_dir(args, scenario_id), 'edited_agents.json')
    if not os.path.exists(agent_json):
        return None
    with open(agent_json, 'r', encoding='utf-8') as file:
        payload = json.load(file)
    return payload.get('agents')


def load_mllm(text_inp, text_inp_, model=None):
    if args.mock_mllm_response:
        return args.mock_mllm_response
    if args.mock_mllm_response_path:
        with open(args.mock_mllm_response_path, 'r', encoding='utf-8') as file:
            return file.read()

    image_path = os.path.join(args.input_path3, args.subset_name, selected_scenario_id, 'llm_image.png')
    text_inp_ = json.dumps(text_inp_, ensure_ascii=False, indent=2)
    api_key = args.mllm_api_key or os.getenv("MLLM_API_KEY")
    if not api_key:
        raise RuntimeError("MLLM_API_KEY is not set. Use --mllm-api-key or export MLLM_API_KEY.")

    with open(args.system_prompt_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    image_inp = encode_image(image_path)

    if model == 'Gemini':
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options={"base_url": args.gemini_base_url}
        )

        image_inp = Image.open(image_path)

        response = client.models.generate_content(
            model=args.gemini_model,
            config=types.GenerateContentConfig(
                system_instruction=prompt,
            ),
            contents=[text_inp, text_inp_, image_inp]
        )

        return response.text

    elif model == 'Claude':
        from anthropic import Anthropic

        client = Anthropic(
            api_key=api_key,
            base_url=args.claude_base_url
        )

        message = client.messages.create(
            model=args.claude_model,
            max_tokens=2048,
            system=prompt,
            messages=[{"role": "user", "content": [
                           {"type": "text", "text": text_inp},
                           {"type": "text", "text": text_inp_},
                           {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_inp}}
                     ]}]
        )

        return next(block.text for block in message.content if block.type == "text")

    elif model == 'GPT':
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=args.openai_base_url
        )

        response = client.responses.create(
            model=args.gpt_model,
            input=[{"role": "system", "content": prompt},
                   {"role": "user", "content": [
                       {"type": "input_text", "text": text_inp},
                       {"type": "input_text", "text": text_inp_},
                       {"type": "input_image", "image_url": f"data:image/png;base64,{image_inp}", "detail": "low"}
                   ]}]
        )

        return response.output_text

    else:
        raise ValueError(f"{model} is not supported")


def main(argv=None):
    global args, selected_scenario_id

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--infer-config', '--infer_config', dest='infer_config', default=default_infer_config_path())
    known_args, _ = base_parser.parse_known_args(argv)
    infer_config = load_infer_config(known_args.infer_config)

    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument('--input-path1', '--input_path1', dest='input_path1', type=str,
                        default=config_path_value(infer_config, "DATA", "MAP_CODE_PATH", default="data/inference_data/map_codes"))
    parser.add_argument('--input-path2', '--input_path2', dest='input_path2', type=str,
                        default=config_path_value(
                            infer_config,
                            "DATA",
                            "OPTIMIZED_PATH",
                            default=config_value(infer_config, "DATA", "SOURCE_PATH", default="datasets/optimized_dataset"),
                        ))
    parser.add_argument('--input-path3', '--input_path3', dest='input_path3', type=str, default=None,
                        help='Directory that contains edited_agents.json. Defaults to output_path1.')
    parser.add_argument('--input-path4', '--input_path4', dest='input_path4', type=str,
                        default=config_path_value(infer_config, "CRITICAL", "PROMPT_DIR", default="data/inference_data/prompt/critical_scenario"))
    parser.add_argument('--config-path', type=str, default=config_path_value(infer_config, "CONFIG_PATH", default=default_config_path()))
    parser.add_argument('--weight-path', type=str, default=config_path_value(infer_config, "WEIGHT_PATH", default=default_weight_path()))
    parser.add_argument('--device', type=str, default=config_value(infer_config, "DEVICE", default=None))
    parser.add_argument('--subset-name', '--subset_name', dest='subset_name', type=str,
                        default=config_value(infer_config, "SUBSET_NAME", default='waymo-open'))
    parser.add_argument('--prompt-name', '--prompt_name', dest='prompt_name', type=str,
                        default=config_value(infer_config, "CRITICAL", "PROMPT_NAME", default='A.6.12'))
    parser.add_argument('--output-path1', '--output_path1', dest='output_path1', type=str,
                        default=config_path_value(infer_config, "OUTPUT", "SCENARIO_PATH", default="datasets/generated_dataset"))
    parser.add_argument('--reconstructor', type=str,
                        default=config_value(
                            infer_config,
                            "CRITICAL",
                            "RECONSTRUCTOR",
                            default=config_value(infer_config, "RECONSTRUCTOR", default="motion_reconstructor_v2"),
                        ),
                        help='Trajectory reconstructor: motion_reconstructor_v2.')
    parser.add_argument('--num-generations', type=int,
                        default=config_value(infer_config, "CRITICAL", "NUM_GENERATIONS", default=1))
    parser.add_argument('--interpreter-models', type=str,
                        default=config_value(infer_config, "CRITICAL", "INTERPRETER_MODELS", default='GPT'),
                        help='Comma-separated provider names: Gemini,Claude,GPT')
    parser.add_argument('--mllm-api-key', type=str, default=None)
    parser.add_argument('--system-prompt-path', type=str,
                        default=config_path_value(infer_config, "CRITICAL", "SYSTEM_PROMPT_PATH", default="data/inference_data/prompt/mllm/mllm.prompt"))
    parser.add_argument('--gemini-base-url', type=str,
                        default=config_value(infer_config, "MLLM", "GEMINI_BASE_URL", default='https://llm-api.mmchat.xyz/gemini'))
    parser.add_argument('--claude-base-url', type=str,
                        default=config_value(infer_config, "MLLM", "CLAUDE_BASE_URL", default='https://llm-api.mmchat.xyz'))
    parser.add_argument('--openai-base-url', type=str,
                        default=config_value(infer_config, "MLLM", "OPENAI_BASE_URL", default='https://llm-api.mmchat.xyz/v1'))
    parser.add_argument('--gemini-model', type=str,
                        default=config_value(infer_config, "MLLM", "GEMINI_MODEL", default='gemini-3.1-pro-preview'))
    parser.add_argument('--claude-model', type=str,
                        default=config_value(infer_config, "MLLM", "CLAUDE_MODEL", default='claude-opus-4-6'))
    parser.add_argument('--gpt-model', type=str,
                        default=config_value(infer_config, "MLLM", "GPT_MODEL", default='gpt-5.4'))
    parser.add_argument('--mock-mllm-response-path', type=str, default=None,
                        help='Read an MLLM response from a local file instead of calling a remote model.')
    parser.add_argument('--mock-mllm-response', type=str, default=None,
                        help='Use this literal MLLM response instead of calling a remote model.')
    args = parser.parse_args(argv)
    if args.input_path3 is None:
        args.input_path3 = args.output_path1

    map_codes = load_map_codes(args.input_path1, args.subset_name)

    with open(os.path.join(args.input_path4, args.prompt_name) + '.prompt', 'r', encoding='utf-8') as file:
        scenario_description = file.read()

    print(scenario_description)

    map_code_ = get_map_code()
    matched_scenario_ids = map_matching(map_codes, map_code_)
    matched_scenario_ids = [
        scenario_id for scenario_id in matched_scenario_ids
        if os.path.exists(os.path.join(args.input_path2, args.subset_name, scenario_id + '.pkl'))
    ]
    if matched_scenario_ids:
        selected_scenario_id = random.choice(matched_scenario_ids)
    else:
        raise ValueError("No matching scenario found")

    with open(os.path.join(args.input_path2, args.subset_name, selected_scenario_id+'.pkl'), 'rb') as file:
        data = pickle.load(file)

    agent_init = load_agent_inputs(args, selected_scenario_id)
    if agent_init is None:
        agent_init = save_default_agent_inputs(data, args, selected_scenario_id)
    else:
        ensure_llm_image(args, selected_scenario_id)
    print(agent_init)

    config = load_config(args.config_path)
    config.setdefault("DATASET", {})
    set_optimized_path(config, os.path.join(args.input_path2, args.subset_name))
    require_cached_samples(config, [selected_scenario_id + '.pkl'])
    device = resolve_device(args.device)
    model = load_model(config, args.weight_path, device)
    val_loader = make_dataloader(
        config,
        mode='val',
        batch_size=1,
        shuffle=False,
        data_list=[selected_scenario_id + '.pkl'],
    )
    interpreter_models = [item.strip() for item in args.interpreter_models.split(',') if item.strip()]

    for idx in range(1, args.num_generations + 1):
        for interpreter_model in interpreter_models:
            # MLLM-powered Interpret
            response_text = load_mllm(scenario_description, agent_init, model=interpreter_model)   # model='Gemini'/'Claude'/'GPT'
            print(f"\n{response_text}\n")

            pattern = r"-- agent \d+: (\[[\d,\s]+\])"
            matches = re.findall(pattern, response_text)
            object_code_1 = [json.loads(m) for m in matches]

            with open(file=os.path.join(args.input_path3, args.subset_name, selected_scenario_id, 'edited_agents.json'), mode='r', encoding='utf-8') as file:
                data = json.load(file)

            object_info = data['agents']

            object_code_2 = np.array([[object_info[i]['x'], object_info[i]['y'], object_info[i]['heading_radians']] for i in range(len(object_info))])
            object_pose = np.array([[object_info[i]['x'] / 50,
                                     object_info[i]['y'] / 50,
                                     np.sin(object_info[i]['heading_radians']),
                                     np.cos(object_info[i]['heading_radians'])] for i in range(len(object_info))])

            MAX_OBJECT_NUM = 96
            MAX_CODE_DIM = 17
            MAX_POSE_DIM = 4
            MAX_CODE_SPLIT_DIM = 13

            object_code_arr = np.zeros((MAX_OBJECT_NUM, MAX_CODE_DIM))
            object_pose_arr = np.zeros((MAX_OBJECT_NUM, MAX_POSE_DIM))
            object_code_split_arr = np.zeros((MAX_OBJECT_NUM, MAX_CODE_SPLIT_DIM))
            object_mask = np.zeros((MAX_OBJECT_NUM,), dtype=bool)

            object_code_1_np = np.array(object_code_1)
            object_code_arr[0:len(object_info)] = np.concatenate((object_code_1_np[:, 0][:, np.newaxis], object_pose, object_code_1_np[:, 1:]), axis=-1)
            object_pose_arr[0:len(object_info)] = object_pose
            object_code_split_arr[0:len(object_info)] = object_code_1_np
            object_mask[0:len(object_info)] = True

            with torch.no_grad():
                batch = next(iter(val_loader))

                batch['object'][:, 0:len(object_code_2), :2] = torch.tensor(object_code_2)[:, :2]
                batch['object'][:, 0:len(object_code_2), 6] = torch.tensor(object_code_2)[:, 2]
                batch['object_code'] = torch.tensor(object_code_arr, dtype=torch.float64)[None, ...]
                batch['object_pose'] = torch.tensor(object_pose_arr, dtype=torch.float64)[None, ...]
                batch['object_code_split'] = torch.tensor(object_code_split_arr, dtype=torch.float64)[None, ...]
                batch['object_mask'] = torch.tensor(object_mask, dtype=torch.bool)[None, ...]

                scene = infer_scenes(config, model, batch, device, add_labels=False)

                with open(os.path.join(args.input_path2, args.subset_name, selected_scenario_id) + '.pkl', 'rb') as file:
                    data = pickle.load(file)

                scenario = build_generated_scenario(scene[0], data, recompute_kinematics=False)
                scenario = reconstruct_generated_scenario(
                    scenario,
                    reconstructor_method=args.reconstructor,
                )
                output_scenario_id = selected_scenario_id + '-' + interpreter_model + '-' + str(idx)
                output_file = save_scenario(scenario, args.output_path1, args.subset_name, output_scenario_id)
                print(f"Generated critical scenario: {output_file}")
