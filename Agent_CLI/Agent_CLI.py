#!/usr/bin/env python-real

import re
import json
import os
import argparse
import logging
from Agent_CLI_utils.utils import (load_manifest, build_tool_spec, extract_parameters, build_cli_args,
                                   complete_with_defaults, chat_with_auto_pull,
                                   cross_encoder_retrieve_candidates, get_router_model)
from Agent_CLI_utils.parameter_extraction_improved import ImprovedParameterExtractor
from Agent_CLI_utils.parameter_validator import ParameterValidator
import traceback


logger = logging.getLogger(__name__)


def build_candidates_block(candidates, include_parameters=False):
    """
    Build a compact, one-line-per-tool block describing the candidate tools for
    the router prompt. When `include_parameters` is True, each line also lists
    the tool's parameters (used by the consultant/"ask" mode).
    """
    lines = []
    for i, s in enumerate(candidates, 1):
        name = s.get("name", "")
        desc = (s.get("description", "") or "").strip()
        desc = re.sub(r"\s+", " ", desc)
        desc = desc[:140]  # Keep the description short.
        tags = s.get("tags") or []
        tags = [str(x) for x in tags[:8]]  # At most 8 tags.
        tag_str = ", ".join(tags)
        line = f"{i}) {name} — {desc} | tags: {tag_str}"
        if include_parameters:
            line += f"| parameters:{s.get('parameters', '')}"
        lines.append(line)
    return "\n".join(lines)


def main(input):
    module_dir = os.path.dirname(__file__)

    try:
        _run(input, module_dir)
    except Exception as e:
        # Print the full traceback to stderr (visible in Slicer's CLI module
        # log / Python console) so the actual failure point is debuggable,
        # while stdout always stays valid, parseable JSON - Agent_UI.py
        # relies on json.loads(output) and would otherwise crash on an
        # empty/non-JSON stdout.
        traceback.print_exc()
        output = {
            "tool": None,
            "tool_confidence": None,
            "parameters_confidence": None,
            "parameters": None,
            "missing_required": None,
            "command": None,
            "error": f"{type(e).__name__}: {e}",
        }
        print(json.dumps(output))


def _run(input, module_dir):
    # Packaged Slicer installs may stage RESOURCES under a Resources/
    # subfolder instead of next to the script, so check both.
    manifest_path = os.path.join(module_dir, "manifest.yaml")
    if not os.path.isfile(manifest_path):
        manifest_path = os.path.join(module_dir, "Resources", "manifest.yaml")
    manifest = load_manifest(manifest_path)
    tools = build_tool_spec(manifest)
    improved_extractor = ImprovedParameterExtractor(manifest_path)
    parameter_validator = ParameterValidator(manifest_path)

    # Earlier turns of this conversation (list of {"role","content"}), so the
    # model isn't limited to the latest message - lets e.g. a follow-up that
    # only supplies a previously-missing parameter still make sense.
    try:
        history = json.loads(getattr(input, "history", "") or "[]")
        if not isinstance(history, list):
            history = []
    except (json.JSONDecodeError, TypeError):
        history = []

    if input.modeagent == "Agent (Automated)":

        folder_list_str = "\n".join(input.folders.split(","))

        folders_context = f"""
        FOLDERS_CONTEXT:
        {folder_list_str}

        Rules:
        - Treat FOLDERS_CONTEXT as the source of truth for paths.
        - If a required path parameter is missing, try to infer it from FOLDERS_CONTEXT.
        - If multiple candidates exist, pick the most specific match and explain briefly in 'reason'.
        """.strip()

        modele = get_router_model()

        candidates = cross_encoder_retrieve_candidates(manifest, input.prompt, k=3)
        candidates_block = build_candidates_block(candidates)

        router_system = """
You are a tool router. Output ONLY one JSON object.
Schema:
{"tool": string|null, "confidence": number, "reason": string}

Rules:
- Choose tool ONLY from the candidate list provided by the user message.
- If none match, set tool = null and confidence <= 0.4.
- Do not invent tool names.
- Keep reason short (max 1 sentence).
""".strip()

        router_user = f"""
USER_REQUEST:
{input.prompt}

CANDIDATES (choose exactly one name from this list):
{candidates_block}
""".strip()

        try:
            response = chat_with_auto_pull(
                modele,
                messages=[
                    {"role": "system", "content": router_system},
                    *history,
                    {"role": "user", "content": router_user}
                ],
                format="json"
            )
        except Exception as e:
            raise RuntimeError(f"Router error: {e}")

        try:
            data = json.loads(response["message"]["content"])
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise RuntimeError(f"Router returned invalid JSON: {e}")
        selected_tool = data.get("tool")
        confidence = float(data.get("confidence", 0.0))

        if selected_tool and selected_tool!="null" and selected_tool!="None":
            new_prompt = f"""{input.prompt}\n\n{folders_context}"""

            params, param_conf, missing_required,removed = extract_parameters(manifest, selected_tool, new_prompt,improved_extractor,parameter_validator,modele,history)

            params = complete_with_defaults(manifest,selected_tool,params)

            for removed_name in removed:
                params[removed_name] = input.temp_folder
                if removed_name in missing_required:
                    missing_required.remove(removed_name)

            cli_args = build_cli_args(selected_tool, params, manifest, module_dir)
            
            output = {
                "tool": selected_tool,
                "tool_confidence": round(confidence, 3),
                "parameters_confidence": round(param_conf, 3) if params else 0.0,
                "parameters": params,
                "missing_required": missing_required,
                "command": cli_args,
            }
            print(json.dumps(output))

        else:

            output = {
                "tool": None,
                "tool_confidence": None,
                "parameters_confidence": None,
                "parameters": None,
                "missing_required": None,
                "command": None,
            }
            print(json.dumps(output))

    else:
        candidates = cross_encoder_retrieve_candidates(manifest, input.prompt, k=3)
        candidates_block = build_candidates_block(candidates, include_parameters=True)
        modele = get_router_model()

        system_prompt = f"""You are an expert medical image analysis consultant specializing in dental and orthodontic imaging.

Your role is to provide methodology advice and workflow recommendations for image analysis projects.

Available Tools and their purposes:
{candidates_block}

When answering questions:
1. Recommend the most appropriate tools from the available set
2. Explain the recommended workflow order
3. Provide reasoning for your recommendations
4. Consider preprocessing requirements
5. Mention any important parameters or settings

Keep your response focused and practical."""
        
        try:
            response = chat_with_auto_pull(
                modele,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *history,
                    {"role": "user", "content": input.prompt}
                ]
            )
        except Exception as e:
            raise RuntimeError(f"Router error: {e}")

        output = response["message"]["content"].strip()
        output = output.replace("*","")
        output = output.replace("#","")
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('prompt',type=str)
    parser.add_argument('folders',type=str)
    parser.add_argument('modeagent',type=str)
    parser.add_argument('temp_folder',type=str)
    parser.add_argument('history',type=str, nargs='?', default='[]')

    try:
        args = parser.parse_args()
    except SystemExit:
        print("Argument parsing error: check the number of arguments passed to the CLI.")
        raise
    main(args)
