"""The VISOR server's schema translation, carried by the extension for LOCAL runs.

With a server, a panel is built from what `GET /tools` publishes, and the server
turns the panel's values back into what a tool's `run()` takes. Locally there is
no server: the raw schema comes straight from SADT-VISOR's `scripts/describe.py`
(run with the tool's own interpreter), and this module does both halves of what
the server would have done with it:

    panel_schema(name, raw)            the entry GET /tools publishes for a tool
    facade_schema(name, raws)          the same for a facade (ALI, AREG)
    resolve_facade(name, values)       which tool a facade run actually runs
    wire_arguments(schema, values, data_dir, tool)
                                       the panel's values as run() takes them
    data_slug(tool_name, data_dir)     the DATA folder a tool's files live in

**This is a deliberate COPY of server code**, accepted so the extension runs with
no server at all. Every rule names the server file it was copied from (in the
VISOR-serve repository, under server/), and
`ServerToolsCore/Testing/Python/test_local_schema.py` holds it to golden
fixtures produced by the real server's registry: when the server's translation
changes, regenerate those fixtures and this file must follow.

Left out ON PURPOSE, because a local run has no use for them:

* `batch` and `paired_batch` (how a client splits a cohort into several
  requests): a local run takes the whole cohort in one call.
* `stop_after` and `keep_intermediate` (the arguments the server injects into
  every tool that calls another): a local run has no checkpoints to stop at and
  resume from, and no job directory to collect intermediate files in.

Standard library only, Python 3.9 compatible, and no slicer/qt/requests import,
so it is testable in plain CI. No TOML is parsed at run time (tomllib is 3.11+):
the deployment overrides that matter are copied below as Python literals.
"""

import copy
import logging
import os
import re

logger = logging.getLogger(__name__)


class LocalSchemaError(ValueError):
    """A raw schema this translation cannot publish, or panel values it cannot
    hand to run(). The server answers the first by not serving the tool and the
    second with a 422."""


# ---------------------------------------------------------------------------
# Constants copied from the server
# ---------------------------------------------------------------------------

# server/registry/conventions.py: MODEL_NAMES, WEIGHTS_NAME, OUTPUT_DIR_ARGUMENT,
# TECHNICAL, the section names, _OUTPUT_TOKENS, _EXPANSIONS, _CODE_TOKEN.
MODEL_NAMES = ("model", "reference")
WEIGHTS_NAME = "model"
OUTPUT_DIR_ARGUMENT = "output_dir"
TECHNICAL = frozenset({
    "device",
    "gpu_resampling",
    "tile_step_size",
    "num_workers",
    "n_workers",
    "batch_size",
    "threads",
    "search_steps",
    "seed",
    "max_triplets",
    "surface_smoothing",
    "surface_decimation",
})
SECTION_INPUTS = "Inputs"
SECTION_MODEL = "Model"
SECTION_OPTIONS = "Options"
SECTION_OUTPUTS = "Outputs"
SECTION_ADVANCED = "Advanced"
_OUTPUT_TOKENS = frozenset({"output", "outputs", "suffix", "prediction", "naming"})
_EXPANSIONS = {"num": "number of", "nb": "number of", "max": "maximum", "min": "minimum"}
_CODE_TOKEN = re.compile(r"^[a-z]{1,3}\d{1,2}$")

# server/registry/deployment.py: SERVER_SELECTABLE_NONE, SCOPE_SEPARATOR.
SERVER_SELECTABLE_MODEL = "model"
SERVER_SELECTABLE_TESTFILE = "testfile"
SERVER_SELECTABLE_NONE = "none"
SCOPE_SEPARATOR = ":"

# server/data_store.py and server/execution/dispatch.py: DATA/<slug>/{models,testfiles}.
MODELS_DIRNAME = "models"
TESTFILES_DIRNAME = "testfiles"

# server/base.py: the published type names.
PATH_TYPE = "path"
CHOICE_TYPE = "choice"
MULTICHOICE_TYPE = "multichoice"
LIST_TYPE = "list[str]"
VEC2_TYPE = "vec2"

# server/registry/schema_tool.py: ARGUMENT_TYPES, as the type NAME GET /tools
# publishes (main.py `_type_name`). Every list type is published as "list[str]".
ARGUMENT_TYPES = {
    "path": PATH_TYPE,
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "list[str]": LIST_TYPE,
    "list[int]": LIST_TYPE,
    "list[float]": LIST_TYPE,
    "list[bool]": LIST_TYPE,
    "list[path]": LIST_TYPE,
    "vec2": VEC2_TYPE,
}
MULTICHOICE_BASE_TYPES = ("list[str]",)

# server/registry/schema_tool.py: RETURN_KINDS, DEFAULT_RETURN_KIND.
RETURN_KINDS = {
    "path": "files",
    "dict[str, path]": "files",
    "paths": "files",
    "text": "text",
    "json": "text",
}
DEFAULT_RETURN_KIND = "text"

# server/registry/schema_tool.py: _PRESENTATION_KEYS.
_PRESENTATION_KEYS = ("section", "ui", "groups", "visible_when", "options_when", "label",
                      "option_help", "note", "min_selected", "select_all", "required_when",
                      "x_labels", "y_labels", "x_label", "y_label",
                      "section_columns", "cell")

# server/registry/facade.py: MODE_ARGUMENT and the mode argument's wording.
MODE_ARGUMENT = "mode"
_MODE_DESCRIPTION = "Which of this tool's engines to run."
_MODE_LABEL = "Mode"

# server/deployment.toml, the [tools.X] entries that change what is published or
# how a run is wired: `server_selectable` (written as in the file, "kind" or
# "kind:scope"), `hidden`, `data_dir` and `dispatch`. The rest of that file
# (width_from, cores_per_channel, timeout_seconds, batch, max_upload_mb) is
# about admission and transfers on a shared server and means nothing locally.
DEPLOYMENT = {
    "AREG_CBCT": {
        "data_dir": "AREG",
        "server_selectable": {
            "t1_masks": "none", "segmentation_model": "none", "landmark_model": "none",
            "t1": "testfile:CBCT_T1", "t2": "testfile:CBCT_T2", "patients": "none",
        },
    },
    "AREG_IOS": {
        "data_dir": "AREG",
        "server_selectable": {
            "t1": "testfile:IOS_T1", "t2": "testfile:IOS_T2", "crown_model": "none",
            "registration_model": "none", "mgl_model": "none", "ios_reference": "none",
            "patients": "none",
        },
    },
    "AREG_IOSCBCT": {
        "data_dir": "AREG",
        "server_selectable": {
            "ios": "testfile:IOSCBCT_T1", "cbct": "testfile:IOSCBCT_T2",
            "ios_landmarks": "testfile:IOSCBCT_IOS Landmarks",
            "cbct_landmarks": "testfile:IOSCBCT_CBCT Landmarks",
            "crown_model": "none", "ios_landmark_model": "none", "landmark_model": "none",
            "cbct_reference": "none", "patients": "none",
        },
    },
    "ALI_CBCT": {"data_dir": "ALI"},
    "ALI_IOS": {"data_dir": "ALI"},
    "AMASSS": {
        "server_selectable": {"model": "none"},
        "hidden": ["model"],
    },
    "AREG": {
        "dispatch": {"CBCT": "AREG_CBCT", "IOS": "AREG_IOS", "IOS to CBCT": "AREG_IOSCBCT"},
    },
    "ASO": {"server_selectable": {"landmark_model": "none"}},
    "ALI": {"dispatch": {"CBCT": "ALI_CBCT", "IOS": "ALI_IOS"}},
    "AutoMatrix": {"server_selectable": {"reference": "none"}},
    "FlexReg": {"server_selectable": {"surfaces": "testfile", "reference": "testfile"}},
    "VFACE": {
        "data_dir": "VFACE",
        "server_selectable": {
            "t1": "testfile:CBCT_Or_FullyAuto", "t2": "none", "segmentation_model": "none",
            "landmark_model": "none", "classifier_model": "none", "surface_model": "none",
            "cranial_base_reference": "none", "maxilla_reference": "none",
            "mirror_reference": "none", "measurements": "none", "feature_template": "none",
            "registration_transforms": "none",
        },
    },
    "GreedyReg": {
        "server_selectable": {"t1": "testfile:CBCT_T1_T2_pair", "t2": "testfile:CBCT_T1_T2_pair"},
    },
}

# {facade name: {mode label: target tool}}, in the order the modes are offered.
# Iterating it gives the facade names; see facade_targets().
FACADES = {
    name: dict(entry["dispatch"]) for name, entry in DEPLOYMENT.items() if entry.get("dispatch")
}


# ---------------------------------------------------------------------------
# Naming conventions (server/registry/conventions.py)
# ---------------------------------------------------------------------------

def is_model(argument_name: str) -> bool:
    """`model`, `*_model`, `reference`, `*_reference`: hosted, named, never uploaded."""
    return any(argument_name == name or argument_name.endswith("_" + name)
               for name in MODEL_NAMES)


def names_weights(argument_name: str) -> bool:
    """`model` or `*_model`: weights, as opposed to a reference scan."""
    return argument_name == WEIGHTS_NAME or argument_name.endswith("_" + WEIGHTS_NAME)


def section_for(argument_name: str, declaration: dict) -> str:
    """The section an argument lands in when its schema declares none."""
    if is_model(argument_name):
        return SECTION_MODEL
    if argument_name in TECHNICAL:
        return SECTION_ADVANCED
    tokens = set(argument_name.lower().split("_"))
    if tokens & _OUTPUT_TOKENS:
        return SECTION_OUTPUTS
    if isinstance(declaration, dict) and declaration.get("type") == "path":
        return SECTION_INPUTS
    if isinstance(declaration, dict) and declaration.get("required"):
        return SECTION_INPUTS
    return SECTION_OPTIONS


def label_for(argument_name: str) -> str:
    """The argument name written out, sentence case, abbreviations spelled out."""
    words = []
    for index, token in enumerate(argument_name.split("_")):
        if not token:
            continue
        lowered = token.lower()
        if lowered in _EXPANSIONS and index == 0:
            words.append(_EXPANSIONS[lowered].capitalize())
        elif lowered in _EXPANSIONS:
            words.append(_EXPANSIONS[lowered])
        elif token.isupper():
            words.append(token)
        elif _CODE_TOKEN.match(lowered):
            words.append(token.upper())
        elif not words:
            words.append(lowered.capitalize())
        else:
            words.append(lowered)
    return " ".join(words) or argument_name


def _declared(tool_name: str) -> dict:
    """What DEPLOYMENT says about one tool, with `server_selectable` split into
    kinds and scopes (server/registry/deployment.py `_tool_deployment`)."""
    entry = DEPLOYMENT.get(tool_name, {})
    kinds, scopes = {}, {}
    for argument, value in entry.get("server_selectable", {}).items():
        kind, _, scope = str(value).partition(SCOPE_SEPARATOR)
        kinds[argument] = kind
        if scope:
            scopes[argument] = scope
    return {
        "server_selectable": kinds,
        "selectable_scopes": scopes,
        "hidden": tuple(entry.get("hidden", ())),
        "data_dir": entry.get("data_dir"),
    }


def _derive(tool_name: str, arguments: dict) -> dict:
    """The deployment merged over the conventions (conventions.derive), for the
    three fields that shape the panel: server_selectable, its scopes, hidden."""
    declared = _declared(tool_name)
    selectable = {}
    for name, declaration in arguments.items():
        if not isinstance(declaration, dict) or declaration.get("type") != "path":
            continue
        selectable[name] = SERVER_SELECTABLE_MODEL if is_model(name) else SERVER_SELECTABLE_TESTFILE
    for name, kind in declared["server_selectable"].items():
        declaration = arguments.get(name)
        # The server refuses to load a tool whose deployment entry names an
        # argument it does not declare, or marks a non-path one selectable
        # (schema_tool.SchemaTool / _argument_spec). Locally that would take a
        # working tool off the panel over a stale override, so it is ignored.
        if not isinstance(declaration, dict) or (
                kind != SERVER_SELECTABLE_NONE and declaration.get("type") != "path"):
            logger.warning("%s: ignoring the deployment override for '%s'.", tool_name, name)
            continue
        selectable[name] = kind
    opted_out = [name for name, kind in selectable.items() if kind == SERVER_SELECTABLE_NONE]
    for name in opted_out:
        del selectable[name]

    hidden = {name for name in arguments if name in TECHNICAL}
    hidden.update(declared["hidden"])
    hidden.update(name for name in opted_out if names_weights(name))
    return {
        "server_selectable": selectable,
        "selectable_scopes": declared["selectable_scopes"],
        "hidden": hidden,
    }


# ---------------------------------------------------------------------------
# One argument (server/registry/schema_tool.py `_argument_spec`, `_choice_spec`,
# `_presentation`, then server/main.py `list_tools` for the wire shape)
# ---------------------------------------------------------------------------

def _presentation(declaration: dict, argument_name: str) -> dict:
    hints = {key: declaration[key] for key in _PRESENTATION_KEYS
             if declaration.get(key) is not None}
    hints.setdefault("section", section_for(argument_name, declaration))
    hints.setdefault("label", label_for(argument_name))
    for axis in ("x_range", "y_range"):
        bounds = declaration.get(axis)
        if bounds is None:
            continue
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise LocalSchemaError("{!r} of '{}' must be two numbers.".format(axis, argument_name))
        try:
            hints[axis] = [float(bound) for bound in bounds]
        except (TypeError, ValueError):
            raise LocalSchemaError("{!r} of '{}' must be two numbers.".format(axis, argument_name))
    return hints


def _published(type_name, required, description, hints, server_selectable=None,
               selectable_scope=None, choices=None, initial=None, accepts=None,
               hidden=False) -> dict:
    """One argument as GET /tools writes it, keys in the server's order."""
    extensions = None
    if type_name == PATH_TYPE:
        # main.py `_extensions_of`: FILE_TYPES["path"] is None, `accepts` wins.
        extensions = {PATH_TYPE: list(accepts) if accepts else None}

    def listed(key):
        value = hints.get(key)
        return list(value) if value else None

    groups = hints.get("groups")
    entry = {
        "type": type_name,
        "types": [type_name],
        "required": required,
        "description": description,
        "server_selectable": server_selectable,
        "selectable_scope": selectable_scope,
        "choices": choices,
        "initial": initial,
        "extensions": extensions,
        "label": hints.get("label"),
        "section": hints.get("section"),
        "visible_when": hints.get("visible_when"),
    }
    if hints.get("required_when"):
        entry["required_when"] = hints["required_when"]
    entry.update({
        "options_when": hints.get("options_when"),
        "hidden": hidden,
        "ui": hints.get("ui"),
        "x_range": listed("x_range"),
        "y_range": listed("y_range"),
        "x_labels": listed("x_labels"),
        "section_columns": hints.get("section_columns"),
        "cell": hints.get("cell"),
        "x_label": hints.get("x_label"),
        "y_label": hints.get("y_label"),
        "y_labels": listed("y_labels"),
        "groups": {name: list(options) for name, options in groups.items()} if groups else None,
    })
    # Published only when set, exactly as main.py does.
    for key in ("groups_when", "option_help", "note", "option_kind", "min_selected"):
        if hints.get(key):
            entry[key] = hints[key]
    if hints.get("select_all"):
        entry["select_all"] = True
    return entry


def _argument(tool_name: str, argument_name: str, declaration, derived: dict) -> dict:
    where = "Tool '{}', argument '{}'".format(tool_name, argument_name)
    if not isinstance(declaration, dict):
        raise LocalSchemaError(where + ": expected an object with at least a 'type'.")
    declared_type = declaration.get("type")
    if declared_type not in ARGUMENT_TYPES:
        raise LocalSchemaError("{}: unknown type {!r}.".format(where, declared_type))
    required = declaration.get("required", True)
    if not isinstance(required, bool):
        raise LocalSchemaError(where + ": 'required' must be true or false.")

    selectable = derived["server_selectable"].get(argument_name)
    scope = derived["selectable_scopes"].get(argument_name)
    hidden = argument_name in derived["hidden"] or bool(declaration.get("hidden"))
    description = declaration.get("description", "")

    choices = declaration.get("choices")
    if choices and isinstance(choices, list) and all(isinstance(o, str) for o in choices):
        default = declaration.get("default")
        if declared_type in MULTICHOICE_BASE_TYPES:
            selected = default if isinstance(default, list) else []
            return _published(MULTICHOICE_TYPE, declaration.get("required", True), description,
                              _presentation(declaration, argument_name),
                              choices={option: option in selected for option in choices},
                              hidden=hidden)
        chosen = default if default in choices else choices[0]
        return _published(CHOICE_TYPE, declaration.get("required", True), description,
                          _presentation(declaration, argument_name),
                          choices={option: option == chosen for option in choices},
                          hidden=hidden)
    # Non-string options (Literal[1, 2]) fall through to the plain type, as on
    # the server.

    if selectable == SERVER_SELECTABLE_MODEL and declared_type == "path":
        # A model is published as a NAME (type str), never as a file to upload.
        return _published("str", required and not hidden, description,
                          _presentation(declaration, argument_name),
                          server_selectable=selectable, selectable_scope=scope, hidden=hidden)

    accepts = declaration.get("extensions")
    if accepts is not None and declared_type != "path":
        raise LocalSchemaError(where + ": 'extensions' is only meaningful on a path.")
    return _published(ARGUMENT_TYPES[declared_type], required, description,
                      _presentation(declaration, argument_name),
                      server_selectable=selectable, selectable_scope=scope,
                      # Advisory pre-fill; dropped for a path.
                      initial=None if declared_type == "path" else declaration.get("default"),
                      accepts=tuple(accepts) if accepts else None,
                      hidden=hidden)


# ---------------------------------------------------------------------------
# Public: the published schema
# ---------------------------------------------------------------------------

def panel_schema(name: str, raw: dict) -> dict:
    """The entry GET /tools publishes for tool `name`, from describe.py's raw
    schema: `{"name", "arguments", "output_kind"}`. Without `batch`,
    `paired_batch`, `stop_after` and `keep_intermediate` (module docstring)."""
    if not isinstance(raw, dict):
        raise LocalSchemaError("{}: the schema is not an object.".format(name))
    arguments = raw.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise LocalSchemaError("{}: 'arguments' must be an object.".format(name))
    derived = _derive(name, arguments)
    published = {
        argument_name: _argument(name, argument_name, declaration, derived)
        for argument_name, declaration in arguments.items()
        if argument_name != OUTPUT_DIR_ARGUMENT
    }
    return {
        "name": name,
        "arguments": published,
        "output_kind": RETURN_KINDS.get(raw.get("returns"), DEFAULT_RETURN_KIND),
    }


def facade_targets(name: str) -> list:
    """The tools facade `name` dispatches to, in mode order; [] for a plain tool."""
    return list(FACADES.get(name, {}).values())


def _accepts_of(spec: dict) -> tuple:
    extensions = spec.get("extensions") or {}
    return tuple(extensions.get(PATH_TYPE) or ())


def facade_schema(name: str, raws: dict) -> dict:
    """The published schema of facade `name`, composed from its targets' raw
    schemas `{target: raw}` (server/registry/facade.py `compose`): a `mode`
    choice first, then every target's arguments, each shown only for the modes
    that have it.

    Only the targets present in `raws` become modes. The server instead refuses
    to publish a facade at all unless every target is served; locally, a
    machine with one engine installed still gets the facade, with one mode.
    """
    targets = {mode: target for mode, target in FACADES.get(name, {}).items() if target in raws}
    if not targets:
        raise LocalSchemaError("{} is not a facade over any installed tool.".format(name))
    schemas = {target: panel_schema(target, raws[target]) for target in targets.values()}
    modes = list(targets)

    arguments, seen_in, identity = {}, {}, {}
    choices_by_mode, groups_by_mode, help_by_mode, scope_by_mode = {}, {}, {}, {}
    for mode, target in targets.items():
        for argument_name, spec in schemas[target]["arguments"].items():
            here = (spec["type"], _accepts_of(spec), spec["required"])
            if argument_name in identity and identity[argument_name] != here:
                raise LocalSchemaError(
                    "'{}' cannot publish '{}': two of its engines declare it differently."
                    .format(name, argument_name))
            identity[argument_name] = here
            seen_in.setdefault(argument_name, []).append(mode)
            if argument_name not in arguments:
                arguments[argument_name] = copy.deepcopy(spec)
            if spec.get("choices"):
                choices_by_mode.setdefault(argument_name, {})[mode] = dict(spec["choices"])
            if spec.get("groups"):
                groups_by_mode.setdefault(argument_name, {})[mode] = dict(spec["groups"])
            if spec.get("option_help"):
                help_by_mode.setdefault(argument_name, {})[mode] = dict(spec["option_help"])
            scope_by_mode.setdefault(argument_name, {})[mode] = spec.get("selectable_scope")

    for argument_name, spec in arguments.items():
        appears_in = seen_in[argument_name]
        if len(appears_in) < len(modes):
            spec["visible_when"] = dict(spec.get("visible_when") or {})
            spec["visible_when"][MODE_ARGUMENT] = list(appears_in)
            spec["required"] = False
        per_mode = choices_by_mode.get(argument_name, {})
        if len({tuple(sorted(options)) for options in per_mode.values()}) > 1:
            spec["options_when"] = {
                MODE_ARGUMENT: {mode: list(options) for mode, options in per_mode.items()}
            }
            # The union of the options, the FIRST mode's default winning.
            merged = {}
            for options in per_mode.values():
                for option, selected in options.items():
                    merged.setdefault(option, selected)
            if spec["type"] == CHOICE_TYPE:
                seen_default = False
                for option, selected in list(merged.items()):
                    if selected and not seen_default:
                        seen_default = True
                    elif selected:
                        merged[option] = False
            spec["choices"] = merged
        per_mode_help = help_by_mode.get(argument_name, {})
        if per_mode_help:
            merged_help = {}
            for texts in per_mode_help.values():
                merged_help.update(texts)
            spec["option_help"] = merged_help
        # AREG's `t1` draws from CBCT_T1 in one mode and IOS_T1 in another: the
        # first engine's scope alone offered CBCT test files in IOS mode.
        per_mode_scope = scope_by_mode.get(argument_name, {})
        if len(set(per_mode_scope.values())) > 1:
            spec["selectable_scope_when"] = {MODE_ARGUMENT: dict(per_mode_scope)}
        per_mode_groups = groups_by_mode.get(argument_name, {})
        if len({tuple(sorted(groups)) for groups in per_mode_groups.values()}) > 1:
            spec["groups_when"] = {
                MODE_ARGUMENT: {mode: dict(groups) for mode, groups in per_mode_groups.items()}
            }

    # The mode argument carries no section and no label convention: it is
    # built directly, not from a schema.
    composed = {MODE_ARGUMENT: _published(
        CHOICE_TYPE, True, _MODE_DESCRIPTION, {"label": _MODE_LABEL},
        choices={mode: index == 0 for index, mode in enumerate(modes)})}
    composed.update(arguments)

    kinds = {schemas[target]["output_kind"] for target in targets.values()}
    return {
        "name": name,
        "arguments": composed,
        "output_kind": kinds.pop() if len(kinds) == 1 else "files",
    }


# ---------------------------------------------------------------------------
# Public: running
# ---------------------------------------------------------------------------

def resolve_facade(name: str, values: dict) -> tuple:
    """`(tool to run, values without "mode")` for a facade (server/main.py,
    `_run_tool`); `(name, values)` unchanged for a plain tool."""
    targets = FACADES.get(name)
    values = dict(values or {})
    if not targets:
        return name, values
    chosen = values.pop(MODE_ARGUMENT, None)
    if not chosen:
        raise LocalSchemaError("Missing required argument '{}' for '{}'. Choose one of: {}."
                               .format(MODE_ARGUMENT, name, ", ".join(sorted(targets))))
    if str(chosen) not in targets:
        raise LocalSchemaError("'{}' is not a mode of {}. Choose one of: {}."
                               .format(chosen, name, ", ".join(sorted(targets))))
    return targets[str(chosen)], values


def data_slug(tool_name: str, data_dir: str) -> str:
    """The folder under DATA holding `tool_name`'s models and test files
    (server/registry/deployment.py `data_slug`): the deployment's `data_dir`,
    else the literal name if DATA/<name> exists, else the name without
    underscores (`Batch_Dental_Seg` -> `BatchDentalSeg`)."""
    declared = DEPLOYMENT.get(tool_name, {}).get("data_dir")
    if declared:
        return declared
    if os.path.isdir(os.path.join(data_dir, tool_name)):
        return tool_name
    return tool_name.replace("_", "")


def _empty(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _hosted_path(data_dir, slug, kind_dir, scope, value, argument_name) -> str:
    """DATA/<slug>/<models|testfiles>/[<scope>/]<name> (server/data_store.py
    `_resolve`). An absolute path is a file the panel already has locally and
    is kept as it is."""
    if os.path.isabs(value):
        return value
    if os.path.basename(value) != value or value.startswith("."):
        raise LocalSchemaError("Invalid file name for '{}': '{}'".format(argument_name, value))
    parts = [data_dir, slug, kind_dir] + ([scope] if scope else []) + [value]
    path = os.path.join(*parts)
    if not os.path.exists(path):
        raise LocalSchemaError("'{}': no such {} in {}.".format(
            argument_name, kind_dir[:-1], os.path.join(*parts[:-1])))
    return path


def _as_bool(name, value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
        return value.lower() in ("true", "1")
    raise LocalSchemaError("Argument '{}' must be a boolean".format(name))


def _multichoice(name, value, options) -> list:
    """base.Tool._coerce_multichoice then SchemaTool.for_the_wire: whatever
    arrives is the complete selection; run() gets the ticked options, in
    declaration order."""
    if isinstance(value, dict):
        provided = value
    elif isinstance(value, (list, tuple, set)):
        provided = {str(option): True for option in value}
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            import json
            try:
                provided = json.loads(text)
            except ValueError as error:
                raise LocalSchemaError("Argument '{}': invalid JSON ({})".format(name, error))
            if not isinstance(provided, dict):
                raise LocalSchemaError("Argument '{}': expected an option -> true/false object"
                                       .format(name))
        else:
            provided = {part.strip(): True for part in text.split(",") if part.strip()}
    else:
        raise LocalSchemaError("Argument '{}': expected an option -> true/false mapping"
                               .format(name))
    unknown = [option for option in provided if option not in options]
    if unknown:
        raise LocalSchemaError("Argument '{}': unknown option(s) {}.".format(
            name, ", ".join(sorted(unknown))))
    return [option for option in options
            if option in provided and _as_bool(name, provided[option])]


def _coerce(name, value, spec, options):
    """base.Tool._coerce for every published non-file type."""
    declared = spec["type"]
    if declared == CHOICE_TYPE:
        if not isinstance(value, str) or value not in options:
            raise LocalSchemaError("Argument '{}': unknown option '{}'. Expected one of: {}"
                                   .format(name, value, ", ".join(options)))
        return value
    if declared == MULTICHOICE_TYPE:
        selected = _multichoice(name, value, options)
        minimum = spec.get("min_selected")
        if minimum is not None and len(selected) < minimum:
            raise LocalSchemaError("Argument '{}': select at least {} of {}."
                                   .format(name, minimum, ", ".join(options)))
        return selected
    if declared == LIST_TYPE:
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        if not isinstance(value, str):
            raise LocalSchemaError("Argument '{}' must be a list of strings".format(name))
        text = value.strip()
        if text.startswith("["):
            import json
            try:
                parsed = json.loads(text)
            except ValueError as error:
                raise LocalSchemaError("Argument '{}': invalid JSON ({})".format(name, error))
            if not isinstance(parsed, list):
                raise LocalSchemaError("Argument '{}': expected a JSON array".format(name))
            return [str(item) for item in parsed]
        return [item.strip() for item in text.split(",") if item.strip()]
    if declared == VEC2_TYPE:
        if isinstance(value, str):
            parts = [part.strip() for part in value.strip("[]() ").split(",") if part.strip()]
        elif isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            raise LocalSchemaError("Argument '{}' must be two numbers".format(name))
        if len(parts) != 2:
            raise LocalSchemaError("Argument '{}' must be exactly two numbers".format(name))
        try:
            numbers = [float(part) for part in parts]
        except (TypeError, ValueError):
            raise LocalSchemaError("Argument '{}' must be two numbers".format(name))
        for number, bounds, axis in zip(numbers, (spec.get("x_range"), spec.get("y_range")),
                                        ("x", "y")):
            if bounds is not None and not min(bounds) <= number <= max(bounds):
                raise LocalSchemaError("Argument '{}': {} is {}, outside [{}, {}]".format(
                    name, axis, number, bounds[0], bounds[1]))
        return numbers
    if declared == "str":
        if isinstance(value, str):
            return value
        raise LocalSchemaError("Argument '{}' must be a string".format(name))
    if declared == "bool":
        return _as_bool(name, value)
    if declared in ("int", "float"):
        try:
            return (int if declared == "int" else float)(value)
        except (TypeError, ValueError):
            raise LocalSchemaError("Argument '{}' must be a {}".format(name, declared))
    return value


def wire_arguments(schema: dict, values: dict, data_dir: str, tool: str) -> dict:
    """The panel's `values` as `tool`'s run() takes them, as the server would
    have passed them (server/main.py `_run_tool`, base.Tool.validate,
    SchemaTool.for_the_wire, execution/dispatch.py `_server_provided`).

    `schema` is the published schema the panel was built from (a facade's when
    a facade was picked); `tool` the tool that actually runs, after
    `resolve_facade`. `data_dir` is the local DATA root.

    * Facade: arguments the running engine does not declare (its
      `visible_when["mode"]` excludes it) are dropped, and a choice is checked
      against that engine's own options (`options_when`).
    * None is dropped for every argument, as validate() does; a blank string
      for a file or hosted argument (the form placeholder main.py ignores) or
      for any OPTIONAL argument (what base_widget.collectArgs never sends).
    * A hosted argument's bare NAME becomes DATA/<slug>/<models|testfiles>/
      [<scope>/]<name>; an absolute path is kept as it is.
    * A choice or multichoice left out gets its published default; a
      multichoice `{option: ticked}` becomes the list of ticked options.
    * A hosted MODEL argument nobody named gets DATA/<slug>/models when that
      folder exists, and the tool picks its own bundle inside it.

    NOT done here, the caller's job: `output_dir` (the job's own folder) and
    `device` (the deployment default the server injects when the panel names
    none). A choice left out gets its declared default, `device` included (as
    validate() does), so the caller overrides `device` rather than adds it. A
    blank REQUIRED `str` is passed as "", which the server takes literally.
    """
    arguments = schema.get("arguments") or {}
    values = dict(values or {})

    mode = _mode_of(schema.get("name"), tool)
    if mode is not None:
        # The union a facade publishes, narrowed to what this engine declares:
        # a panel switched between modes legitimately still holds the other
        # engine's fields, and the server drops them rather than refuse them.
        arguments = {name: spec for name, spec in arguments.items()
                     if name != MODE_ARGUMENT and _in_mode(spec, mode)}
        for extra in [name for name in values if name not in arguments]:
            values.pop(extra)

    unknown = sorted(set(values) - set(arguments))
    if unknown:
        raise LocalSchemaError("Unexpected argument(s) for tool '{}': {}"
                               .format(tool, ", ".join(unknown)))

    declared = _declared(tool)
    slug = data_slug(tool, data_dir)
    params = {}
    for name, spec in arguments.items():
        kind, scope = _selectable_of(name, spec, declared)
        value = values.get(name)
        # A blank file or hosted value is the form's placeholder (main.py drops
        # it); a blank OPTIONAL value is what the panel never sends
        # (base_widget.collectArgs), so the tool's own default applies.
        if _empty(value) and (kind or spec["type"] == PATH_TYPE or not spec["required"]):
            value = None
        if value is None:
            if spec["type"] in (CHOICE_TYPE, MULTICHOICE_TYPE):
                options = _options_in_mode(spec, mode)
                chosen = [option for option in options if spec["choices"].get(option)]
                params[name] = chosen if spec["type"] == MULTICHOICE_TYPE else (
                    chosen[0] if chosen else options[0])
            elif spec["required"]:
                raise LocalSchemaError("Missing required argument '{}' for tool '{}'"
                                       .format(name, tool))
            continue
        if kind:
            kind_dir = MODELS_DIRNAME if kind == SERVER_SELECTABLE_MODEL else TESTFILES_DIRNAME
            params[name] = _hosted_path(data_dir, slug, kind_dir, scope, str(value), name)
        elif spec["type"] == PATH_TYPE:
            params[name] = str(value)
        else:
            params[name] = _coerce(name, value, spec, _options_in_mode(spec, mode))

    models = os.path.join(data_dir, slug, MODELS_DIRNAME)
    for name, spec in arguments.items():
        kind, _ = _selectable_of(name, spec, declared)
        if kind != SERVER_SELECTABLE_MODEL or name in params or not os.path.isdir(models):
            continue
        # A model the panel SHOWS, optional and left on "(automatic)", is the
        # user saying "let the tool choose": CNE picks the fine-tune matching
        # its note type when given nothing, and refuses a folder holding both.
        # A hidden one keeps the server's rule: the tool picks inside the folder.
        if not spec.get("hidden") and not spec.get("required"):
            continue
        params[name] = models
    return params


def _mode_of(schema_name, tool):
    """The facade mode that runs `tool`, or None when `schema_name` is no facade."""
    for mode, target in FACADES.get(schema_name or "", {}).items():
        if target == tool:
            return mode
    return None


def _in_mode(spec, mode) -> bool:
    modes = (spec.get("visible_when") or {}).get(MODE_ARGUMENT)
    return modes is None or mode in (modes if isinstance(modes, list) else [modes])


def _options_in_mode(spec, mode) -> list:
    """The options this engine itself declares: a facade publishes the union."""
    options = list(spec.get("choices") or {})
    narrowed = ((spec.get("options_when") or {}).get(MODE_ARGUMENT) or {}).get(mode)
    if mode is not None and narrowed:
        return [option for option in options if option in narrowed]
    return options


def _selectable_of(name, spec, declared) -> tuple:
    """`(kind, scope)` the RUNNING tool hosts this argument as. A facade
    publishes its first engine's spec, so the engine's own deployment entry is
    read first: AREG's `t1` is scoped CBCT_T1 for one engine, IOS_T1 for another."""
    kind = declared["server_selectable"].get(name, spec.get("server_selectable"))
    if kind == SERVER_SELECTABLE_NONE:
        return None, None
    if name in declared["server_selectable"]:
        return kind, declared["selectable_scopes"].get(name)
    return kind, spec.get("selectable_scope")
