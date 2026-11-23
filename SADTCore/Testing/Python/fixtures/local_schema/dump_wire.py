"""Run each wire case through the server's own code: facade resolution and the
form handling of main._run_tool, data_store, validate, for_the_wire and
dispatch._server_provided. Uploads are simulated by passing the local path."""
import json, os, shutil, sys, tempfile
server_dir, cases_path, out_path = sys.argv[1:4]
sys.path.insert(0, server_dir)
import main
from base import ResolvedPath, ToolArgumentError
from config import settings
from data_store import LocalDataStore, DataNotFoundError
from execution import dispatch
from registry import TOOLS
from registry.deployment import deployment_config
from registry.facade import FacadeTool, MODE_ARGUMENT

published = {entry["name"]: entry for entry in main.list_tools()}

def client_collect(name, values):
    # base_widget.collectArgs: an optional "" is not sent.
    arguments = published[name]["arguments"]
    return {k: v for k, v in values.items()
            if not (v == "" and not arguments.get(k, {}).get("required"))}

def server_wire(name, values, root):
    settings.DATA_DIR = root
    store = LocalDataStore(root)
    tool = TOOLS[name]
    args = client_collect(name, values)
    if isinstance(tool, FacadeTool):
        tool = TOOLS[tool.target_for(args.pop(MODE_ARGUMENT))]
        for extra in [k for k in args if k not in tool.arguments]:
            args.pop(extra)
    # An absolute path stands for an UPLOAD (it travels in uploaded_files on the
    # real route); anything else for a selectable argument is a hosted name.
    server_file_args = {k: args.pop(k) for k in list(args)
                        if k in tool.arguments and tool.arguments[k].server_selectable
                        and not (isinstance(args[k], str) and os.path.isabs(args[k]))}
    for k in list(args):
        spec = tool.arguments.get(k)
        if spec is None or not spec.is_file:
            continue
        if args[k] in ("", None):
            args.pop(k)
        else:
            args[k] = ResolvedPath(args[k], "path")   # stands for the upload
    for k, filename in server_file_args.items():
        spec = tool.arguments[k]
        resolver = store.resolve_model if spec.server_selectable == "model" else store.resolve_testfile
        scope = spec.selectable_scope or ""
        resolved = resolver(deployment_config.data_slug(tool.name), filename, *((scope,) if scope else ()))
        args[k] = ResolvedPath(resolved.path, "file")
    params = tool.for_the_wire(tool.validate(args))
    job = tempfile.mkdtemp()
    try:
        filled = dispatch._server_provided(tool, params, job)
    finally:
        shutil.rmtree(job)
    filled.pop("output_dir", None)
    return tool.name, filled

def placeholder(value, root):
    if isinstance(value, str) and value.startswith(root):
        return "{DATA}" + value[len(root):]
    if isinstance(value, (list, tuple)):
        return [placeholder(v, root) for v in value]
    return value

out = []
for case in json.load(open(cases_path)):
    root = tempfile.mkdtemp()
    for entry in case["tree"]:
        os.makedirs(os.path.join(root, entry), exist_ok=True)
    try:
        target, params = server_wire(case["tool"], case["values"], root)
        case["expected"] = {"tool": target,
                            "params": {k: placeholder(str(v) if isinstance(v, ResolvedPath) else v, root)
                                       for k, v in params.items()}}
    except (ToolArgumentError, DataNotFoundError, Exception) as error:
        case["expected"] = {"error": type(error).__name__}
    finally:
        shutil.rmtree(root)
    out.append(case)
json.dump(out, open(out_path, "w"), indent=1)
for case in out:
    print(case["case"], json.dumps(case["expected"])[:400])
