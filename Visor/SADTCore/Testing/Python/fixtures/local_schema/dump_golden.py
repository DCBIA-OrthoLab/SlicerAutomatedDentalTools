"""Dump, for every packaged tool, the raw schema the server read and the entry
GET /tools publishes for it, using the server's own registry and handler."""
import json
import os
import sys

server_dir, out_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, server_dir)

import main  # noqa: E402  builds the registry at import
from registry import TOOLS, FAILED_TOOLS  # noqa: E402
from registry.schema_tool import SchemaTool, resolve_schema  # noqa: E402
from registry.facade import FacadeTool  # noqa: E402

os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "published"), exist_ok=True)
published = {entry["name"]: entry for entry in main.list_tools()}
kept = []
for name, tool in sorted(TOOLS.items()):
    if isinstance(tool, SchemaTool):
        raw = resolve_schema(tool.folder)
        with open(os.path.join(out_dir, "raw", name + ".json"), "w") as handle:
            json.dump(raw, handle, indent=1)
    elif not isinstance(tool, FacadeTool):
        continue  # the server's own in-process fixtures
    with open(os.path.join(out_dir, "published", name + ".json"), "w") as handle:
        json.dump(published[name], handle, indent=1)
    kept.append(name)
print("kept", kept)
print("failed", sorted(FAILED_TOOLS))
