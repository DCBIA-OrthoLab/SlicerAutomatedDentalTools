"""Unit tests for SADTCoreLib.local_schema, run outside Slicer.

local_schema is a COPY of the VISOR server's schema translation, so the only
test that means anything is a comparison with the server itself. The fixtures
under fixtures/local_schema/ were produced by the server's own code, not by
hand:

    raw/<tool>.json        what scripts/describe.py printed for the tool (the
                           schema the server's registry read)
    published/<tool>.json  that tool's entry in the server's GET /tools,
                           from main.list_tools() over the real registry,
                           facades included
    wire_cases.json        panel values pushed through the server's run path
                           (facade resolution, data_store, validate,
                           for_the_wire, dispatch._server_provided) over a
                           throwaway DATA tree of placeholder names

To regenerate them after a server or tool change, run dump_golden.py and
dump_wire.py (beside the fixtures) with the server's own interpreter, from an
empty working directory, with API_TOKEN, TOOLS_DIR (the SADT-VISOR tools),
DESCRIBE_PATH, DATA_DIR (an empty folder) and SCHEMA_CACHE_DIR (a scratch
folder) set in the environment. No server file or setting is edited.

What is deliberately NOT compared, and why, is listed in LEFT_OUT_* below.

Usage:
    python3 -m unittest test_local_schema
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from SADTCoreLib import local_schema  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "local_schema")

# Top-level keys of a GET /tools entry the local translation leaves out: how a
# client splits a cohort into several requests. A local run takes the whole
# cohort in one call.
LEFT_OUT_TOOL_KEYS = ("batch", "paired_batch")

# Arguments the server injects into every tool that calls another. A local run
# has no checkpoint to stop at and resume from, and no job directory to collect
# intermediate files in. The server also hands them to run() as empty
# selections (validate() fills every omitted multichoice), and its runner takes
# them back out, so they are left out of the wire comparison too.
LEFT_OUT_ARGUMENTS = ("stop_after", "keep_intermediate")


def _load(*parts):
    with open(os.path.join(FIXTURES, *parts), encoding="utf-8") as handle:
        return json.load(handle)


def _names(kind):
    return sorted(name[:-len(".json")] for name in os.listdir(os.path.join(FIXTURES, kind))
                  if name.endswith(".json"))


def _without_left_out(entry):
    entry = {key: value for key, value in entry.items() if key not in LEFT_OUT_TOOL_KEYS}
    entry["arguments"] = {name: spec for name, spec in entry["arguments"].items()
                          if name not in LEFT_OUT_ARGUMENTS}
    return entry


def _raws():
    return {name: _load("raw", name + ".json") for name in _names("raw")}


def _translated(name, raws):
    if local_schema.facade_targets(name):
        return local_schema.facade_schema(
            name, {target: raws[target] for target in local_schema.facade_targets(name)})
    return local_schema.panel_schema(name, raws[name])


class PublishedSchemaTest(unittest.TestCase):
    """panel_schema / facade_schema == what the server publishes, tool by tool."""

    def test_the_fixtures_cover_every_tool_and_both_facades(self):
        published = _names("published")
        self.assertGreaterEqual(len(published), 20)
        for facade in local_schema.FACADES:
            self.assertIn(facade, published)
        for tool in set(published) - set(local_schema.FACADES):
            self.assertIn(tool, _names("raw"))

    def test_every_tool_matches_the_server(self):
        raws = _raws()
        matched = []
        for name in _names("published"):
            with self.subTest(tool=name):
                expected = _without_left_out(_load("published", name + ".json"))
                actual = _translated(name, raws)
                # A deliberate local addition the server does not publish (see
                # test_a_facade_scope_follows_the_mode).
                for spec in actual["arguments"].values():
                    spec.pop("selectable_scope_when", None)
                for argument in expected["arguments"]:
                    self.assertEqual(actual["arguments"].get(argument),
                                     expected["arguments"][argument],
                                     "{}.{}".format(name, argument))
                self.assertEqual(actual, expected)
                # Argument order is section order on the panel.
                self.assertEqual(list(actual["arguments"]), list(expected["arguments"]))
                matched.append(name)
        self.assertEqual(len(matched), len(_names("published")))

    def test_output_dir_is_never_published(self):
        raws = _raws()
        for name, raw in raws.items():
            if local_schema.OUTPUT_DIR_ARGUMENT in raw["arguments"]:
                self.assertNotIn(local_schema.OUTPUT_DIR_ARGUMENT,
                                 local_schema.panel_schema(name, raw)["arguments"])

    def test_a_model_is_published_as_a_name(self):
        schema = local_schema.panel_schema("Batch_Dental_Seg", _raws()["Batch_Dental_Seg"])
        model = schema["arguments"]["model"]
        self.assertEqual((model["type"], model["server_selectable"], model["extensions"]),
                         ("str", "model", None))


class FacadeTest(unittest.TestCase):

    def test_facades_and_their_targets(self):
        self.assertEqual(sorted(local_schema.FACADES), ["ALI", "AREG"])
        self.assertEqual(local_schema.facade_targets("ALI"), ["ALI_CBCT", "ALI_IOS"])
        self.assertEqual(local_schema.facade_targets("AREG"),
                         ["AREG_CBCT", "AREG_IOS", "AREG_IOSCBCT"])
        self.assertEqual(local_schema.facade_targets("AMASSS"), [])

    def test_mode_comes_first_and_opens_on_the_first_engine(self):
        schema = _translated("AREG", _raws())
        self.assertEqual(list(schema["arguments"])[0], "mode")
        self.assertEqual(schema["arguments"]["mode"]["choices"],
                         {"CBCT": True, "IOS": False, "IOS to CBCT": False})

    def test_only_the_installed_engines_become_modes(self):
        # A deliberate difference: the server does not publish a facade at all
        # unless every target is served.
        raws = _raws()
        schema = local_schema.facade_schema("ALI", {"ALI_IOS": raws["ALI_IOS"]})
        self.assertEqual(schema["arguments"]["mode"]["choices"], {"IOS": True})
        for name, spec in schema["arguments"].items():
            self.assertNotIn("mode", spec.get("visible_when") or {}, name)

    def test_a_facade_scope_follows_the_mode(self):
        # The server publishes the first engine's scope only, so AREG offered
        # CBCT test files in IOS mode; locally the scope is published per mode.
        t1 = _translated("AREG", _raws())["arguments"]["t1"]
        per_mode = t1["selectable_scope_when"]["mode"]
        self.assertEqual(per_mode["CBCT"], "CBCT_T1")
        self.assertEqual(per_mode["IOS"], "IOS_T1")
        self.assertNotIn("selectable_scope_when",
                         _translated("AREG", _raws())["arguments"]["cbct"])

    def test_resolve_facade(self):
        self.assertEqual(local_schema.resolve_facade("AREG", {"mode": "IOS", "t1": "x"}),
                         ("AREG_IOS", {"t1": "x"}))
        self.assertEqual(local_schema.resolve_facade("AMASSS", {"mode": "x"}),
                         ("AMASSS", {"mode": "x"}))
        with self.assertRaises(local_schema.LocalSchemaError):
            local_schema.resolve_facade("ALI", {"input": "x"})
        with self.assertRaises(local_schema.LocalSchemaError):
            local_schema.resolve_facade("ALI", {"mode": "Panoramic"})


class WireArgumentsTest(unittest.TestCase):
    """wire_arguments == the server's run path, case by case (wire_cases.json)."""

    def _run_case(self, case):
        data = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, data, ignore_errors=True)
        for entry in case["tree"]:
            os.makedirs(os.path.join(data, entry), exist_ok=True)
        schema = _translated(case["tool"], _raws())
        target, values = local_schema.resolve_facade(case["tool"], case["values"])
        return data, target, local_schema.wire_arguments(schema, values, data, target)

    def test_every_case_matches_the_server(self):
        cases = _load("wire_cases.json")
        self.assertGreaterEqual(len(cases), 8)
        for case in cases:
            with self.subTest(case=case["case"]):
                expected = case["expected"]
                if "error" in expected:
                    with self.assertRaises(local_schema.LocalSchemaError):
                        self._run_case(case)
                    continue
                data, target, params = self._run_case(case)
                self.assertEqual(target, expected["tool"])
                wanted = {
                    name: (value.replace("{DATA}", data) if isinstance(value, str) else value)
                    for name, value in expected["params"].items()
                    if name not in LEFT_OUT_ARGUMENTS
                }
                self.assertEqual(params, wanted)

    def test_an_absolute_path_is_kept_for_a_hosted_argument(self):
        schema = _translated("Batch_Dental_Seg", _raws())
        params = local_schema.wire_arguments(
            schema, {"scans": "/inputs/a", "model": "/models/mine"}, "/nonexistent", "Batch_Dental_Seg")
        self.assertEqual(params["model"], "/models/mine")

    def test_a_name_with_a_separator_is_refused(self):
        schema = _translated("Batch_Dental_Seg", _raws())
        with self.assertRaises(local_schema.LocalSchemaError):
            local_schema.wire_arguments(
                schema, {"scans": "/inputs/a", "model": "../elsewhere"}, "/nonexistent",
                "Batch_Dental_Seg")

    def test_an_unknown_argument_is_refused(self):
        schema = _translated("AMASSS", _raws())
        with self.assertRaises(local_schema.LocalSchemaError):
            local_schema.wire_arguments(schema, {"scans": "/a", "colour": "red"}, "/x", "AMASSS")


class DataSlugTest(unittest.TestCase):
    """server/registry/deployment.py `data_slug`."""

    def setUp(self):
        self.data = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.data, ignore_errors=True)

    def test_underscores_removed_when_the_literal_folder_is_absent(self):
        self.assertEqual(local_schema.data_slug("Batch_Dental_Seg", self.data), "BatchDentalSeg")

    def test_the_literal_folder_wins_when_it_exists(self):
        os.makedirs(os.path.join(self.data, "Batch_Dental_Seg"))
        self.assertEqual(local_schema.data_slug("Batch_Dental_Seg", self.data), "Batch_Dental_Seg")

    def test_the_deployment_data_dir_wins_over_both(self):
        os.makedirs(os.path.join(self.data, "AREG_IOS"))
        self.assertEqual(local_schema.data_slug("AREG_IOS", self.data), "AREG")
        self.assertEqual(local_schema.data_slug("ALI_CBCT", self.data), "ALI")


class CopiedConstantsTest(unittest.TestCase):
    """The literals copied from the server, held to the server's own files when
    a VISOR-serve checkout sits beside this repository (skipped otherwise)."""

    SERVER = os.environ.get("VISOR_SERVE_DIR") or os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "VISOR-serve"))

    def test_deployment_matches_deployment_toml(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest("tomllib needs Python 3.11")
        path = os.path.join(self.SERVER, "server", "deployment.toml")
        if not os.path.isfile(path):
            self.skipTest("no VISOR-serve checkout beside this repository")
        with open(path, "rb") as handle:
            tools = tomllib.load(handle).get("tools", {})
        kept = ("server_selectable", "hidden", "data_dir", "dispatch")
        expected = {name: {key: table[key] for key in kept if key in table}
                    for name, table in tools.items()}
        expected = {name: entry for name, entry in expected.items() if entry}
        actual = {name: {key: (list(value) if key == "hidden" else value)
                         for key, value in entry.items()}
                  for name, entry in local_schema.DEPLOYMENT.items()}
        self.assertEqual(actual, expected)

    def test_technical_matches_conventions(self):
        import ast
        path = os.path.join(self.SERVER, "server", "registry", "conventions.py")
        if not os.path.isfile(path):
            self.skipTest("no VISOR-serve checkout beside this repository")
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TECHNICAL":
                values = {element.value for element in node.value.args[0].elts}
                self.assertEqual(set(local_schema.TECHNICAL), values)
                return
        self.fail("TECHNICAL not found in conventions.py")


if __name__ == "__main__":
    unittest.main()


class AutomaticModelTest(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.data, "CNE", "models", "TMJ"))
        os.makedirs(os.path.join(self.data, "ALI", "models"))

    def test_a_shown_optional_model_left_automatic_is_not_sent(self):
        schema = _translated("CNE", _raws())
        params = local_schema.wire_arguments(
            schema, {"notes": "/n", "notes_type": "TMJ"}, self.data, "CNE")
        self.assertNotIn("model", params)

    def test_a_hidden_model_still_gets_the_models_folder(self):
        schema = _translated("ALI", _raws())
        params = local_schema.wire_arguments(
            schema, {"mode": "IOS", "input": "/x"}, self.data, "ALI_IOS")
        self.assertEqual(params["model"], os.path.join(self.data, "ALI", "models"))
