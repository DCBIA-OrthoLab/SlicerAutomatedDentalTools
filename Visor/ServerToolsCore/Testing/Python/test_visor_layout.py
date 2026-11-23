"""The two sets of modules, as the build and Slicer will see them.

No Slicer: it reads CMakeLists.txt and the sources. What it pins is the rule of
the whole arrangement: the classic modules are upstream's, byte for byte, and
everything VISOR is under `Visor/`, with names that cannot collide with them.
"""

import ast
import os
import re
import subprocess
import sys
import unittest

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
_VISOR = os.path.abspath(os.path.join(_CORE, ".."))
_REPO = os.path.abspath(os.path.join(_VISOR, ".."))
sys.path.insert(0, _CORE)

from ServerToolsCoreLib import console, mode_switch  # noqa: E402


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as handle:
        return handle.read()


def _modules(folder):
    """{name: folder} of the scripted or command line modules under `folder`.

    A module is a folder whose CMakeLists.txt sets MODULE_NAME to a name whose
    `<name>.py` or `<name>.xml` is in that folder.
    """
    found = {}
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in (".git", "Testing", "__pycache__")]
        if "CMakeLists.txt" not in files:
            continue
        for name in re.findall(r"set\(MODULE_NAME (\w+)\)", _read(root, "CMakeLists.txt")):
            if os.path.isfile(os.path.join(root, name + ".py")) or os.path.isfile(os.path.join(root, name + ".xml")):
                found[name] = root
    return found


def _manifest(folder):
    """The files a module folder's CMakeLists.txt lists as scripts, as relative paths."""
    text = _read(folder, "CMakeLists.txt")
    name = re.search(r"set\(MODULE_NAME (\w+)\)", text).group(1)
    match = re.search(r"set\(MODULE_PYTHON_SCRIPTS(.*?)\)", text, re.S)
    if match is None:
        return set()
    block = match.group(1)
    return {line.strip().replace("${MODULE_NAME}", name) for line in block.splitlines() if line.strip()}


def _sources(folder):
    out = set()
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in ("Testing", "__pycache__")]
        for f in files:
            if f.endswith(".py"):
                out.add(os.path.relpath(os.path.join(root, f), folder))
    return out


VISOR = _modules(_VISOR)
ROOT_MODULES = {n: p for n, p in _modules(_REPO).items() if not p.startswith(_VISOR + os.sep)}


class CollisionTest(unittest.TestCase):
    def test_the_root_builds_the_visor_folder_and_the_classic_modules(self):
        text = _read(_REPO, "CMakeLists.txt")
        self.assertRegex(text, re.compile(r"^add_subdirectory\(Visor\)", re.M))
        self.assertTrue(any(m.startswith("ALI") for m in ROOT_MODULES))
        self.assertIn("ALI", ROOT_MODULES)

    def test_no_module_name_is_in_both_sets_in_any_case(self):
        mine = {n.lower() for n in VISOR}
        theirs = {n.lower() for n in ROOT_MODULES}
        self.assertEqual(mine & theirs, set())

    def test_every_folder_name_is_unique_in_the_installed_flat_layout(self):
        """Installed, every script of every module lands in one folder."""
        owners = {}
        for name, folder in {**ROOT_MODULES, **VISOR}.items():
            for rel in _manifest(folder):
                if "${" in rel:
                    continue
                owners.setdefault(rel.lower(), []).append(name)
        # `__init__.py` of a package is not a top-level file: only the first component counts.
        tops = {}
        for rel, who in owners.items():
            top = rel.split("/")[0]
            if top.endswith(".py") or "/" in rel:
                tops.setdefault(top, set()).update(who)
        shared = {t: w for t, w in tops.items() if len(w) > 1 and t in
                  {x.lower() for x in console.OUR_PACKAGES} | {f.lower() + ".py" for f in console.OUR_MODULE_FILES}}
        self.assertEqual(shared, {}, "a VISOR file is also installed by another module")
        for top in tops:
            if top in {f.lower() + ".py" for f in console.OUR_MODULE_FILES}:
                self.assertEqual(len(tops[top]), 1, top)


class ManifestTest(unittest.TestCase):
    def test_every_visor_module_is_built_and_every_built_one_exists(self):
        on_disk = {d for d in os.listdir(_VISOR) if os.path.isfile(os.path.join(_VISOR, d, d + ".py"))}
        self.assertEqual(set(VISOR), on_disk)

    def test_module_name_is_the_folder_name(self):
        for name, folder in VISOR.items():
            self.assertEqual(re.search(r"set\(MODULE_NAME (\w+)\)", _read(folder, "CMakeLists.txt")).group(1), name)

    def test_the_manifest_of_each_module_is_exactly_its_python_files(self):
        for name, folder in VISOR.items():
            self.assertEqual(_manifest(folder), _sources(folder), name)

    def test_the_lib_modules_are_all_listed(self):
        """`tools_ui.py` was missing, and an installed extension then failed to import it."""
        listed = _manifest(os.path.join(_VISOR, "ServerToolsCore"))
        for f in os.listdir(os.path.join(_VISOR, "ServerToolsCore", "ServerToolsCoreLib")):
            if f.endswith(".py"):
                self.assertIn("ServerToolsCoreLib/" + f, listed, f)

    def test_every_import_of_a_shared_package_is_in_a_manifest(self):
        """Exhaustive: every `ServerToolsCoreLib.x`, `FlexRegLib.x` and `VISULib.x` a file imports is installed."""
        listed = set()
        for folder in VISOR.values():
            listed |= _manifest(folder)
        problems = []
        for name, folder in VISOR.items():
            for rel in _sources(folder):
                tree = ast.parse(_read(folder, rel))
                here = os.path.dirname(rel)
                for node in ast.walk(tree):
                    wanted = []
                    if isinstance(node, ast.ImportFrom):
                        if node.level == 0 and node.module and node.module.split(".")[0] in console.OUR_PACKAGES:
                            pkg = node.module.split(".")
                            if len(pkg) > 1:
                                wanted.append("/".join(pkg) + ".py")
                            else:
                                wanted += [pkg[0] + "/" + a.name + ".py" for a in node.names
                                           if os.path.exists(os.path.join(_VISOR, "*", pkg[0]) ) or True]
                        elif node.level == 1 and here:
                            if node.module:
                                wanted.append(os.path.join(here, node.module.split(".")[0]) + ".py")
                            else:
                                wanted += [os.path.join(here, a.name) + ".py" for a in node.names]
                    for w in wanted:
                        w = w.replace(os.sep, "/")
                        pkg_dir = os.path.dirname(w)
                        # A name imported from a package may be an attribute rather than a file.
                        exists = any(os.path.isfile(os.path.join(f, w)) for f in VISOR.values())
                        if exists and w not in listed:
                            problems.append(f"{name}/{rel} imports {w}, not in any manifest")
        self.assertEqual(problems, [])

    def test_the_mode_switch_and_tools_ui_are_installed(self):
        listed = _manifest(os.path.join(_VISOR, "ServerToolsCore"))
        self.assertIn("ServerToolsCoreLib/mode_switch.py", listed)
        self.assertIn("ServerToolsCoreLib/tools_ui.py", listed)


class ClassesTest(unittest.TestCase):
    def test_each_twin_defines_the_classes_slicer_looks_for(self):
        for classic in list(mode_switch.EQUIVALENTS) + list(mode_switch.NO_PARITY):
            name = classic + "Visor"
            tree = ast.parse(_read(VISOR[name], name + ".py"))
            classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
            self.assertIn(name, classes)
            self.assertIn(name + "Widget", classes)
            self.assertNotIn(classic, classes, "the classic class name must not be reused")
            self.assertNotIn(classic + "Widget", classes)

    def test_tool_names_are_the_sadt_visor_names_not_module_names(self):
        for classic in list(mode_switch.EQUIVALENTS) + list(mode_switch.NO_PARITY):
            text = _read(VISOR[classic + "Visor"], classic + "Visor.py")
            tool = re.search(r'^\s+TOOL_NAME = "(\w+)"', text, re.M).group(1)
            self.assertFalse(tool.endswith("Visor"), classic)

    def test_a_visor_file_never_imports_a_classic_module(self):
        classic_packages = set()
        for folder in ROOT_MODULES.values():
            classic_packages |= {r.split("/")[0] for r in _manifest(folder) if "/" in r}
        classic_names = set(ROOT_MODULES) | classic_packages
        offenders = []
        for name, folder in VISOR.items():
            for rel in _sources(folder):
                for node in ast.walk(ast.parse(_read(folder, rel))):
                    mods = []
                    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        mods.append(node.module.split(".")[0])
                    elif isinstance(node, ast.Import):
                        mods += [a.name.split(".")[0] for a in node.names]
                    for m in mods:
                        if m in classic_names:
                            offenders.append(f"{name}/{rel}: {m}")
        self.assertEqual(offenders, [])


class TablesMatchTheTreeTest(unittest.TestCase):
    def test_the_pairs_are_the_modules_of_both_sets(self):
        for classic in list(mode_switch.EQUIVALENTS) + list(mode_switch.NO_PARITY):
            self.assertIn(classic, ROOT_MODULES, classic)
            self.assertIn(classic + "Visor", VISOR)
        self.assertEqual(set(mode_switch.VISOR_MODULES), {n for n in VISOR if n.endswith("Visor")})

    def test_every_classic_module_without_a_twin_is_listed_as_such(self):
        twins = set(mode_switch.EQUIVALENTS) | set(mode_switch.NO_PARITY)
        scripted = {n for n, f in ROOT_MODULES.items() if os.path.isfile(os.path.join(f, n + ".py"))
                    and not os.path.isfile(os.path.join(f, n + ".xml")) and not n.startswith("ADT")}
        self.assertEqual(scripted - twins - set(mode_switch.CLASSIC_ONLY), set())
        for name in mode_switch.CLASSIC_ONLY:
            self.assertIn(name, ROOT_MODULES)

    def test_the_command_line_modules_named_exist(self):
        titles = set()
        for folder in ROOT_MODULES.values():
            pass
        for root, _dirs, files in os.walk(_REPO):
            if os.sep + "Visor" in root or ".git" in root:
                continue
            if "CMakeLists.txt" in files:
                titles |= set(re.findall(r"set\(MODULE_NAME (\w+)\)", _read(root, "CMakeLists.txt")))
        for classic, clis in mode_switch.EQUIVALENTS.items():
            for cli in clis:
                self.assertIn(cli, titles, f"{classic}: {cli}")

    def test_visor_only_modules_are_in_the_visor_folder(self):
        for name in mode_switch.VISOR_ONLY:
            self.assertIn(name, VISOR)


def _git(*args):
    return subprocess.run(["git", "-C", _REPO, *args], capture_output=True, text=True)


class ClassicIsUpstreamTest(unittest.TestCase):
    """The classic modules are upstream's, byte for byte."""

    @classmethod
    def setUpClass(cls):
        probe = _git("rev-parse", "--verify", "--quiet", "upstream/main")
        if probe.returncode != 0:
            raise unittest.SkipTest("no upstream/main ref to compare with")

    def test_nothing_outside_visor_and_the_root_cmake_differs_from_upstream(self):
        out = _git("diff", "--stat", "upstream/main", "--", ".", ":!Visor", ":!CMakeLists.txt")
        self.assertEqual(out.stdout.strip(), "")

    def test_the_root_cmake_only_adds_the_visor_folder(self):
        out = _git("diff", "-U0", "upstream/main", "--", "CMakeLists.txt").stdout
        changed = [l for l in out.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        self.assertTrue(all(l.startswith("+") for l in changed), changed)
        self.assertIn("+add_subdirectory(Visor)", changed)


if __name__ == "__main__":
    unittest.main()
