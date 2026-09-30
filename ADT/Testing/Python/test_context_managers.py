# A `with` that works, on both sides of the decorator.
#
# VFACE carried the two halves of the same mistake, and neither raised until
# the step ran:
#
#   outputToFile       yields, no decorator  -> `with self.logic.outputToFile()`
#                      raised "'generator' object does not support the context
#                      manager protocol" before the step it wraps ever ran, and
#                      the output redirection it exists for never happened.
#   _reportStepOutput  decorated, no yield   -> the plain call at its one call
#                      site returned a context manager object instead of running
#                      the body, so the step's last lines never reached the log
#                      and its temporary file was leaked, once per step.
#
# Neither is visible by reading one line: the decorator sits ten lines above the
# yield it belongs to. These two rules read every module and say which function.
import ast
import glob
import os
import sys
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "ADT"))

#: Third-party trees vendored under the modules: not ours to judge.
SKIPPED = ("/.git/", "/build/", "/site-packages/", "/__pycache__/")


def module_sources():
    for path in glob.glob(os.path.join(_ROOT, "**", "*.py"), recursive=True):
        if any(part in path.replace(os.sep, "/") for part in SKIPPED):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                yield path, ast.parse(handle.read(), filename=path)
        except (OSError, SyntaxError):
            # A file this interpreter cannot parse is the packaging check's
            # problem, not this one's. Saying nothing beats a false finding.
            continue


def is_contextmanager(node):
    for decorator in node.decorator_list:
        name = ast.unparse(decorator)
        if name.endswith("contextmanager"):
            return True
    return False


def own_yields(node):
    nested = {id(n) for child in ast.iter_child_nodes(node)
              for n in ast.walk(child)
              if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return any(isinstance(n, (ast.Yield, ast.YieldFrom))
               for n in ast.walk(node) if id(n) not in nested)


def functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def names_entered_with(tree):
    """The attribute names appearing as `with something.NAME(...)`."""
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                found.add(call.func.attr)
    return found


class ContextManagerTest(unittest.TestCase):

    def test_a_decorated_function_yields(self):
        """@contextmanager on something that never yields runs nothing at all."""
        offenders = []
        for path, tree in module_sources():
            for node in functions(tree):
                if is_contextmanager(node) and not own_yields(node):
                    offenders.append("%s:%d %s"
                                     % (os.path.relpath(path, _ROOT),
                                        node.lineno, node.name))
        self.assertEqual(offenders, [], "decorated but never yields:\n  "
                         + "\n  ".join(offenders))

    def test_a_generator_entered_with_with_is_decorated(self):
        """A bare generator in a `with` raises before its body ever runs."""
        offenders = []
        for path, tree in module_sources():
            entered = names_entered_with(tree)
            for node in functions(tree):
                if (node.name in entered and own_yields(node)
                        and not is_contextmanager(node)):
                    offenders.append("%s:%d %s"
                                     % (os.path.relpath(path, _ROOT),
                                        node.lineno, node.name))
        self.assertEqual(offenders, [], "used with `with` but not decorated:\n  "
                         + "\n  ".join(offenders))


class RulesDetectTheDefectTest(unittest.TestCase):
    """The rules above, read against the two shapes they exist to catch.

    A probe that cannot fail on the defect it describes is worth nothing, so
    both are run here against source that has it.
    """

    DECORATED_WITHOUT_YIELD = (
        "import contextlib\n"
        "class A:\n"
        "    @contextlib.contextmanager\n"
        "    def report(self, path):\n"
        "        return None\n"
    )
    GENERATOR_WITHOUT_DECORATOR = (
        "class A:\n"
        "    def capture(self):\n"
        "        yield 1\n"
        "    def run(self):\n"
        "        with self.capture() as handle:\n"
        "            pass\n"
    )

    def test_it_catches_a_decorated_function_that_never_yields(self):
        tree = ast.parse(self.DECORATED_WITHOUT_YIELD)
        bad = [n for n in functions(tree) if is_contextmanager(n) and not own_yields(n)]
        self.assertEqual([n.name for n in bad], ["report"])

    def test_it_catches_a_generator_used_without_the_decorator(self):
        tree = ast.parse(self.GENERATOR_WITHOUT_DECORATOR)
        entered = names_entered_with(tree)
        bad = [n for n in functions(tree)
               if n.name in entered and own_yields(n) and not is_contextmanager(n)]
        self.assertEqual([n.name for n in bad], ["capture"])

    def test_a_nested_generator_does_not_count_as_its_parents_yield(self):
        """Otherwise a decorated wrapper passes on a yield it does not own."""
        tree = ast.parse(
            "import contextlib\n"
            "class A:\n"
            "    @contextlib.contextmanager\n"
            "    def outer(self):\n"
            "        def inner():\n"
            "            yield 1\n"
            "        return inner\n")
        bad = [n.name for n in functions(tree)
               if is_contextmanager(n) and not own_yields(n)]
        self.assertIn("outer", bad)


if __name__ == "__main__":
    unittest.main()
