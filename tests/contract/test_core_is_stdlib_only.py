# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Policy-as-test: ``src/zeos/core`` must remain stdlib-only.

The comment in ``pyproject.toml`` says it, the module map documents it, and this
file enforces it: the kernel core is the portability hedge for an eventual
on-platform port, so nothing reachable from ``zeos.core`` may depend on a
third-party package. PyYAML lives behind ``descriptor/loader.py``, which the
core never imports.

Two layers, because each catches what the other cannot:

* the **AST check** pins the declared import boundary file by file, so a new
  import of ``zeos.driver`` or ``zeos.demo`` fails even if it would be harmless
  at runtime;
* the **closure check** imports the real modules in a clean interpreter and
  inspects what actually loaded, so a third-party dependency smuggled in
  *transitively* (say, ``descriptor/schema.py`` growing a yaml import) fails
  even though every declared import still looks clean.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import zeos.core

CORE = Path(zeos.core.__file__).parent

#: The declared seams. Everything here must itself stay stdlib-only, which is
#: exactly what the closure check below verifies.
ALLOWED_ZEOS_PREFIXES = (
    "zeos.core",
    "zeos.world",
    "zeos.descriptor.schema",
    "zeos.machine.base",
    "zeos.nli",
)


def _absolute_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return modules


def test_core_imports_only_stdlib_and_the_declared_seams() -> None:
    violations: list[str] = []
    for path in sorted(CORE.glob("*.py")):
        for module in _absolute_imports(path):
            top = module.split(".")[0]
            if top in sys.stdlib_module_names:
                continue
            if top == "zeos" and module.startswith(ALLOWED_ZEOS_PREFIXES):
                continue
            violations.append(f"{path.name}: import of {module!r}")
    assert not violations, (
        "src/zeos/core must import only the standard library and the declared "
        "seams " + str(list(ALLOWED_ZEOS_PREFIXES)) + "; found: " + "; ".join(violations)
    )


def test_ids_is_a_leaf() -> None:
    """``core/ids.py`` imports nothing from zeos at all -- that is what keeps the
    package cycle-free, so it gets its own assertion."""
    zeos_imports = [m for m in _absolute_imports(CORE / "ids.py") if m.split(".")[0] == "zeos"]
    assert not zeos_imports, f"core/ids.py must stay a leaf; found {zeos_imports}"


def test_the_core_import_closure_contains_no_third_party() -> None:
    """Import every core module in a fresh interpreter and check what loaded."""
    program = """
import sys
baseline = {name.split(".")[0] for name in sys.modules}
import importlib, pkgutil
import zeos.core
for info in pkgutil.iter_modules(zeos.core.__path__):
    importlib.import_module(f"zeos.core.{info.name}")
loaded = {name.split(".")[0] for name in sys.modules} - baseline
foreign = sorted(
    name for name in loaded if name != "zeos" and name not in sys.stdlib_module_names
)
assert not foreign, f"importing zeos.core loaded third-party modules: {foreign}"
"""
    subprocess.run([sys.executable, "-c", program], check=True)
