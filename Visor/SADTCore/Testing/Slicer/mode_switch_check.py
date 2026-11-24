"""Checks, inside a running Slicer, what the unit tests cannot: that the bar is
injected into a classic panel, survives a Reload, sits in a VISOR panel, and
that its buttons do what they say.

Not a unittest: run it with Slicer, without a main window, with the module
paths of BOTH sets, and with them first on PYTHONPATH when another copy of the
extension is installed:

    Slicer --no-main-window --disable-settings --ignore-slicerrc \
        --additional-module-paths <classic dirs> <Visor dirs> \
        --python-script mode_switch_check.py

Exit status 0 when every line says PASS. The main window is NOT exercised: the
moduleSelected signal and the real selectModule are replaced or skipped here.
"""
import time, slicer, qt, traceback
from SADTCoreLib import mode_switch as ms
print("mode_switch from", ms.__file__)
res = []
def check(label, ok, extra=""):
    res.append(ok); print(("PASS " if ok else "FAIL ") + label + (" | " + str(extra) if extra else ""), flush=True)

def wait(seconds):
    t = time.time()
    while time.time() - t < seconds:
        slicer.app.processEvents(); time.sleep(0.05)

def bars_in(layout, name=ms.BAR_NAME):
    out = []
    for i in range(layout.count()):
        w = layout.itemAt(i).widget()
        if w is not None and w.objectName == name: out.append(i)
    return out

# --- titles, categories, collisions
for n in ms.VISOR_MODULES:
    m = getattr(slicer.modules, n.lower(), None)
    check(f"{n} loaded", m is not None)
    if m is not None:
        check(f"{n} title/category", "(VISOR local)" in m.title and "VISOR local" in m.categories[0], (m.title, m.categories))
check("title of ALI classic differs", slicer.modules.ali.title != slicer.modules.alivisor.title, (slicer.modules.ali.title, slicer.modules.alivisor.title))

# --- classic injection, every module of the table
for n in ms.EQUIVALENTS:
    try:
        rep = getattr(slicer.modules, n.lower()).widgetRepresentation()
        ok = ms.ensure_bar(n)
        lay = rep.layout()
        check(f"classic {n}: bar injected at index 0", ok and bars_in(lay) == [0], bars_in(lay))
        # idempotent
        ms.ensure_bar(n); ms.ensure_existing()
        check(f"classic {n}: still one bar after 3 calls", len(bars_in(lay)) == 1)
    except Exception:
        check(f"classic {n}: injection", False, traceback.format_exc().splitlines()[-1])
gl = slicer.modules.greedyreg.widgetRepresentation().layout()
ms.ensure_existing()
check("GreedyReg (no parity): no bar", bars_in(gl) == [] and not ms.ensure_bar("GreedyReg"))
fl = slicer.modules.flexreg.widgetRepresentation().layout()
ms.ensure_existing()
check("FlexReg (classic only until its VISOR port): no bar", bars_in(fl) == [] and not ms.ensure_bar("FlexReg"))
check("no FlexRegVisor module", getattr(slicer.modules, "flexregvisor", None) is None)

# --- reload keeps the bar (poll)
lay = slicer.modules.ali.widgetRepresentation().layout()
slicer.util.reloadScriptedModule("ALI")
lay = slicer.modules.ali.widgetRepresentation().layout()
print("after reload, bars in layout:", bars_in(lay))
check("after Reload the bar is gone from the layout (reproduces the critique)", bars_in(lay) == [])
ms._INJECTOR.start()
wait(ms.POLL_MS / 1000 * 2.5)
lay = slicer.modules.ali.widgetRepresentation().layout()
check("after Reload + poll, the bar is back at index 0, once", bars_in(lay) == [0], bars_in(lay))
bar = ms._BARS[(ms.CLASSIC, "ALI")]
check("bar frame visible", bar.frame.isVisibleTo(slicer.modules.ali.widgetRepresentation()) )
# second reload
slicer.util.reloadScriptedModule("ALI"); wait(2.5)
lay = slicer.modules.ali.widgetRepresentation().layout()
check("second Reload also recovers", bars_in(lay) == [0], bars_in(lay))

# --- visor panel
vrep = slicer.modules.alivisor.widgetRepresentation()
vlay = vrep.layout()
check("ALIVisor panel has the bar at index 0", bars_in(vlay) == [0], bars_in(vlay))
vb = ms._BARS[(ms.VISOR, "ALI")]
slicer.util.reloadScriptedModule("ALIVisor")
vlay = slicer.modules.alivisor.widgetRepresentation().layout()
check("ALIVisor after Reload: bar rebuilt by setup(), once", bars_in(vlay) == [0], bars_in(vlay))
check("GreedyRegVisor: no bar", bars_in(slicer.modules.greedyregvisor.widgetRepresentation().layout()) == [])

# --- click, with selectModule recorded and a throw-away settings object
chosen = []
slicer.util.selectModule = lambda name: chosen.append(name)
s = qt.QSettings()
s.remove(ms.settings_key("ALI"))
check("default mode is classic", ms.get_mode("ALI") == "classic")
cb = ms._BARS[(ms.CLASSIC, "ALI")]
check("classic button text", cb.button.text == "Passer à la version VISOR locale", cb.button.text)
cb.button.click()
check("click classic -> selectModule(ALIVisor), mode visor", chosen == ["ALIVisor"] and ms.get_mode("ALI") == "visor", (chosen, ms.get_mode("ALI")))
vb = ms._BARS[(ms.VISOR, "ALI")]
check("visor button text", vb.button.text == "Revenir à la version classique", vb.button.text)
vb.button.click()
check("click visor -> selectModule(ALI), mode classic", chosen == ["ALIVisor", "ALI"] and ms.get_mode("ALI") == "classic", (chosen, ms.get_mode("ALI")))
# failing select restores
def boom(n): raise RuntimeError("no main window")
slicer.util.selectModule = boom
cb.button.click()
check("failed selectModule leaves the stored mode unchanged", ms.get_mode("ALI") == "classic")
slicer.util.selectModule = lambda name: chosen.append(name)

# --- busy: classic
real = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCommandLineModuleNode")
check("real CLI node exposes IsBusy/GetModuleTitle", hasattr(real, "IsBusy") and hasattr(real, "GetModuleTitle"), real.IsBusy())
class FakeNode:
    def __init__(self, title, busy): self.t, self.b = title, busy
    def GetModuleTitle(self): return self.t
    def IsBusy(self): return self.b
nodes = [FakeNode("ALI_CBCT", True)]
orig = slicer.util.getNodesByClass
slicer.util.getNodesByClass = lambda c: nodes
cb.refresh()
check("classic busy: button disabled and text says why", (not cb.button.enabled) and "en cours" in cb.explanation.text, cb.explanation.text)
cb.button.click(); n_before = len(chosen)
check("disabled button does nothing", len(chosen) == n_before)
nodes[0].b = False
cb.refresh()
check("classic idle again: enabled", cb.button.enabled)
nodes[:] = [FakeNode("AREG_CBCT", True)]
cb.refresh()
check("a running AREG does not grey the ALI bar", cb.button.enabled)
slicer.util.getNodesByClass = orig
# --- busy: visor
w = slicer.modules.ALIVisorWidget
w._runs.append(object())
vb = ms._BARS[(ms.VISOR, "ALI")]; vb.refresh()
check("visor busy: disabled with reason", (not vb.button.enabled) and "en cours" in vb.explanation.text, vb.explanation.text)
w._runs.clear(); vb.refresh()
check("visor idle: enabled", vb.button.enabled)
# --- target missing
st = ms.bar_state("classic", "ALI", "", False)
check("target missing: disabled, says so", (not st.enabled) and "pas chargée" in st.explanation)
print("SUMMARY", sum(res), "/", len(res))
slicer.app.exit(0 if all(res) else 1)
