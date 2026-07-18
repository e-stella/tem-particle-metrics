"""Select + configure the Qt binding BEFORE qtpy/napari import.

Import this FIRST in every GUI entry point. It exists because the two ways of
running the app want different Qt bindings:

  - consolidated `tem-app` env: uniform **PyQt6** — npsam hard-imports PyQt6,
    and napari 0.5+ supports it, so the whole stack must be PyQt6.
  - multi-env dev (`tem-gui`): **PyQt5**.

This picks whichever binding is installed (PyQt6 first) via QT_API, and for
PyQt6 points Qt at its bundled plugins so the platform plugin loads in launch
contexts where the wheel's path isn't auto-discovered. No-op if QT_API is
already set. Harmless in a Qt-less env (nothing installed -> nothing set).
"""
import importlib.util
import os


def _configure() -> None:
    if "QT_API" not in os.environ:
        if importlib.util.find_spec("PyQt6") is not None:
            os.environ["QT_API"] = "pyqt6"
        elif importlib.util.find_spec("PyQt5") is not None:
            os.environ["QT_API"] = "pyqt5"

    if os.environ.get("QT_API") == "pyqt6":
        spec = importlib.util.find_spec("PyQt6")
        locs = getattr(spec, "submodule_search_locations", None) if spec else None
        if locs:
            plugins = os.path.join(list(locs)[0], "Qt6", "plugins")
            if os.path.isdir(plugins):
                os.environ.setdefault("QT_PLUGIN_PATH", plugins)
                os.environ.setdefault(
                    "QT_QPA_PLATFORM_PLUGIN_PATH", os.path.join(plugins, "platforms")
                )


_configure()
