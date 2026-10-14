"""Desktop application entry point."""

from __future__ import annotations

import sys


def self_test() -> int:
    """Train, save, reload and predict on a bundled example; exit code 0 means OK.

    Lets users (and the build script) verify that a packaged app is complete.
    """
    import tempfile

    from ..data import load_dataset
    from ..storage import load_model, model_path, save_experiment
    from ..trainer import Settings, run_experiment
    from .widgets import resource_path

    try:
        for name, target in [("iris.csv", "species"), ("diabetes.csv", "disease_progression")]:
            df = load_dataset(resource_path("examples", name))
            exp, bundles = run_experiment(df, target, settings=Settings(cv_folds=2))
            failed = [r["name"] for r in exp.results if r["status"] != "ok"]
            if failed:
                raise RuntimeError(f"{name}: failed algorithms {failed}")
            with tempfile.TemporaryDirectory() as tmp:
                save_experiment(exp, bundles, tmp)
                load_model(model_path(exp.id, exp.best["name"], tmp)).predict(df.head())
    except Exception as exc:
        _say(f"Self-test failed: {exc}", sys.stderr)
        return 1
    _say("Self-test passed.", sys.stdout)
    return 0


def _say(message: str, stream) -> None:
    # A windowed executable has no console: stdout/stderr are None there.
    if stream is not None:
        print(message, file=stream)


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()

    import matplotlib

    matplotlib.use("QtAgg")
    matplotlib.rcParams.update({
        "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
    })

    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    if sys.platform == "win32":
        # Show our own icon in the taskbar instead of the Python one.
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MLperformance.App")

    app = QApplication(sys.argv)
    app.setApplicationName("MLperformance")
    app.setOrganizationName("MLperformance")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
