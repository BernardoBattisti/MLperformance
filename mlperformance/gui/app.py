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

    from PySide6.QtWidgets import QApplication

    if sys.platform == "win32":
        # Show our own icon in the taskbar instead of the Python one.
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MLperformance.App")

    app = QApplication(sys.argv)
    app.setApplicationName("MLperformance")
    app.setOrganizationName("MLperformance")
    app.setStyle("Fusion")

    # A second launch (e.g. a double click while the app is still loading)
    # brings the running window forward instead of opening another copy.
    if _activate_running_instance():
        return 0

    # Loading scikit-learn and matplotlib takes a few seconds in the packaged
    # app, so show something right away.
    splash = _splash_screen()
    front = [splash]  # the widget a second launch brings forward
    server = _listen_for_other_instances(lambda: front[0])  # noqa: F841 (kept alive until exit)
    splash.show()
    app.processEvents()

    import matplotlib

    matplotlib.use("QtAgg")
    matplotlib.rcParams.update({
        "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
    })

    from .main_window import MainWindow

    window = MainWindow()
    front[0] = window
    window.show()
    splash.finish(window)
    return app.exec()


def _splash_screen():
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QPainter, QPixmap
    from PySide6.QtWidgets import QSplashScreen

    from .resources import resource_path

    pixmap = QPixmap(320, 220)
    pixmap.fill(QColor("#ffffff"))
    icon = QPixmap(str(resource_path("assets", "icon.png"))).scaled(
        110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    painter = QPainter(pixmap)
    painter.drawPixmap((pixmap.width() - icon.width()) // 2, 32, icon)
    painter.setPen(QColor("#4a5260"))
    painter.drawText(QRect(0, 160, pixmap.width(), 30), Qt.AlignCenter, "Loading MLperformance…")
    painter.end()
    return QSplashScreen(pixmap)


def _instance_name() -> str:
    import getpass

    try:
        user = getpass.getuser()
    except Exception:  # no user name in the environment
        user = "user"
    return f"MLperformance-{user}"


def _activate_running_instance() -> bool:
    """Ask an already running app to show its window; True if one answered."""
    from PySide6.QtNetwork import QLocalSocket

    socket = QLocalSocket()
    socket.connectToServer(_instance_name())
    if not socket.waitForConnected(300):
        return False
    socket.write(b"show")
    socket.waitForBytesWritten(300)
    socket.disconnectFromServer()
    return True


def _listen_for_other_instances(current_widget):
    """Bring ``current_widget()`` to the front whenever the app is launched again."""
    from PySide6.QtNetwork import QLocalServer

    QLocalServer.removeServer(_instance_name())  # leftover from a crashed run
    server = QLocalServer()
    server.listen(_instance_name())

    def bring_to_front() -> None:
        server.nextPendingConnection()
        widget = current_widget()
        widget.showNormal()
        widget.raise_()
        widget.activateWindow()

    server.newConnection.connect(bring_to_front)
    return server


if __name__ == "__main__":
    sys.exit(main())
