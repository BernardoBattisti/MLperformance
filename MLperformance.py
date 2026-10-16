"""Launches the MLperformance desktop app (also the PyInstaller entry point)."""

import multiprocessing
import sys

from mlperformance.gui.app import main

if __name__ == "__main__":
    # Without this, a worker process started by the packaged app would open the app again.
    multiprocessing.freeze_support()
    sys.exit(main())
