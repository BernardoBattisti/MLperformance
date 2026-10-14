"""Launches the MLperformance desktop app (also the PyInstaller entry point)."""

import sys

from mlperformance.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
