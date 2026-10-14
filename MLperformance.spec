# PyInstaller build recipe. Build with:  pyinstaller MLperformance.spec
# Output: dist/MLperformance/MLperformance.exe (plus its support files).

a = Analysis(
    ["MLperformance.py"],
    datas=[
        ("assets/icon.png", "assets"),
        ("assets/icon.ico", "assets"),
        ("examples/*.csv", "examples"),
    ],
    hiddenimports=["sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes"],
    excludes=["tkinter", "IPython", "jupyter", "notebook", "pytest", "streamlit"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MLperformance",
    icon="assets/icon.ico",
    console=False,  # a window app, no terminal
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="MLperformance", upx=False)
