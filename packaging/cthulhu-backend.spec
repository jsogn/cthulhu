# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_all

_ROOT = os.path.abspath(os.path.join(os.path.dirname(SPEC), ".."))

datas, binaries, hiddenimports = [], [], []
for package in ("scipy", "numpy"):
    collected_datas, collected_binaries, collected_hidden = collect_all(package)
    datas += collected_datas
    binaries += collected_binaries
    hiddenimports += collected_hidden

a = Analysis(
    [os.path.join(_ROOT, "backend", "launcher.py")],
    pathex=[os.path.join(_ROOT, "backend", "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="cthulhu-backend",
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="cthulhu-backend",
)
