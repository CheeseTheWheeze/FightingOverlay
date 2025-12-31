from PyInstaller.utils.hooks import collect_all


cv2_datas, cv2_binaries, cv2_hidden = collect_all("cv2")

a = Analysis(
    ["apps/control_center/main.py"],
    pathex=["."],
    binaries=cv2_binaries,
    datas=cv2_datas,
    hiddenimports=cv2_hidden + ["cv2"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ControlCenter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="ControlCenter",
)
