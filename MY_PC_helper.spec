# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['model_test_ui.py'],
    pathex=[],
    binaries=[('E:\\anaconda3\\envs\\esp_robot_env\\Lib\\site-packages\\xgboost\\lib\\xgboost.dll', 'xgboost\\lib')],
    datas=[('.\\label_encoder.joblib', '.'), ('.\\xgboost_model.joblib', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MY_PC_helper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MY_PC_helper',
)
