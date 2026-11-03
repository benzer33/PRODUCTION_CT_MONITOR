# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec สำหรับ AI Production CT Monitor
# รันด้วย:  pyinstaller build_exe.spec

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ---------------------------------------------------------------------------
# Collect mediapipe model files (สำคัญมาก — ถ้าไม่มีจะ crash)
# ---------------------------------------------------------------------------
mediapipe_datas = collect_data_files("mediapipe")

a = Analysis(
	["ai_production_monitor/run.py"],
	pathex=["ai_production_monitor"],
	binaries=[],
	datas=[
		# Config files
		("ai_production_monitor/config/*.json",  "config"),
		# mediapipe models
		*mediapipe_datas,
	],
	hiddenimports=[
		# PyQt5
		"PyQt5",
		"PyQt5.QtCore",
		"PyQt5.QtGui",
		"PyQt5.QtWidgets",
		# pyqtgraph
		"pyqtgraph",
		"pyqtgraph.graphicsItems",
		# mediapipe
		"mediapipe",
		*collect_submodules("mediapipe"),
		# OpenCV
		"cv2",
		# SQLAlchemy
		"sqlalchemy",
		"sqlalchemy.dialects.sqlite",
		# google-generativeai (Gemini)
		"google.generativeai",
		"google.ai.generativelanguage_v1beta",
		# python-dotenv
		"dotenv",
		# อื่นๆ
		"numpy",
		"scipy",
		"PIL",
	],
	hookspath=[],
	hooksconfig={},
	runtime_hooks=[],
	excludes=[],
	win_no_prefer_redirects=False,
	win_private_assemblies=False,
	cipher=block_cipher,
	noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
	pyz,
	a.scripts,
	[],
	exclude_binaries=True,
	name="ProductionCTMonitor",
	debug=False,
	bootloader_ignore_signals=False,
	strip=False,
	upx=True,
	console=False,          # False = ไม่มีหน้าต่าง CMD โผล่มา
	disable_windowed_traceback=False,
	target_arch=None,
	codesign_identity=None,
	entitlements_file=None,
	# icon="ai_production_monitor/assets/icon.ico",  # ← uncomment ถ้ามี icon
)

coll = COLLECT(
	exe,
	a.binaries,
	a.zipfiles,
	a.datas,
	strip=False,
	upx=True,
	upx_exclude=[],
	name="ProductionCTMonitor",
)
