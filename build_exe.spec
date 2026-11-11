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

# ---------------------------------------------------------------------------
# Collect ultralytics data files (yaml configs, assets, cfg)
# ใช้ try/except เผื่อเครื่องที่ build ไม่ได้ติดตั้ง ultralytics
# ---------------------------------------------------------------------------
try:
	ultralytics_datas   = collect_data_files("ultralytics")
	ultralytics_hidden  = collect_submodules("ultralytics")
except Exception:
	ultralytics_datas   = []
	ultralytics_hidden  = []

# ---------------------------------------------------------------------------
# Bundle YOLO model weights (.pt) from ultralytics cache or local folder
# ไฟล์ .pt จะถูก bundle ไปที่ models/ ใน dist
# ถ้าไม่มีไฟล์ก็ข้ามไป (mediapipe ยังทำงานได้ปกติ)
# ---------------------------------------------------------------------------
import os, glob as _glob

def _find_pt_files():
	"""Find .pt files from common cache locations."""
	candidates = []
	# 1. Local folder next to spec file
	local = _glob.glob(os.path.join(os.path.dirname(os.path.abspath("build_exe.spec")), "*.pt"))
	candidates.extend(local)
	# 2. ultralytics default cache  %USERPROFILE%\AppData\Roaming\Ultralytics\
	cache = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Ultralytics")
	if os.path.isdir(cache):
		candidates.extend(_glob.glob(os.path.join(cache, "*.pt")))
	# 3. Current working directory
	candidates.extend(_glob.glob("*.pt"))
	seen = set()
	result = []
	for p in candidates:
		p = os.path.abspath(p)
		if p not in seen and os.path.isfile(p):
			seen.add(p)
			result.append((p, "models"))   # → dist/models/*.pt
	return result

yolo_model_datas = _find_pt_files()
print(f"[spec] Bundling {len(yolo_model_datas)} YOLO .pt file(s): {[d[0] for d in yolo_model_datas]}")

a = Analysis(
	["ai_production_monitor/run.py"],
	pathex=["ai_production_monitor"],
	binaries=[],
	datas=[
		# Config files
		("ai_production_monitor/config/*.json",  "config"),
		# mediapipe models
		*mediapipe_datas,
		# ultralytics yaml/assets
		*ultralytics_datas,
		# YOLO .pt weight files → dist/models/
		*yolo_model_datas,
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
		# ultralytics + torch (YOLO Pose backend — optional)
		*ultralytics_hidden,
		"torch",
		"torch.nn",
		"torch.nn.modules",
		"torch.cuda",
		"torch.backends.cuda",
		"torch.backends.cudnn",
		# torchvision excluded — conflicts with torch CPU in PyInstaller
		# torchvision is NOT needed for YOLO Pose inference
		"ultralytics",
		"ultralytics.models.yolo",
		"ultralytics.engine",
		"ultralytics.utils",
	],
	hookspath=[],
	hooksconfig={},
	runtime_hooks=["hooks/rthook_torch_preload.py"],
	excludes=[
		# torchvision not needed for YOLO Pose + causes nms operator conflict
		"torchvision",
	],
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
