# วิธีสร้าง EXE สำหรับ AI Production CT Monitor
# =============================================

## ขั้นตอนที่ 1 — ติดตั้ง PyInstaller
```powershell
pip install pyinstaller
```

## ขั้นตอนที่ 2 — Build
```powershell
cd D:\PROJECT\PYTHON\PRODUCTION_CT_MONITOR
pyinstaller build_exe.spec --clean
```

## ขั้นตอนที่ 3 — ผลลัพธ์
ไฟล์จะอยู่ที่:
```
dist\ProductionCTMonitor\
	ProductionCTMonitor.exe   ← ไฟล์รัน
	(+ DLL และ dependencies อื่นๆ)
```

## ขั้นตอนที่ 4 — สิ่งที่ต้องทำก่อนส่งให้คนอื่น

### 4.1 copy ไฟล์ .env ใส่เองด้วยมือ (ห้าม bundle ใน exe)
```
dist\ProductionCTMonitor\.env
```
เนื้อหา .env:
```
GOOGLE_API_KEY=ใส่ key จริงตรงนี้
```

### 4.2 ตรวจสอบว่า config ติดมาด้วย
```
dist\ProductionCTMonitor\config\default_config.json   ✅
dist\ProductionCTMonitor\config\PACKING_CD.json       ✅
```

## ⚠️ หมายเหตุสำคัญ

- **EXE ทำงานแบบ folder** ไม่ใช่ไฟล์เดียว (เพราะ mediapipe ใหญ่มาก)
- **ต้องรันบน Windows 64-bit** เท่านั้น (build บนเครื่องไหน ใช้ได้แค่ architecture นั้น)
- **ห้าม commit dist/ และ build/** ลง git (ขนาดใหญ่มาก)
- ขนาดโดยประมาณ: ~500MB–1GB (เพราะ mediapipe + OpenCV)
