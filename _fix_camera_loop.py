import re, pathlib

path = pathlib.Path(r"D:\PROJECT\PYTHON\PRODUCTION_CT_MONITOR\ai_production_monitor\vision\point_tracker_thread.py")
text = path.read_text(encoding="utf-8")

old = (
    r"(# \u2500\u2500 Main frame loop \u2500+\r?\n"
    r"        while self\._running:\r?\n"
    r"            ok, frame = self\._camera\.read\(\)\r?\n"
    r"            if not ok or frame is None:\r?\n"
    r"                self\.error_occurred\.emit\(\"Camera read failed \u2014 retrying\u2026\"\)\r?\n"
    r"                time\.sleep\(0\.05\)\r?\n"
    r"                continue)"
)

new = (
    "# \u2500\u2500 Main frame loop \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "        _consecutive_fails = 0\n"
    "        _MAX_CONSECUTIVE_FAILS = 30          # ~3 s at 100 ms sleep\n"
    "        while self._running:\n"
    "            ok, frame = self._camera.read()\n"
    "            if not ok or frame is None:\n"
    "                _consecutive_fails += 1\n"
    "                if _consecutive_fails == 1:\n"
    "                    self.error_occurred.emit(\"Camera read failed \u2014 retrying\u2026\")\n"
    "                if _consecutive_fails >= _MAX_CONSECUTIVE_FAILS:\n"
    "                    self.error_occurred.emit(\"Camera lost \u2014 stopping.\")\n"
    "                    self._running = False\n"
    "                    break\n"
    "                time.sleep(0.1)\n"
    "                continue\n"
    "            _consecutive_fails = 0"
)

result, n = re.subn(old, new, text, flags=re.MULTILINE)
if n:
    path.write_text(result, encoding="utf-8")
    print(f"OK: replaced {n} occurrence(s)")
else:
    print("NO MATCH — dumping search area for diagnosis:")
    idx = text.find("Main frame loop")
    print(repr(text[max(0,idx-5):idx+300]))
