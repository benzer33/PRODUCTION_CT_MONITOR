"""
ai/gemini_client.py
Google Gemini API integration for the AI Summary feature.

Drop-in replacement for ai/anthropic_client.py — exposes the same public
interface so gui/summary_screen.py only needs a single import change.

SDK
---
Uses the new  google-genai  package (google.genai), NOT the deprecated
google-generativeai package.

    pip install google-genai

Model choice
------------
Default: gemini-2.5-flash
  • Fast and capable — suitable for plain-text statistics summarisation
  • Fallback option: gemini-2.0-flash  (previous stable version)

API key
-------
Set via:
  1. Constructor parameter
  2. Environment variable  GOOGLE_API_KEY
  3. config/default_config.json  →  "google_api_key"
Never hardcode keys in source files.

Error handling
--------------
All API errors are caught and re-raised as RuntimeError with Thai messages
so the GUI can display them without crashing.
"""

from __future__ import annotations

import os
from typing import Any

from PyQt5.QtCore import QThread, pyqtSignal

from ai.prompt_templates import (
    build_session_analysis_prompt,
    build_summary_prompt,
    build_zone_improvement_prompt,
)


# ---------------------------------------------------------------------------
# GeminiAnalyser
# ---------------------------------------------------------------------------

class GeminiAnalyser:
    """
    Wraps the google-genai Python SDK for session analysis.

    Raises
    ------
    ImportError  : if google-genai package is not installed
    ValueError   : if api_key is missing / placeholder
    RuntimeError : on API errors (with Thai-language messages)
    """

    PLACEHOLDER = "YOUR_API_KEY_HERE"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-flash",
        max_tokens: int = 8192,
        timeout: float = 60.0,
    ) -> None:
        try:
            from google import genai as _genai
            self._genai = _genai
        except ImportError as exc:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            ) from exc

        resolved_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY", "")
            or self.PLACEHOLDER
        )

        if not resolved_key or resolved_key == self.PLACEHOLDER:
            raise ValueError(
                "Google API key not configured. "
                "Set GOOGLE_API_KEY environment variable or provide it "
                "in config/default_config.json → 'google_api_key'."
            )

        self._client     = self._genai.Client(api_key=resolved_key)
        self._model      = model
        self._max_tokens = max_tokens
        self._timeout    = timeout

    # ------------------------------------------------------------------
    # Public methods  (same signatures as AnthropicAnalyser)
    # ------------------------------------------------------------------

    def analyse_session(
        self,
        station_name: str,
        session_stats: dict[str, Any],
        golden_standard: dict[str, Any],
        zone_names: dict[int, str],
        alert_history: list[dict],
        operator_name: str | None = None,
    ) -> str:
        """
        Send session statistics to Gemini and return the analysis text.
        Parameters are identical to AnthropicAnalyser.analyse_session().
        """
        prompt = build_session_analysis_prompt(
            station_name    = station_name,
            session_stats   = session_stats,
            golden_standard = golden_standard,
            zone_names      = zone_names,
            alert_history   = alert_history,
            operator_name   = operator_name,
        )
        return self._call_api(prompt)

    def analyse_summary(self, session_data: dict) -> str:
        """
        Run Thai-language AI analysis on a completed session's numeric data.

        Parameters
        ----------
        session_data : output of DatabaseManager.get_session_full_data()

        Returns
        -------
        Thai-language analysis text from Gemini
        """
        prompt = build_summary_prompt(session_data)
        return self._call_api(prompt)

    def analyse_zone(
        self,
        zone_name: str,
        zone_times: list[float],
        standard_time: float,
        sequence_errors: list[str],
    ) -> str:
        """Zone-specific improvement analysis."""
        prompt = build_zone_improvement_prompt(
            zone_name       = zone_name,
            zone_times      = zone_times,
            standard_time   = standard_time,
            sequence_errors = sequence_errors,
        )
        return self._call_api(prompt)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str) -> str:
        """
        Call the Gemini API and return the response text.
        Maps google-genai exceptions → Thai RuntimeError messages.
        """
        try:
            from google.genai import types as _types

            # Build generation config:
            # - thinking_config: budget=0 disables chain-of-thought reasoning
            #   (not needed for stats summarisation; saves output token quota)
            # - max_output_tokens: raised to 8192 so responses are never cut short
            try:
                gen_config = _types.GenerateContentConfig(
                    max_output_tokens = self._max_tokens,
                    thinking_config   = _types.ThinkingConfig(thinking_budget=0),
                )
            except (AttributeError, TypeError):
                # Older SDK versions may not have ThinkingConfig — fall back gracefully
                gen_config = _types.GenerateContentConfig(
                    max_output_tokens = self._max_tokens,
                )

            response = self._client.models.generate_content(
                model    = self._model,
                contents = prompt,
                config   = gen_config,
            )

            # Log finish_reason so we can diagnose truncation
            try:
                candidate    = response.candidates[0]
                finish_reason = candidate.finish_reason
                print(f"[Gemini] finish_reason={finish_reason}  "
                      f"model={self._model}  max_tokens={self._max_tokens}")
                if str(finish_reason) in ("MAX_TOKENS", "2"):
                    print("[Gemini] WARNING: response was cut off by MAX_TOKENS — "
                          "consider increasing max_tokens further")
            except Exception:
                pass  # non-critical

            return response.text

        except Exception as exc:
            # Classify common error patterns from google-genai
            exc_str  = str(exc)
            exc_type = type(exc).__name__

            if "API_KEY_INVALID" in exc_str or "invalid" in exc_str.lower() and "key" in exc_str.lower():
                raise RuntimeError(
                    "API key ไม่ถูกต้อง กรุณาตรวจสอบค่า GOOGLE_API_KEY ใน .env"
                ) from exc

            if "RESOURCE_EXHAUSTED" in exc_str or "quota" in exc_str.lower() or "429" in exc_str:
                raise RuntimeError(
                    "เกินขีดจำกัดการใช้งาน Google Gemini API กรุณาลองใหม่ในภายหลัง"
                ) from exc

            if "PERMISSION_DENIED" in exc_str or "403" in exc_str:
                raise RuntimeError(
                    "ไม่มีสิทธิ์ใช้งาน Gemini API กรุณาตรวจสอบ API key และการเปิดใช้งาน Billing"
                ) from exc

            if "NOT_FOUND" in exc_str or "404" in exc_str:
                raise RuntimeError(
                    f"ไม่พบโมเดล '{self._model}' กรุณาตรวจสอบชื่อโมเดลใน gemini_client.py"
                ) from exc

            if "DeadlineExceeded" in exc_type or "timeout" in exc_str.lower() or "timed out" in exc_str.lower():
                raise RuntimeError(
                    "การเชื่อมต่อ Gemini API หมดเวลา (timeout) กรุณาตรวจสอบอินเทอร์เน็ต"
                ) from exc

            if isinstance(exc, OSError) or "network" in exc_str.lower() or "connection" in exc_str.lower():
                raise RuntimeError(
                    "ไม่สามารถเชื่อมต่ออินเทอร์เน็ตได้ กรุณาตรวจสอบการเชื่อมต่อเครือข่าย"
                ) from exc

            raise RuntimeError(f"เกิดข้อผิดพลาดไม่คาดคิด: {exc}") from exc


# ---------------------------------------------------------------------------
# AnalysisWorker — QThread wrapper (same interface as anthropic_client.py)
# ---------------------------------------------------------------------------

class AnalysisWorker(QThread):
    """
    Run a Gemini API call in a background thread so the GUI stays responsive.

    This class has an identical public interface to the AnalysisWorker in
    anthropic_client.py — gui/summary_screen.py only needs the import changed.

    Signals
    -------
    analysis_ready(str)  — emitted with the response text on success
    analysis_error(str)  — emitted with a Thai error message on failure
    """

    analysis_ready = pyqtSignal(str)
    analysis_error = pyqtSignal(str)

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash",
        # Mode A — session_data dict (AI Summary Screen)
        session_data: dict | None = None,
        # Mode B — legacy individual fields (older callers)
        station_name: str = "",
        session_stats: dict | None = None,
        golden_standard: dict | None = None,
        zone_names: dict | None = None,
        alert_history: list | None = None,
        operator_name: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._api_key       = api_key
        self._model         = model
        self._session_data  = session_data           # Mode A
        self._station_name  = station_name           # Mode B
        self._session_stats = session_stats or {}
        self._golden        = golden_standard or {}
        self._zone_names    = zone_names or {}
        self._alert_history = alert_history or []
        self._operator_name = operator_name

    def run(self) -> None:
        try:
            analyser = GeminiAnalyser(
                api_key = self._api_key,
                model   = self._model,
            )
            if self._session_data is not None:
                # Mode A: AI Summary Screen path
                text = analyser.analyse_summary(self._session_data)
            else:
                # Mode B: legacy path
                text = analyser.analyse_session(
                    station_name    = self._station_name,
                    session_stats   = self._session_stats,
                    golden_standard = self._golden,
                    zone_names      = self._zone_names,
                    alert_history   = self._alert_history,
                    operator_name   = self._operator_name,
                )
            self.analysis_ready.emit(text)
        except Exception as exc:
            self.analysis_error.emit(str(exc))
