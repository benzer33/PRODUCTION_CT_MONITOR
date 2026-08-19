"""
ai/anthropic_client.py
Anthropic Claude API integration for the AI Summary feature.

Usage
-----
    from ai.anthropic_client import AnthropicAnalyser

    analyser = AnthropicAnalyser(api_key="sk-...", model="claude-sonnet-4-5")

    # Synchronous (blocking — run in a QThread)
    result = analyser.analyse_session(session_stats, golden_standard, ...)

    # The analyser also provides a QThread-compatible wrapper (AnalysisWorker)
    # so the GUI never blocks during API calls.

API key
-------
Set via:
  1. Constructor parameter
  2. Environment variable  ANTHROPIC_API_KEY
  3. config/default_config.json  →  "anthropic_api_key"
Never hardcode keys in source files.
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
# AnthropicAnalyser
# ---------------------------------------------------------------------------

class AnthropicAnalyser:
    """
    Wraps the Anthropic Python SDK for session analysis.

    Raises
    ------
    ImportError  : if anthropic package is not installed
    ValueError   : if api_key is missing / placeholder
    RuntimeError : on API errors
    """

    PLACEHOLDER = "YOUR_API_KEY_HERE"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        try:
            import anthropic as _anthropic
            self._anthropic = _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc

        resolved_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY", "")
            or self.PLACEHOLDER
        )

        if not resolved_key or resolved_key == self.PLACEHOLDER:
            raise ValueError(
                "Anthropic API key not configured. "
                "Set ANTHROPIC_API_KEY environment variable or provide it "
                "in config/default_config.json → 'anthropic_api_key'."
            )

        self._client    = self._anthropic.Anthropic(api_key=resolved_key)
        self._model     = model
        self._max_tokens = max_tokens
        self._timeout   = timeout

    # ------------------------------------------------------------------
    # Public methods
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
        Send session statistics to Claude and return the analysis text.

        Parameters are the same as build_session_analysis_prompt().
        Returns the full text response from Claude.
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
        Thai-language analysis text from Claude
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
        try:
            message = self._client.messages.create(
                model      = self._model,
                max_tokens = self._max_tokens,
                messages   = [{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except self._anthropic.AuthenticationError:
            raise RuntimeError(
                "API key ไม่ถูกต้อง กรุณาตรวจสอบค่า ANTHROPIC_API_KEY ใน .env"
            )
        except self._anthropic.RateLimitError:
            raise RuntimeError(
                "เกินขีดจำกัดการใช้งาน Anthropic API กรุณาลองใหม่ในภายหลัง"
            )
        except self._anthropic.APIStatusError as exc:
            raise RuntimeError(
                f"Anthropic API ตอบกลับด้วย error {exc.status_code}: {exc.message}"
            )
        except TimeoutError:
            raise RuntimeError(
                "การเชื่อมต่อ Anthropic API หมดเวลา (timeout) กรุณาตรวจสอบอินเทอร์เน็ต"
            )
        except OSError:
            raise RuntimeError(
                "ไม่สามารถเชื่อมต่ออินเทอร์เน็ตได้ กรุณาตรวจสอบการเชื่อมต่อเครือข่าย"
            )
        except Exception as exc:
            raise RuntimeError(f"เกิดข้อผิดพลาดไม่คาดคิด: {exc}") from exc


# ---------------------------------------------------------------------------
# AnalysisWorker — QThread wrapper for non-blocking GUI use
# ---------------------------------------------------------------------------

class AnalysisWorker(QThread):
    """
    Run an Anthropic API call in a background thread so the GUI stays responsive.

    Signals
    -------
    analysis_ready(str)  — emitted with the response text on success
    analysis_error(str)  — emitted with error message on failure
    """

    analysis_ready = pyqtSignal(str)
    analysis_error = pyqtSignal(str)

    # ---------------------------------------------------------------
    # Mode A: full session_data dict (AI Summary Screen)
    # Mode B: legacy individual-field call (older callers)
    # ---------------------------------------------------------------

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        # Mode A — session_data dict
        session_data: dict | None = None,
        # Mode B — legacy individual fields
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
        self._session_data  = session_data          # Mode A
        self._station_name  = station_name          # Mode B
        self._session_stats = session_stats or {}
        self._golden        = golden_standard or {}
        self._zone_names    = zone_names or {}
        self._alert_history = alert_history or []
        self._operator_name = operator_name

    def run(self) -> None:
        try:
            analyser = AnthropicAnalyser(
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
