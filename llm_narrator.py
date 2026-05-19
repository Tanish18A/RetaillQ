"""
LLM Demand Narrator — Google Gemini Integration
=================================================
Generates human-readable, expert-level demand intelligence narratives
using Gemini Pro. Combines ML output + festival calendar + drift signals
into actionable business language.

This replaces generic SHAP text with context-aware explanations
that a supply chain manager can actually use.
"""

import os
import json
import re
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")


def _clean_md(text: str) -> str:
    """Remove markdown bold/italic for clean display."""
    text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    return text.strip()


def build_context_payload(
    store_id:    str,
    product_name: str,
    forecast_total: float,
    avg_daily:   float,
    stock_status: str,
    curr_inv:    float,
    order_qty:   float,
    festival_window: str,
    trust_score: float,
    trust_label: str,
    drift_level: str,
    changepoints: list,
    causal_lift: float,
    naive_lift:  float,
    cost_opt_qty: float,
    cost_savings: float,
    mc_stockout_prob: float,
    mc_profit_p5: float,
    mc_profit_mean: float,
) -> str:
    """Build a structured context string for Gemini."""
    cp_str  = ", ".join(changepoints[-3:]) if changepoints else "None detected"
    fest_str = festival_window if festival_window else "No festival in next 14 days"

    return f"""You are Riya, a sharp senior supply chain analyst at a D-Mart style retail chain in India. You have just reviewed the AI system's output for one SKU and need to brief the store manager in 4-5 sentences — conversational, no jargon, no bullet points, no headers, no markdown.

Data you reviewed:
- Store: {store_id} | Product: {product_name}
- 7-day forecast: {forecast_total:.0f} units (avg {avg_daily:.1f}/day) | Trust: {trust_score:.0f}/100 ({trust_label})
- Current stock: {curr_inv:.0f} units | Status: {stock_status}
- Festival window: {fest_str}
- True causal lift: ×{causal_lift:.2f} (naive was ×{naive_lift:.2f} — {(naive_lift-causal_lift)*100:.1f}% was confounding bias)
- Data drift: {drift_level} | Changepoints: {cp_str}
- Recommended order: {order_qty:.0f} units | Cost-optimal: {cost_opt_qty:.0f} units | Saves ₹{cost_savings:,.0f} vs heuristic
- Monte Carlo stockout risk: {mc_stockout_prob*100:.1f}% over 30 days
- Expected profit: ₹{mc_profit_p5:,.0f} (pessimistic) to ₹{mc_profit_mean:,.0f} (mean)

Write your brief to the store manager now. Start directly — no "Based on the data" or "As a supply chain analyst". Sound like you're talking to a colleague, not writing a report. Mention the most important number. End with one clear action."""


class GeminiNarrator:
    """
    Google Gemini-powered demand intelligence narrator.
    Generates expert-level narrative explanations for inventory decisions.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client = None
        self._available = False

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel("gemini-2.0-flash")
                self._available = True
            except Exception as e:
                self._available = False
                print(f"⚠️ Gemini init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def generate_narrative(self, context: str) -> str:
        """Call Gemini and return a plain-text expert narrative."""
        if not self._available:
            return self._fallback_narrative(context)

        try:
            resp = self._client.generate_content(
                context,
                generation_config={
                    "temperature":      0.75,
                    "max_output_tokens": 320,
                    "top_p":            0.92,
                }
            )
            text = resp.text if hasattr(resp, "text") else str(resp)
            return _clean_md(text)
        except Exception as e:
            return self._fallback_narrative(context, error=str(e))

    def generate_alert_narrative(
        self, store: str, product: str,
        stock_status: str, order_qty: float,
        days_of_stock: float, mc_stockout_pct: float,
        festival_note: str,
    ) -> str:
        """Generate a short alert narrative for the alerts tab."""
        prompt = (
            f"Write ONE sentence (max 20 words) for a retail store manager alert. "
            f"Store: {store}. Product: {product}. "
            f"Status: {stock_status}. Days of stock: {days_of_stock:.1f}. "
            f"Stockout risk: {mc_stockout_pct:.1f}%. "
            f"Order recommended: {order_qty:.0f} units. "
            f"{festival_note}. "
            f"Be direct and urgent if critical. No markdown."
        )
        if not self._available:
            return f"Order {order_qty:.0f} units — {days_of_stock:.1f} days of stock remaining."

        try:
            resp = self._client.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 60}
            )
            return _clean_md(resp.text if hasattr(resp, "text") else str(resp))
        except Exception:
            return f"Order {order_qty:.0f} units — {days_of_stock:.1f} days of stock remaining."

    def _fallback_narrative(self, context: str, error: str = "") -> str:
        """Deterministic fallback when Gemini unavailable."""
        lines = context.split("\n")
        parts = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                parts[k.strip()] = v.strip()

        store   = parts.get("STORE", "this store")
        product = parts.get("PRODUCT", "this product")

        return (
            f"Gemini API not configured — displaying rule-based narrative. "
            f"Based on ML forecast, demand patterns for {product} appear stable. "
            f"Please add your GEMINI_API_KEY in the sidebar to enable AI-generated narratives. "
            f"Error: {error}" if error else
            f"Add your GEMINI_API_KEY in the sidebar Settings panel to activate AI narratives."
        )
