from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# -----------------------------
# Optional PDF support (ReportLab)
# -----------------------------
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# =============================
# Models
# =============================

@dataclass
class Asset:
    name: str
    asset_type: str  # "crypto" | "stable" | "lp" | "token"
    value_usd: float
    annual_vol: float  # e.g., 0.80 = 80% annualized
    liquidity_haircut: float = 0.00     # forced liquidation haircut
    protocol_risk_haircut: float = 0.00 # applied in protocol incident scenarios


@dataclass
class Treasury:
    assets: List[Asset]
    monthly_burn_usd: float  # operating costs

    def total_value(self) -> float:
        return float(sum(a.value_usd for a in self.assets))

    def asset_names(self) -> List[str]:
        return [a.name for a in self.assets]


@dataclass
class Scenario:
    name: str
    shocks: Dict[str, float]  # deterministic simple returns by asset name
    apply_liquidity_haircut: bool = False
    apply_protocol_haircut: bool = False


# =============================
# Helpers: risk labels + warnings
# =============================

def traffic_light_label(runway_months: float, green_min: float = 24.0, amber_min: float = 12.0) -> str:
    """
    GREEN: runway >= green_min
    AMBER: amber_min <= runway < green_min
    RED: runway < amber_min
    """
    if runway_months >= green_min:
        return "GREEN"
    if runway_months >= amber_min:
        return "AMBER"
    return "RED"


def trigger_warnings(
    mc_report: dict,
    scenario_reports: List[dict],
    warn_runway_below_months: float = 18.0
) -> List[str]:
    """
    Returns a list of plain-English warnings.
    """
    warnings = []

    # Monte Carlo tail runway warnings
    if mc_report.get("worst_runway_months", float("inf")) < warn_runway_below_months:
        warnings.append(
            f"WARNING: In worst simulated months, runway can drop below {warn_runway_below_months:.0f} months."
        )
    if mc_report.get("p05_runway_months", float("inf")) < warn_runway_below_months:
        warnings.append(
            f"WARNING: In the 5% worst outcomes, runway can drop below {warn_runway_below_months:.0f} months."
        )

    # Deterministic scenario warnings
    for rep in scenario_reports:
        if rep["runway_months"] < warn_runway_below_months:
            warnings.append(
                f"WARNING: Scenario '{rep['scenario']}' reduces runway to {rep['runway_months']:.1f} months "
                f"(below {warn_runway_below_months:.0f})."
            )

    return warnings


# =============================
# Core stress / risk functions
# =============================

def annual_to_monthly_vol(annual_vol: float) -> float:
    return float(annual_vol / math.sqrt(12.0))


def portfolio_value_after_returns(
    treasury: Treasury,
    asset_returns: np.ndarray,
    apply_liquidity: bool = False,
    apply_protocol: bool = False,
) -> float:
    if asset_returns.shape[0] != len(treasury.assets):
        raise ValueError("asset_returns length must match number of assets.")

    total = 0.0
    for a, r in zip(treasury.assets, asset_returns):
        v = a.value_usd * (1.0 + float(r))
        if apply_liquidity and a.liquidity_haircut > 0:
            v *= (1.0 - a.liquidity_haircut)
        if apply_protocol and a.protocol_risk_haircut > 0:
            v *= (1.0 - a.protocol_risk_haircut)
        total += v
    return float(total)


def runway_months(treasury_value_usd: float, monthly_burn_usd: float) -> float:
    if monthly_burn_usd <= 0:
        return float("inf")
    return float(treasury_value_usd / monthly_burn_usd)


def var_cvar(losses: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    if losses.size == 0:
        raise ValueError("losses array is empty.")
    q = float(np.quantile(losses, alpha))
    tail = losses[losses >= q]
    cvar = float(np.mean(tail)) if tail.size > 0 else q
    return q, cvar


def apply_scenario_shocks(treasury: Treasury, scenario: Scenario) -> np.ndarray:
    r = np.zeros(len(treasury.assets), dtype=float)
    for i, a in enumerate(treasury.assets):
        r[i] = scenario.shocks.get(a.name, 0.0)
    return r


# =============================
# Monte Carlo simulation
# =============================

def simulate_correlated_returns(
    treasury: Treasury,
    corr: np.ndarray,
    n_sims: int = 50_000,
    horizon_months: int = 1,
    seed: Optional[int] = 42,
) -> np.ndarray:
    n = len(treasury.assets)
    if corr.shape != (n, n):
        raise ValueError(f"corr must be shape ({n},{n}).")

    rng = np.random.default_rng(seed)

    # Make correlation matrix Cholesky-able (jitter if needed)
    jitter = 1e-10
    for _ in range(8):
        try:
            L = np.linalg.cholesky(corr + np.eye(n) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        raise ValueError("Correlation matrix not positive definite (even with jitter).")

    monthly_vols = np.array([annual_to_monthly_vol(a.annual_vol) for a in treasury.assets], dtype=float)
    vol_h = monthly_vols * math.sqrt(max(1, horizon_months))

    z = rng.standard_normal(size=(n_sims, n))
    x = z @ L.T
    log_rets = x * vol_h
    simple_rets = np.expm1(log_rets)  # exp(logret) - 1
    return simple_rets


def monte_carlo_risk_report(
    treasury: Treasury,
    corr: np.ndarray,
    n_sims: int = 50_000,
    horizon_months: int = 1,
    alpha: float = 0.95,
) -> Dict[str, float]:
    initial = treasury.total_value()
    sim_rets = simulate_correlated_returns(treasury, corr, n_sims=n_sims, horizon_months=horizon_months)

    stressed_values = np.array(
        [portfolio_value_after_returns(treasury, sim_rets[i, :]) for i in range(n_sims)],
        dtype=float
    )

    losses = np.maximum(0.0, initial - stressed_values)
    var_a, cvar_a = var_cvar(losses, alpha=alpha)

    worst = float(np.min(stressed_values))
    p01 = float(np.quantile(stressed_values, 0.01))
    p05 = float(np.quantile(stressed_values, 0.05))
    median = float(np.quantile(stressed_values, 0.50))

    return {
        "initial_value": initial,
        "horizon_months": float(horizon_months),
        f"VaR_{int(alpha*100)}": var_a,
        f"CVaR_{int(alpha*100)}": cvar_a,
        "worst_value": worst,
        "p01_value": p01,
        "p05_value": p05,
        "median_value": median,
        "worst_runway_months": runway_months(worst, treasury.monthly_burn_usd),
        "p05_runway_months": runway_months(p05, treasury.monthly_burn_usd),
    }


def deterministic_scenario_report(
    treasury: Treasury,
    scenario: Scenario
) -> Dict[str, float]:
    initial = treasury.total_value()
    r = apply_scenario_shocks(treasury, scenario)

    stressed = portfolio_value_after_returns(
        treasury,
        r,
        apply_liquidity=scenario.apply_liquidity_haircut,
        apply_protocol=scenario.apply_protocol_haircut
    )
    loss = max(0.0, initial - stressed)
    dd = (loss / initial) if initial > 0 else 0.0
    run = runway_months(stressed, treasury.monthly_burn_usd)

    return {
        "scenario": scenario.name,
        "initial_value": initial,
        "stressed_value": stressed,
        "loss": loss,
        "drawdown_pct": dd * 100.0,
        "runway_months": run,
        "risk_label": traffic_light_label(run),
    }


# =============================
# Plain-English explanation
# =============================

def explain_results(mc_report: dict, scenario_reports: List[dict], monthly_burn: float) -> str:
    lines = []
    lines.append("\n=== Plain-English Risk Summary ===\n")

    # Monte Carlo (probabilistic)
    lines.append("Overall Treasury Risk (1-Month Outlook):")
    lines.append(
        f"- The treasury starts at ${mc_report['initial_value']:,.0f} with monthly operating costs of ${monthly_burn:,.0f}."
    )
    lines.append(
        f"- In 95% of realistic short-term market conditions, losses should not exceed about ${mc_report['VaR_95']:,.0f}."
    )
    lines.append(
        f"- In unusually bad market conditions (the tail), the average loss is closer to ${mc_report['CVaR_95']:,.0f}."
    )
    lines.append(
        f"- In the worst simulated month, the treasury could fall to ${mc_report['worst_value']:,.0f}, "
        f"which is about {mc_report['worst_runway_months']:.1f} months of runway."
    )
    lines.append(
        "- Bottom line: the organization remains financially viable even under extreme short-term stress."
    )

    # Deterministic scenarios (story-based)
    lines.append("\nSpecific Stress Scenarios:")
    for rep in scenario_reports:
        lines.append(f"\nScenario: {rep['scenario']}")
        lines.append(f"- Risk level (runway-based): {rep['risk_label']}")
        lines.append(f"- Estimated loss: ${rep['loss']:,.0f} ({rep['drawdown_pct']:.1f}% drawdown).")
        lines.append(
            f"- Remaining treasury: ${rep['stressed_value']:,.0f} "
            f"= about {rep['runway_months']:.1f} months of operating runway."
        )

        if rep["risk_label"] == "GREEN":
            lines.append("- Interpretation: Uncomfortable but manageable; no emergency action required.")
        elif rep["risk_label"] == "AMBER":
            lines.append("- Interpretation: Requires attention; consider tightening risk and/or expenses.")
        else:
            lines.append("- Interpretation: Serious threat; requires immediate governance action.")

    lines.append(
        "\nOverall Conclusion:\n"
        "- The primary risk driver is broad market drawdowns across correlated crypto assets.\n"
        "- Stablecoin or single-protocol events matter, but are less existential when exposure is sized correctly.\n"
        "- The key governance question is: how much runway do we want to guarantee under stress?"
    )
    return "\n".join(lines)


# =============================
# 1-page PDF generator
# =============================

def generate_governance_pdf(
    output_path: str,
    treasury: Treasury,
    mc_report: dict,
    scenario_reports: List[dict],
    warn_runway_below_months: float = 18.0,
    green_min: float = 24.0,
    amber_min: float = 12.0,
) -> None:
    """
    Creates a single-page PDF suitable for governance proposals.
    Requires reportlab: pip install reportlab
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed. Install it with: pip install reportlab")

    W, H = letter
    c = canvas.Canvas(output_path, pagesize=letter)
    margin = 0.75 * inch
    x = margin
    y = H - margin

    def draw_text(txt: str, font="Helvetica", size=10, dy=14):
        nonlocal y
        c.setFont(font, size)
        c.drawString(x, y, txt)
        y -= dy

    def draw_divider():
        nonlocal y
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(1)
        c.line(x, y, W - margin, y)
        y -= 12

    def label_color(label: str):
        if label == "GREEN":
            return colors.green
        if label == "AMBER":
            # amber-ish
            return colors.Color(1.0, 0.65, 0.0)
        return colors.red

    def draw_label_badge(label: str, label_x: float, label_y: float):
        c.setFillColor(label_color(label))
        c.roundRect(label_x, label_y - 10, 58, 16, 4, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(label_x + 29, label_y - 7, label)
        c.setFillColor(colors.black)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "DAO Treasury Risk Snapshot (1 page)")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Treasury value: ${treasury.total_value():,.0f}   |   Monthly burn: ${treasury.monthly_burn_usd:,.0f}")
    y -= 18
    draw_divider()

    # Monte Carlo section
    draw_text("Probabilistic Risk (Monte Carlo, 1-month):", font="Helvetica-Bold", size=11, dy=16)
    draw_text(f"VaR 95% (loss): ${mc_report['VaR_95']:,.0f}")
    draw_text(f"CVaR 95% (avg loss in worst tail): ${mc_report['CVaR_95']:,.0f}")
    draw_text(f"Worst simulated month value: ${mc_report['worst_value']:,.0f}")
    draw_text(f"Worst simulated month runway: {mc_report['worst_runway_months']:.1f} months")

    mc_label = traffic_light_label(mc_report["p05_runway_months"], green_min=green_min, amber_min=amber_min)
    draw_text(f"Risk label (based on 5% worst runway): {mc_label}")
    draw_label_badge(mc_label, W - margin - 70, H - margin - 44)

    draw_divider()

    # Deterministic scenarios
    draw_text("Deterministic Stress Scenarios:", font="Helvetica-Bold", size=11, dy=16)

    # Fit content: show top 4 scenarios (or fewer)
    max_scenarios = 4
    for i, rep in enumerate(scenario_reports[:max_scenarios], start=1):
        draw_text(f"{i}. {rep['scenario']}", font="Helvetica-Bold", size=10, dy=14)
        draw_text(f"   Loss: ${rep['loss']:,.0f} ({rep['drawdown_pct']:.1f}% drawdown)")
        draw_text(f"   Remaining: ${rep['stressed_value']:,.0f} | Runway: {rep['runway_months']:.1f} months")
        draw_text(f"   Risk label: {rep['risk_label']}")
        # badge aligned to right
        draw_label_badge(rep["risk_label"], W - margin - 70, y + 42)
        y -= 4

    draw_divider()

    # Warnings
    warnings = trigger_warnings(mc_report, scenario_reports, warn_runway_below_months=warn_runway_below_months)
    draw_text("Warnings / Triggers:", font="Helvetica-Bold", size=11, dy=16)
    if not warnings:
        draw_text(f"- No triggers breached (runway stayed above {warn_runway_below_months:.0f} months in tested outputs).")
    else:
        for w in warnings[:5]:  # keep it 1-page
            draw_text(f"- {w}")

    draw_divider()

    # Plain-English bottom line (short)
    draw_text("Bottom Line (plain English):", font="Helvetica-Bold", size=11, dy=16)
    draw_text("- This report estimates how much treasury value and operating runway could shrink under stress.")
    draw_text("- The biggest risk is broad market crashes across correlated crypto assets.")
    draw_text("- Governance can use this to size risk exposure and set minimum runway targets before deploying funds.")

    c.showPage()
    c.save()


# =============================
# Example usage (edit these)
# =============================
if __name__ == "__main__":
    # ---- Configuration knobs ----
    WARN_RUNWAY_BELOW_MONTHS = 18.0   # trigger warning if runway < X months in any key output
    GREEN_MIN = 24.0                 # traffic-light threshold
    AMBER_MIN = 12.0                 # traffic-light threshold

    # ---- Example treasury (edit these to match reality) ----
    treasury = Treasury(
        assets=[
            Asset("USDC", "stable", value_usd=3_000_000, annual_vol=0.02, liquidity_haircut=0.002),
            Asset("DAI",  "stable", value_usd=1_000_000, annual_vol=0.03, liquidity_haircut=0.003),
            Asset("ETH",  "crypto", value_usd=2_000_000, annual_vol=0.80, liquidity_haircut=0.010),
            Asset("BTC",  "crypto", value_usd=1_500_000, annual_vol=0.65, liquidity_haircut=0.010),
            Asset("LP_Aave", "lp",  value_usd=1_000_000, annual_vol=0.20, liquidity_haircut=0.020, protocol_risk_haircut=0.15),
            Asset("DAO_TOKEN", "token", value_usd=500_000, annual_vol=1.20, liquidity_haircut=0.050),
        ],
        monthly_burn_usd=250_000
    )

    # ---- Correlation matrix in the same order as assets above ----
    corr = np.array([
        [1.00, 0.20, 0.05, 0.05, 0.10, 0.05],
        [0.20, 1.00, 0.05, 0.05, 0.10, 0.05],
        [0.05, 0.05, 1.00, 0.70, 0.45, 0.60],
        [0.05, 0.05, 0.70, 1.00, 0.40, 0.55],
        [0.10, 0.10, 0.45, 0.40, 1.00, 0.35],
        [0.05, 0.05, 0.60, 0.55, 0.35, 1.00],
    ], dtype=float)

    # ---- Monte Carlo report ----
    print("\n=== Monte Carlo Risk Report (1-month) ===")
    mc = monte_carlo_risk_report(treasury, corr, n_sims=50_000, horizon_months=1, alpha=0.95)
    for k, v in mc.items():
        if "value" in k or "VaR" in k or "CVaR" in k:
            print(f"{k:20s}: ${v:,.0f}")
        else:
            print(f"{k:20s}: {v:,.2f}")

    # ---- Deterministic scenarios ----
    scenarios = [
        Scenario(
            name="ETH -50%, BTC -40%, DAO token -70% (risk-off shock)",
            shocks={"ETH": -0.50, "BTC": -0.40, "DAO_TOKEN": -0.70, "LP_Aave": -0.12},
            apply_liquidity_haircut=True
        ),
        Scenario(
            name="Stablecoin depeg: USDC 0.92, DAI 0.95 (temporary impairment)",
            shocks={"USDC": -0.08, "DAI": -0.05},
            apply_liquidity_haircut=True
        ),
        Scenario(
            name="Protocol incident: LP_Aave loses 30% principal + protocol haircut",
            shocks={"LP_Aave": -0.30},
            apply_protocol_haircut=True,
            apply_liquidity_haircut=True
        ),
        Scenario(
            name="Liquidity crisis: forced unwind haircuts (no price move)",
            shocks={},
            apply_liquidity_haircut=True
        ),
    ]

    scenario_reports: List[dict] = []

    print("\n=== Deterministic Scenarios ===")
    for s in scenarios:
        rep = deterministic_scenario_report(treasury, s)
        # Apply traffic-light thresholds you set
        rep["risk_label"] = traffic_light_label(rep["runway_months"], green_min=GREEN_MIN, amber_min=AMBER_MIN)
        scenario_reports.append(rep)

        print(f"\nScenario: {rep['scenario']}")
        print(f"  Initial Value     : ${rep['initial_value']:,.0f}")
        print(f"  Stressed Value    : ${rep['stressed_value']:,.0f}")
        print(f"  Loss              : ${rep['loss']:,.0f}")
        print(f"  Drawdown          : {rep['drawdown_pct']:.2f}%")
        print(f"  Runway (months)   : {rep['runway_months']:.2f}")
        print(f"  Risk Label        : {rep['risk_label']}")

    # ---- Trigger warnings ----
    print("\n=== Warnings / Triggers ===")
    warnings = trigger_warnings(mc, scenario_reports, warn_runway_below_months=WARN_RUNWAY_BELOW_MONTHS)
    if not warnings:
        print(f"No triggers breached (runway stayed above {WARN_RUNWAY_BELOW_MONTHS:.0f} months in tested outputs).")
    else:
        for w in warnings:
            print(w)

    # ---- Plain-English explainer ----
    print(explain_results(mc, scenario_reports, treasury.monthly_burn_usd))

    # ---- Auto-generate 1-page governance PDF ----
    pdf_path = os.path.join(os.getcwd(), "dao_treasury_governance_snapshot.pdf")
    try:
        generate_governance_pdf(
            output_path=pdf_path,
            treasury=treasury,
            mc_report=mc,
            scenario_reports=scenario_reports,
            warn_runway_below_months=WARN_RUNWAY_BELOW_MONTHS,
            green_min=GREEN_MIN,
            amber_min=AMBER_MIN,
        )
        print(f"\nPDF generated: {pdf_path}")
    except Exception as e:
        print("\nPDF not generated (install reportlab if needed).")
        print(f"Reason: {e}")
