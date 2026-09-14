"""End-to-end synthetic replication of the paper's empirical workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deephedge.agents import BSDeltaAgent, MAAEEMAgent, MinimumVarianceAgent, PPOBaselineAgent, SACBaselineAgent
from deephedge.analysis import compute_breakeven_costs, performance_summary, permutation_style_importance
from deephedge.config import StudyConfig
from deephedge.data import SyntheticMarketData, generate_synthetic_market_data
from deephedge.envs import evaluate_hedge
from deephedge.pricing import compute_eem_targets


def run_full_study(config: StudyConfig | None = None, output: str | Path | None = None) -> dict:
    """Run all synthetic experiments and optionally write JSON outputs."""

    cfg = config or StudyConfig()
    data = generate_synthetic_market_data(cfg)
    targets = compute_eem_targets(data.paths, strike=cfg.strike, rate=data.paths.config.rate)
    initial_premium = float(np.mean(targets.call_value[:, 0]))

    horse_race, raw_results = _run_horse_race(data, targets, cfg, initial_premium)
    ablation = _run_ablation(data, targets, cfg, initial_premium)
    robustness = _run_robustness(data, raw_results, initial_premium)
    term_structure = _run_term_structure(data, raw_results["MAA-EEM"])
    breakeven = compute_breakeven_costs(
        data.paths,
        targets,
        cfg.strike,
        BSDeltaAgent(cfg.strike, data.paths.config.rate),
        [SACBaselineAgent(), MAAEEMAgent(), MAAEEMAgent(long_horizon=True, name="Long-Horizon")],
    )
    feature_importance = permutation_style_importance(data.alpha_factors, raw_results["MAA-EEM"].hedge, data.feature_names)

    report = {
        "config": cfg.__dict__,
        "descriptive_statistics": data.describe(),
        "horse_race": horse_race,
        "ablation": ablation,
        "robustness": robustness,
        "term_structure": term_structure,
        "breakeven_cost_bps": breakeven,
        "feature_importance": feature_importance,
        "acceptance": {
            "maa_mse_below_bs": horse_race["MAA-EEM"]["hedging_mse"] < horse_race["BS-Delta"]["hedging_mse"],
            "long_horizon_turnover_below_maa": horse_race["Long-Horizon"]["turnover"] <= horse_race["MAA-EEM"]["turnover"],
            "no_eem_fails": ablation["Remove-EEM"]["hedging_mse"] > horse_race["MAA-EEM"]["hedging_mse"] * 10.0,
        },
    }

    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return report


def _run_horse_race(data: SyntheticMarketData, targets, cfg: StudyConfig, initial_premium: float):
    agents = [
        BSDeltaAgent(cfg.strike, data.paths.config.rate),
        MinimumVarianceAgent(),
        SACBaselineAgent(),
        PPOBaselineAgent(),
        MAAEEMAgent(),
        MAAEEMAgent(long_horizon=True, name="Long-Horizon"),
    ]
    table = {}
    raw = {}
    for agent in agents:
        result = evaluate_hedge(
            agent.name,
            data.paths,
            targets,
            agent.hedge(data.paths, targets),
            cfg.strike,
            cfg.transaction_cost_bps,
        )
        raw[agent.name] = result
        table[agent.name] = performance_summary(result, initial_premium, data.paths.dt)
    return table, raw


def _run_ablation(data: SyntheticMarketData, targets, cfg: StudyConfig, initial_premium: float):
    variants = [
        MAAEEMAgent(name="Full-MAA"),
        MAAEEMAgent(use_eem=False, name="Remove-EEM"),
        MAAEEMAgent(use_hurst=False, name="Remove-Hurst"),
        MAAEEMAgent(adversarial_strength=0.015, use_adversary=False, name="Remove-MAA"),
        MAAEEMAgent(long_horizon=True, name="Add-Long-Horizon"),
    ]
    table = {}
    for agent in variants:
        result = evaluate_hedge(agent.name, data.paths, targets, agent.hedge(data.paths, targets), cfg.strike, cfg.transaction_cost_bps)
        table[agent.name] = performance_summary(result, initial_premium, data.paths.dt)
    return table


def _run_robustness(data: SyntheticMarketData, raw_results: dict, initial_premium: float):
    out = {}
    bs = raw_results["BS-Delta"]
    maa = raw_results["MAA-EEM"]
    for regime, mask in data.regimes.items():
        path_mask = np.any(mask, axis=1)
        if not np.any(path_mask):
            continue
        bs_return = float(np.mean(bs.pnl[path_mask]) / max(initial_premium, 1e-8) * 100.0)
        maa_return = float(np.mean(maa.pnl[path_mask]) / max(initial_premium, 1e-8) * 100.0)
        out[regime] = {
            "sample_size": int(np.sum(path_mask)),
            "maa_return_pct": maa_return,
            "bs_return_pct": bs_return,
            "excess_return_pct": maa_return - bs_return,
        }
    return out


def _run_term_structure(data: SyntheticMarketData, maa_result):
    buckets = {
        "short_lt_30": data.maturity_days < 30,
        "medium_30_60": (data.maturity_days >= 30) & (data.maturity_days <= 60),
        "long_gt_60": data.maturity_days > 60,
        "otm_lt_097": data.moneyness < 0.97,
        "atm_097_103": (data.moneyness >= 0.97) & (data.moneyness <= 1.03),
        "itm_gt_103": data.moneyness > 1.03,
    }
    step_error = np.pad(np.abs(maa_result.tracking_error), ((0, 0), (0, 1)), mode="edge")
    return {
        name: float(np.mean(step_error[mask])) if np.any(mask) else float("nan")
        for name, mask in buckets.items()
    }
