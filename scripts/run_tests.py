from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def require(path: str) -> Path:
    p = ROOT / path
    if not p.exists():
        raise AssertionError(f"Missing required artifact: {path}")
    return p


def main() -> None:
    panel_path = require("data/processed/crosscity_daily_panel.csv")
    metrics_path = require("results/tables/table_02_model_metrics.csv")
    manifest_path = require("data_manifest.md")
    registry_path = require("source_registry.md")
    for fig in [
        "results/figures/figure_01_workflow.pdf",
        "results/figures/figure_02_daily_demand.pdf",
        "results/figures/figure_03_model_mae.pdf",
    ]:
        require(fig)
    panel = pd.read_csv(panel_path)
    metrics = pd.read_csv(metrics_path)
    assert {"date", "city", "trips", "split", "lag_1"}.issubset(panel.columns)
    assert panel["city"].nunique() >= 3
    assert (panel["split"] == "test").sum() > 0
    assert {"model", "MAE", "RMSE", "MAPE_percent", "R2"}.issubset(metrics.columns)
    assert metrics["MAE"].notna().all()
    assert manifest_path.read_text(encoding="utf-8").count("sha256") >= 1
    assert "OpenStreetMap" in registry_path.read_text(encoding="utf-8")
    print("Tests passed: 8 checks")


if __name__ == "__main__":
    main()
