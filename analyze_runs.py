from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import yaml


@dataclass
class RunSummary:
    name: str
    model: Optional[str]
    imgsz: Optional[int]
    batch: Optional[int]
    epochs: Optional[int]
    best_epoch: Optional[int]
    best_map50_95: Optional[float]
    best_map50: Optional[float]
    final_epoch: Optional[int]
    final_map50_95: Optional[float]
    final_map50: Optional[float]


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def load_args(run_dir: Path) -> tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        return None, None, None, None

    with args_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model = cfg.get("model")
    imgsz = cfg.get("imgsz")
    batch = cfg.get("batch")
    epochs = cfg.get("epochs")
    return model, imgsz, batch, epochs


def summarize_run(run_dir: Path) -> Optional[RunSummary]:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return None

    with results_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # Find row with best mAP50-95
    best_row = None
    best_score = float("-inf")
    for row in rows:
        score = _to_float(row.get("metrics/mAP50-95(B)", "") or "")
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_row = row

    last = rows[-1]
    model, imgsz, batch, epochs = load_args(run_dir)

    def get_metric(row: dict, key: str) -> Optional[float]:
        return _to_float(row.get(key, "") or "")

    best_epoch = int(best_row["epoch"]) if best_row and best_row.get("epoch") is not None else None
    final_epoch = int(last["epoch"]) if last.get("epoch") is not None else None

    return RunSummary(
        name=run_dir.name,
        model=model,
        imgsz=int(imgsz) if isinstance(imgsz, (int, float)) or (isinstance(imgsz, str) and imgsz.isdigit()) else None,
        batch=int(batch) if isinstance(batch, (int, float)) or (isinstance(batch, str) and str(batch).isdigit()) else None,
        epochs=int(epochs) if isinstance(epochs, (int, float)) or (isinstance(epochs, str) and str(epochs).isdigit()) else None,
        best_epoch=best_epoch,
        best_map50_95=get_metric(best_row, "metrics/mAP50-95(B)") if best_row else None,
        best_map50=get_metric(best_row, "metrics/mAP50(B)") if best_row else None,
        final_epoch=final_epoch,
        final_map50_95=get_metric(last, "metrics/mAP50-95(B)"),
        final_map50=get_metric(last, "metrics/mAP50(B)"),
    )


def collect_summaries(runs_dir: Path) -> List[RunSummary]:
    summaries: List[RunSummary] = []
    for sub in sorted(runs_dir.iterdir(), key=lambda p: p.name):
        if not sub.is_dir():
            continue
        summary = summarize_run(sub)
        if summary is not None:
            summaries.append(summary)
    return summaries


def print_table(summaries: List[RunSummary]) -> None:
    header = (
        f"{'run':<16} {'model':<10} {'img':>4} {'bs':>3} "
        f"{'epochs':>6}  {'best@ep':>7} {'best50':>7} {'best50-95':>9}  "
        f"{'final@ep':>8} {'final50':>8} {'final50-95':>10}"
    )
    print(header)
    print("-" * len(header))

    for s in summaries:
        def fmt(v: Optional[float], digits: int = 3) -> str:
            return f"{v:.{digits}f}" if v is not None else "---"

        print(
            f"{s.name:<16} "
            f"{(s.model or ''):<10} "
            f"{(s.imgsz or 0):>4d} "
            f"{(s.batch or 0):>3d} "
            f"{(s.epochs or 0):>6d}  "
            f"{(s.best_epoch or 0):>7d} "
            f"{fmt(s.best_map50):>7} "
            f"{fmt(s.best_map50_95):>9}  "
            f"{(s.final_epoch or 0):>8d} "
            f"{fmt(s.final_map50):>8} "
            f"{fmt(s.final_map50_95):>10}"
        )


def main() -> None:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        raise SystemExit("No se encontró la carpeta 'runs'. Ejecuta antes un entrenamiento.")

    summaries = collect_summaries(runs_dir)
    if not summaries:
        raise SystemExit("No se encontraron 'results.csv' en 'runs/*'.")

    print_table(summaries)


if __name__ == "__main__":
    main()

