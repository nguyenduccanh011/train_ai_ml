from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.leaderboard.schema import LeaderboardRow

ARTIFACT_ROOT_NAMES: tuple[str, ...] = ("results", "models", "trades", "predictions", "logs")
ARTIFACT_KIND_FILENAMES: dict[str, str] = {
    "config": "config.resolved.yaml",
    "metrics": "metrics.json",
    "predictions_meta": "predictions_meta.json",
    "ranking_row": "ranking_row.json",
    "trades": "trades.csv",
}


class ModelRecord(BaseModel):
    id: str
    name: str
    version: str
    status: str = "unknown"
    strategy: str
    feature_set: str
    entry_model: str
    exit_model: str
    market: str = "unknown"
    priority: int = 0
    visible_in_dashboard: bool = False
    created_at: str
    updated_at: str
    retired_at: str | None = None
    retired_reason: str | None = None

    model_config = {"extra": "forbid"}


class RunRecord(BaseModel):
    id: str
    model_id: str
    run_name: str
    config_hash: str
    config_path: str | None = None
    resolved_config_path: str | None = None
    status: str = "unknown"
    created_at: str
    completed_at: str | None = None

    model_config = {"extra": "forbid"}


class ArtifactRecord(BaseModel):
    id: str
    run_id: str
    kind: str
    path: str
    size_bytes: int = 0
    checksum: str | None = None
    created_at: str
    deleted_at: str | None = None

    model_config = {"extra": "forbid"}


class MetricsSnapshotRecord(BaseModel):
    id: str
    run_id: str
    wr: float
    pf: float
    total_pnl: float
    max_drawdown: float
    sharpe: float
    mdd_per_symbol: float
    yearly_consistency: float
    composite_score: float
    captured_at: str

    model_config = {"extra": "forbid"}


class AuditLogRecord(BaseModel):
    id: str
    entity_type: str
    entity_id: str
    action: str
    actor: str = "system"
    reason: str | None = None
    payload_json: dict[str, Any] = Field(default_factory=dict)
    created_at: str

    model_config = {"extra": "forbid"}


class DashboardBundle(BaseModel):
    model: ModelRecord
    run: RunRecord
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    metrics_snapshot: MetricsSnapshotRecord
    audit_log: list[AuditLogRecord] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class LeaderboardEntry(BaseModel):
    row: LeaderboardRow
    model: ModelRecord

    model_config = {"extra": "forbid"}


def leaderboard_row_to_dashboard_bundle(
    row: LeaderboardRow,
    *,
    root: str | Path | None = None,
    artifact_kinds: list[str] | tuple[str, ...] | None = None,
) -> DashboardBundle:
    kinds = tuple(artifact_kinds or ARTIFACT_KIND_FILENAMES.keys())
    model_id = model_id_for_row(row)
    run_dir = canonical_run_dir(row, root=root)
    created_at = str(row.generated_at)

    model = ModelRecord(
        id=model_id,
        name=row.run_name,
        version=row.config_hash[:12],
        status="active" if not row.superseded else "inactive",
        strategy=row.strategy,
        feature_set=row.feature_set,
        entry_model=row.entry_model,
        exit_model=row.exit_model_type,
        market=row.market,
        priority=0,
        visible_in_dashboard=bool(row.score_mode == "live" and not row.superseded),
        created_at=created_at,
        updated_at=created_at,
    )

    run = RunRecord(
        id=row.run_id,
        model_id=model_id,
        run_name=row.run_name,
        config_hash=row.config_hash,
        config_path=_path_to_str(run_dir / ARTIFACT_KIND_FILENAMES["config"]),
        resolved_config_path=_path_to_str(run_dir / ARTIFACT_KIND_FILENAMES["config"]),
        status="completed" if not row.superseded else "superseded",
        created_at=created_at,
        completed_at=created_at,
    )

    artifacts = [
        ArtifactRecord(
            id=_stable_id("artifact", row.run_id, kind, _path_to_str(canonical_artifact_path(row, kind, root=root))),
            run_id=row.run_id,
            kind=kind,
            path=_path_to_str(canonical_artifact_path(row, kind, root=root)),
            created_at=created_at,
        )
        for kind in kinds
    ]

    metrics_snapshot = MetricsSnapshotRecord(
        id=_stable_id("metrics", row.run_id, row.generated_at, row.config_hash),
        run_id=row.run_id,
        wr=row.wr,
        pf=row.pf,
        total_pnl=row.total_pnl,
        max_drawdown=row.max_drawdown,
        sharpe=row.sharpe,
        mdd_per_symbol=row.mdd_per_symbol,
        yearly_consistency=row.yearly_consistency,
        composite_score=row.composite_score,
        captured_at=created_at,
    )

    return DashboardBundle(
        model=model,
        run=run,
        artifacts=artifacts,
        metrics_snapshot=metrics_snapshot,
        audit_log=[],
    )


def model_id_for_row(row: LeaderboardRow) -> str:
    payload = "|".join(
        [
            row.strategy,
            row.feature_set,
            row.entry_model,
            row.exit_model_type,
            row.market,
            row.timeframe,
        ]
    )
    return _stable_id("model", payload)


def canonical_run_dir(row: LeaderboardRow, *, root: str | Path | None = None) -> Path:
    repo_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    results_root = repo_root / "results" / "experiments"
    bundle = _safe_path_component(row.bundle)
    run_name = _safe_path_component(row.run_name)
    if bundle == run_name:
        return results_root / run_name
    return results_root / bundle / run_name


def canonical_artifact_path(
    row: LeaderboardRow, kind: str, *, root: str | Path | None = None
) -> Path:
    filename = ARTIFACT_KIND_FILENAMES.get(kind, f"{_safe_path_component(kind)}.json")
    return canonical_run_dir(row, root=root) / filename


def artifact_root(kind: str, *, root: str | Path | None = None) -> Path:
    if kind not in ARTIFACT_ROOT_NAMES:
        raise ValueError(f"unknown artifact root: {kind}")
    repo_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return repo_root / kind


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _safe_path_component(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return cleaned or "_"


def _path_to_str(path: Path) -> str:
    return path.as_posix()
