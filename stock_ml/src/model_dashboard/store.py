from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_ml.src.leaderboard.schema import LeaderboardRow
from stock_ml.src.model_dashboard.schema import (
    ArtifactRecord,
    AuditLogRecord,
    DashboardBundle,
    LeaderboardEntry,
    MetricsSnapshotRecord,
    ModelRecord,
    RunRecord,
    leaderboard_row_to_dashboard_bundle,
)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "db" / "init" / "001_model_dashboard_schema.sql"


@dataclass(slots=True)
class PurgePlan:
    model_id: str
    run_id: str
    artifact_paths: list[str]


class DashboardStore:
    def __init__(self, db_path: str | Path, *, init_schema: bool = True) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if init_schema:
            self.initialize()

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    def upsert_bundle(
        self,
        bundle: DashboardBundle,
        *,
        actor: str = "system",
        reason: str | None = None,
    ) -> None:
        now = _now_iso()
        with self._connect() as conn:
            self._upsert_model(conn, bundle.model, now)
            self._upsert_run(conn, bundle.run, now)
            self._upsert_metrics_snapshot(conn, bundle.metrics_snapshot)
            self._upsert_artifacts(conn, bundle.artifacts)
            self._write_audit_record(
                conn,
                entity_type="model",
                entity_id=bundle.model.id,
                action="ingest",
                actor=actor,
                reason=reason,
                payload_json={
                    "run_id": bundle.run.id,
                    "artifact_kinds": [artifact.kind for artifact in bundle.artifacts],
                },
                created_at=now,
            )
            for record in bundle.audit_log:
                self._insert_audit_record(conn, record)

    def upsert_row(
        self,
        row: LeaderboardRow,
        *,
        root: str | Path | None = None,
        actor: str = "system",
        reason: str | None = None,
    ) -> DashboardBundle:
        bundle = leaderboard_row_to_dashboard_bundle(row, root=root)
        self.upsert_bundle(bundle, actor=actor, reason=reason)
        self._upsert_leaderboard_row(
            row,
            bundle.model.id,
            bundle.model.visible_in_dashboard,
            actor=actor,
            reason=reason,
        )
        return bundle

    def has_leaderboard_rows(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM leaderboard_rows LIMIT 1").fetchone()
        return row is not None

    def list_leaderboard_rows(
        self,
        *,
        market: str | None = None,
        market_family: str | None = None,
        timeframe: str | None = None,
        visible_only: bool = False,
        include_superseded: bool = True,
    ) -> list[LeaderboardRow]:
        return [
            entry.row
            for entry in self.list_leaderboard_entries(
                market=market,
                market_family=market_family,
                timeframe=timeframe,
                visible_only=visible_only,
                include_superseded=include_superseded,
            )
        ]

    def list_leaderboard_entries(
        self,
        *,
        market: str | None = None,
        market_family: str | None = None,
        timeframe: str | None = None,
        visible_only: bool = False,
        include_superseded: bool = True,
    ) -> list[LeaderboardEntry]:
        clauses: list[str] = []
        params: list[Any] = []
        if market is not None:
            clauses.append("lr.market = ?")
            params.append(market)
        if market_family is not None:
            clauses.append("lr.market_family = ?")
            params.append(market_family)
        if timeframe is not None:
            clauses.append("lr.timeframe = ?")
            params.append(timeframe)
        if visible_only:
            clauses.append("lr.visible_in_dashboard = 1")
        if not include_superseded:
            clauses.append("lr.superseded = 0")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT lr.row_json, m.id, m.name, m.version, m.status, m.strategy, m.feature_set, "
            "m.entry_model, m.exit_model, m.market, m.priority, m.visible_in_dashboard, "
            "m.created_at, m.updated_at, m.retired_at, m.retired_reason "
            "FROM leaderboard_rows lr "
            "JOIN models m ON m.id = lr.model_id"
            f"{where} ORDER BY composite_score DESC, generated_at DESC, run_id ASC"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        entries: list[LeaderboardEntry] = []
        for row in rows:
            row_json = LeaderboardRow.model_validate_json(row["row_json"])
            model = self._model_from_row(row)
            if model is None:
                continue
            entries.append(LeaderboardEntry(row=row_json, model=model))
        return entries

    def list_models(
        self,
        *,
        status: str | None = None,
        visible_only: bool = False,
        search: str | None = None,
        sort_by: str = "updated_at",
        sort_dir: str = "desc",
    ) -> list[ModelRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if visible_only:
            clauses.append("visible_in_dashboard = 1")
        if search:
            needle = f"%{search.strip().lower()}%"
            clauses.append(
                "("
                "lower(id) LIKE ? OR lower(name) LIKE ? OR lower(version) LIKE ? OR "
                "lower(status) LIKE ? OR lower(strategy) LIKE ? OR lower(feature_set) LIKE ? OR "
                "lower(entry_model) LIKE ? OR lower(exit_model) LIKE ? OR lower(market) LIKE ? OR "
                "lower(COALESCE(retired_reason, '')) LIKE ?"
                ")"
            )
            params.extend([needle] * 10)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order_map = {
            "created_at": "created_at",
            "id": "id",
            "market": "market",
            "name": "name",
            "priority": "priority",
            "retired_at": "retired_at",
            "status": "status",
            "updated_at": "updated_at",
            "version": "version",
        }
        order_col = order_map.get(sort_by, "updated_at")
        direction = "ASC" if str(sort_dir).lower() == "asc" else "DESC"
        query = (
            "SELECT id, name, version, status, strategy, feature_set, entry_model, exit_model, "
            "market, priority, visible_in_dashboard, created_at, updated_at, retired_at, "
            "retired_reason FROM models"
            f"{where} ORDER BY {order_col} {direction}, id ASC"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._model_from_row(row) for row in rows]

    def list_audit_log(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        action: str | None = None,
        limit: int = 200,
    ) -> list[AuditLogRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if entity_type is not None:
            clauses.append("entity_type = ?")
            params.append(entity_type)
        if entity_id is not None:
            clauses.append("entity_id = ?")
            params.append(entity_id)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT id, entity_type, entity_id, action, actor, reason, payload_json, created_at "
            "FROM audit_log"
            f"{where} ORDER BY created_at DESC, id DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, [*params, max(1, int(limit))]).fetchall()
        return [self._audit_from_row(row) for row in rows]

    def get_model(self, model_id: str) -> ModelRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, version, status, strategy, feature_set, entry_model, exit_model, "
                "market, priority, visible_in_dashboard, created_at, updated_at, retired_at, "
                "retired_reason FROM models WHERE id = ?",
                (model_id,),
            ).fetchone()
        return self._model_from_row(row) if row is not None else None

    def get_bundle(self, model_id: str) -> DashboardBundle | None:
        model = self.get_model(model_id)
        if model is None:
            return None
        with self._connect() as conn:
            run_row = conn.execute(
                "SELECT id, model_id, run_name, config_hash, config_path, resolved_config_path, "
                "status, created_at, completed_at FROM runs WHERE model_id = ? "
                "ORDER BY COALESCE(completed_at, created_at) DESC, id DESC LIMIT 1",
                (model_id,),
            ).fetchone()
            if run_row is None:
                return None
            artifacts = conn.execute(
                "SELECT id, run_id, kind, path, size_bytes, checksum, created_at, deleted_at "
                "FROM artifacts WHERE run_id = ? ORDER BY kind ASC, id ASC",
                (run_row["id"],),
            ).fetchall()
            metrics_row = conn.execute(
                "SELECT id, run_id, wr, pf, total_pnl, max_drawdown, sharpe, mdd_per_symbol, "
                "yearly_consistency, composite_score, captured_at FROM metrics_snapshots "
                "WHERE run_id = ? ORDER BY captured_at DESC, id DESC LIMIT 1",
                (run_row["id"],),
            ).fetchone()
            audit_rows = conn.execute(
                "SELECT id, entity_type, entity_id, action, actor, reason, payload_json, created_at "
                "FROM audit_log WHERE entity_id IN (?, ?) ORDER BY created_at ASC, id ASC",
                (model_id, run_row["id"]),
            ).fetchall()
        if metrics_row is None:
            return None
        return DashboardBundle(
            model=model,
            run=self._run_from_row(run_row),
            artifacts=[self._artifact_from_row(row) for row in artifacts],
            metrics_snapshot=self._metrics_from_row(metrics_row),
            audit_log=[self._audit_from_row(row) for row in audit_rows],
        )

    def list_active_bundles(self) -> list[DashboardBundle]:
        models = self.list_models(status="active", visible_only=True)
        bundles = [bundle for model in models if (bundle := self.get_bundle(model.id)) is not None]
        return bundles

    def set_model_status(
        self,
        model_id: str,
        status: str,
        *,
        actor: str = "system",
        reason: str | None = None,
        visible_in_dashboard: bool | None = None,
    ) -> ModelRecord:
        model = self.get_model(model_id)
        if model is None:
            raise KeyError(model_id)
        now = _now_iso()
        if visible_in_dashboard is None:
            visible_in_dashboard = status == "active"
        retired_at = None if status == "active" else model.retired_at or now
        retired_reason = None if status == "active" else reason or model.retired_reason
        with self._connect() as conn:
            conn.execute(
                "UPDATE models SET status = ?, visible_in_dashboard = ?, updated_at = ?, "
                "retired_at = ?, retired_reason = ? WHERE id = ?",
                (
                    status,
                    int(bool(visible_in_dashboard)),
                    now,
                    retired_at,
                    retired_reason,
                    model_id,
                ),
            )
            self._write_audit_record(
                conn,
                entity_type="model",
                entity_id=model_id,
                action=status,
                actor=actor,
                reason=reason,
                payload_json={
                    "status": status,
                    "visible_in_dashboard": bool(visible_in_dashboard),
                },
                created_at=now,
            )
        return self.get_model(model_id) or model

    def activate_model(
        self, model_id: str, *, actor: str = "system", reason: str | None = None
    ) -> ModelRecord:
        return self.set_model_status(
            model_id,
            "active",
            actor=actor,
            reason=reason,
            visible_in_dashboard=True,
        )

    def deactivate_model(
        self, model_id: str, *, actor: str = "system", reason: str | None = None
    ) -> ModelRecord:
        return self.set_model_status(
            model_id,
            "inactive",
            actor=actor,
            reason=reason,
            visible_in_dashboard=False,
        )

    def archive_model(
        self, model_id: str, *, actor: str = "system", reason: str | None = None
    ) -> ModelRecord:
        return self.set_model_status(
            model_id,
            "archived",
            actor=actor,
            reason=reason,
            visible_in_dashboard=False,
        )

    def quarantine_model(
        self, model_id: str, *, actor: str = "system", reason: str | None = None
    ) -> ModelRecord:
        return self.set_model_status(
            model_id,
            "quarantined",
            actor=actor,
            reason=reason,
            visible_in_dashboard=False,
        )

    def restore_model(
        self, model_id: str, *, actor: str = "system", reason: str | None = None
    ) -> ModelRecord:
        model = self.get_model(model_id)
        if model is None:
            raise KeyError(model_id)
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE models SET status = ?, visible_in_dashboard = 1, updated_at = ?, "
                "retired_at = NULL, retired_reason = NULL WHERE id = ?",
                ("active", now, model_id),
            )
            self._write_audit_record(
                conn,
                entity_type="model",
                entity_id=model_id,
                action="restore",
                actor=actor,
                reason=reason,
                payload_json={"status": "active"},
                created_at=now,
            )
        return self.get_model(model_id) or model

    def purge_model(
        self, model_id: str, *, actor: str = "system", reason: str | None = None
    ) -> PurgePlan:
        bundle = self.get_bundle(model_id)
        if bundle is None:
            raise KeyError(model_id)
        if bundle.model.status not in {"archived", "quarantined"}:
            raise ValueError("purge requires archived or quarantined model")
        now = _now_iso()
        artifact_paths = [artifact.path for artifact in bundle.artifacts]
        with self._connect() as conn:
            conn.execute(
                "UPDATE models SET status = ?, visible_in_dashboard = 0, updated_at = ?, "
                "retired_at = COALESCE(retired_at, ?), retired_reason = COALESCE(retired_reason, ?) "
                "WHERE id = ?",
                ("purged", now, now, reason, model_id),
            )
            conn.execute(
                "UPDATE runs SET status = ?, completed_at = COALESCE(completed_at, ?) WHERE id = ?",
                ("purged", now, bundle.run.id),
            )
            conn.execute(
                "UPDATE artifacts SET deleted_at = COALESCE(deleted_at, ?) WHERE run_id = ?",
                (now, bundle.run.id),
            )
            self._write_audit_record(
                conn,
                entity_type="model",
                entity_id=model_id,
                action="purge",
                actor=actor,
                reason=reason,
                payload_json={"run_id": bundle.run.id, "artifact_paths": artifact_paths},
                created_at=now,
            )
        return PurgePlan(model_id=model_id, run_id=bundle.run.id, artifact_paths=artifact_paths)

    def export_active_bundles(self) -> list[DashboardBundle]:
        return self.list_active_bundles()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _upsert_model(self, conn: sqlite3.Connection, model: ModelRecord, now: str) -> None:
        conn.execute(
            "INSERT INTO models (id, name, version, status, strategy, feature_set, entry_model, "
            "exit_model, market, priority, visible_in_dashboard, created_at, updated_at, retired_at, "
            "retired_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET name = excluded.name, version = excluded.version, "
            "status = excluded.status, strategy = excluded.strategy, feature_set = excluded.feature_set, "
            "entry_model = excluded.entry_model, exit_model = excluded.exit_model, market = excluded.market, "
            "priority = excluded.priority, visible_in_dashboard = excluded.visible_in_dashboard, "
            "updated_at = excluded.updated_at, retired_at = excluded.retired_at, "
            "retired_reason = excluded.retired_reason",
            (
                model.id,
                model.name,
                model.version,
                model.status,
                model.strategy,
                model.feature_set,
                model.entry_model,
                model.exit_model,
                model.market,
                model.priority,
                int(bool(model.visible_in_dashboard)),
                model.created_at or now,
                model.updated_at or now,
                model.retired_at,
                model.retired_reason,
            ),
        )

    def _upsert_run(self, conn: sqlite3.Connection, run: RunRecord, now: str) -> None:
        conn.execute(
            "INSERT INTO runs (id, model_id, run_name, config_hash, config_path, resolved_config_path, "
            "status, created_at, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET model_id = excluded.model_id, run_name = excluded.run_name, "
            "config_hash = excluded.config_hash, config_path = excluded.config_path, "
            "resolved_config_path = excluded.resolved_config_path, status = excluded.status, "
            "created_at = excluded.created_at, completed_at = excluded.completed_at",
            (
                run.id,
                run.model_id,
                run.run_name,
                run.config_hash,
                run.config_path,
                run.resolved_config_path,
                run.status,
                run.created_at or now,
                run.completed_at,
            ),
        )

    def _upsert_artifacts(
        self, conn: sqlite3.Connection, artifacts: Iterable[ArtifactRecord]
    ) -> None:
        for artifact in artifacts:
            conn.execute(
                "INSERT INTO artifacts (id, run_id, kind, path, size_bytes, checksum, created_at, "
                "deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET run_id = excluded.run_id, kind = excluded.kind, "
                "path = excluded.path, size_bytes = excluded.size_bytes, checksum = excluded.checksum, "
                "created_at = excluded.created_at, deleted_at = excluded.deleted_at",
                (
                    artifact.id,
                    artifact.run_id,
                    artifact.kind,
                    artifact.path,
                    artifact.size_bytes,
                    artifact.checksum,
                    artifact.created_at,
                    artifact.deleted_at,
                ),
            )

    def _upsert_metrics_snapshot(
        self, conn: sqlite3.Connection, metrics_snapshot: MetricsSnapshotRecord
    ) -> None:
        conn.execute(
            "INSERT INTO metrics_snapshots (id, run_id, wr, pf, total_pnl, max_drawdown, sharpe, "
            "mdd_per_symbol, yearly_consistency, composite_score, captured_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET run_id = excluded.run_id, wr = excluded.wr, pf = excluded.pf, "
            "total_pnl = excluded.total_pnl, max_drawdown = excluded.max_drawdown, sharpe = excluded.sharpe, "
            "mdd_per_symbol = excluded.mdd_per_symbol, yearly_consistency = excluded.yearly_consistency, "
            "composite_score = excluded.composite_score, captured_at = excluded.captured_at",
            (
                metrics_snapshot.id,
                metrics_snapshot.run_id,
                metrics_snapshot.wr,
                metrics_snapshot.pf,
                metrics_snapshot.total_pnl,
                metrics_snapshot.max_drawdown,
                metrics_snapshot.sharpe,
                metrics_snapshot.mdd_per_symbol,
                metrics_snapshot.yearly_consistency,
                metrics_snapshot.composite_score,
                metrics_snapshot.captured_at,
            ),
        )

    def _upsert_leaderboard_row(
        self,
        row: LeaderboardRow,
        model_id: str,
        visible_in_dashboard: bool,
        *,
        actor: str,
        reason: str | None,
    ) -> None:
        now = _now_iso()
        row_json = json.dumps(row.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO leaderboard_rows (run_id, model_id, bundle, run_name, market, "
                "market_family, timeframe, generated_at, composite_score, score_mode, "
                "visible_in_dashboard, superseded, row_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET model_id = excluded.model_id, "
                "bundle = excluded.bundle, run_name = excluded.run_name, market = excluded.market, "
                "market_family = excluded.market_family, timeframe = excluded.timeframe, "
                "generated_at = excluded.generated_at, composite_score = excluded.composite_score, "
                "score_mode = excluded.score_mode, visible_in_dashboard = excluded.visible_in_dashboard, "
                "superseded = excluded.superseded, row_json = excluded.row_json, "
                "updated_at = excluded.updated_at",
                (
                    row.run_id,
                    model_id,
                    row.bundle,
                    row.run_name,
                    row.market,
                    row.market_family,
                    row.timeframe,
                    row.generated_at,
                    row.composite_score,
                    row.score_mode,
                    int(bool(visible_in_dashboard)),
                    int(bool(row.superseded)),
                    row_json,
                    now,
                    now,
                ),
            )
            self._write_audit_record(
                conn,
                entity_type="leaderboard_row",
                entity_id=row.run_id,
                action="ingest",
                actor=actor,
                reason=reason,
                payload_json={
                    "market": row.market,
                    "market_family": row.market_family,
                    "timeframe": row.timeframe,
                    "visible_in_dashboard": bool(visible_in_dashboard),
                    "superseded": bool(row.superseded),
                },
                created_at=now,
            )

    def _write_audit_record(
        self,
        conn: sqlite3.Connection,
        *,
        entity_type: str,
        entity_id: str,
        action: str,
        actor: str,
        reason: str | None,
        payload_json: dict[str, Any],
        created_at: str,
    ) -> None:
        conn.execute(
            "INSERT INTO audit_log (id, entity_type, entity_id, action, actor, reason, payload_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _stable_id(
                    "audit", entity_type, entity_id, action, actor, reason or "", created_at
                ),
                entity_type,
                entity_id,
                action,
                actor,
                reason,
                json.dumps(payload_json, ensure_ascii=True, sort_keys=True),
                created_at,
            ),
        )

    def _insert_audit_record(self, conn: sqlite3.Connection, record: AuditLogRecord) -> None:
        conn.execute(
            "INSERT INTO audit_log (id, entity_type, entity_id, action, actor, reason, payload_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET entity_type = excluded.entity_type, entity_id = excluded.entity_id, "
            "action = excluded.action, actor = excluded.actor, reason = excluded.reason, "
            "payload_json = excluded.payload_json, created_at = excluded.created_at",
            (
                record.id,
                record.entity_type,
                record.entity_id,
                record.action,
                record.actor,
                record.reason,
                json.dumps(record.payload_json, ensure_ascii=True, sort_keys=True),
                record.created_at,
            ),
        )

    def _model_from_row(self, row: sqlite3.Row | None) -> ModelRecord | None:
        if row is None:
            return None
        return ModelRecord(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            status=row["status"],
            strategy=row["strategy"],
            feature_set=row["feature_set"],
            entry_model=row["entry_model"],
            exit_model=row["exit_model"],
            market=row["market"],
            priority=int(row["priority"]),
            visible_in_dashboard=bool(row["visible_in_dashboard"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            retired_at=row["retired_at"],
            retired_reason=row["retired_reason"],
        )

    def _run_from_row(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=row["id"],
            model_id=row["model_id"],
            run_name=row["run_name"],
            config_hash=row["config_hash"],
            config_path=row["config_path"],
            resolved_config_path=row["resolved_config_path"],
            status=row["status"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

    def _artifact_from_row(self, row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            id=row["id"],
            run_id=row["run_id"],
            kind=row["kind"],
            path=row["path"],
            size_bytes=int(row["size_bytes"]),
            checksum=row["checksum"],
            created_at=row["created_at"],
            deleted_at=row["deleted_at"],
        )

    def _metrics_from_row(self, row: sqlite3.Row) -> MetricsSnapshotRecord:
        return MetricsSnapshotRecord(
            id=row["id"],
            run_id=row["run_id"],
            wr=float(row["wr"]),
            pf=float(row["pf"]),
            total_pnl=float(row["total_pnl"]),
            max_drawdown=float(row["max_drawdown"]),
            sharpe=float(row["sharpe"]),
            mdd_per_symbol=float(row["mdd_per_symbol"]),
            yearly_consistency=float(row["yearly_consistency"]),
            composite_score=float(row["composite_score"]),
            captured_at=row["captured_at"],
        )

    def _audit_from_row(self, row: sqlite3.Row) -> AuditLogRecord:
        return AuditLogRecord(
            id=row["id"],
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            action=row["action"],
            actor=row["actor"],
            reason=row["reason"],
            payload_json=json.loads(row["payload_json"] or "{}"),
            created_at=row["created_at"],
        )


def _stable_id(*parts: str) -> str:
    import hashlib

    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
