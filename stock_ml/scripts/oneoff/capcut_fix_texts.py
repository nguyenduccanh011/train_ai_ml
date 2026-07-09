from __future__ import annotations

import argparse
import json
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_CAPCUT_FONT = (
    "C:/Users/DUC CANH PC/AppData/Local/CapCut/Apps/8.5.0.3590/Resources/Font/SystemFont/en.ttf"
)


def _to_win_path(path_str: str) -> Path:
    return Path(path_str.replace("/", "\\"))


def _path_exists(path_str: str) -> bool:
    return bool(path_str) and _to_win_path(path_str).exists()


def _parse_content(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _serialize_content(content_obj: dict[str, Any]) -> str:
    return json.dumps(content_obj, ensure_ascii=False, separators=(",", ":"))


def _pick_fallback_font(material_texts: list[dict[str, Any]]) -> str:
    existing = Counter()
    for text in material_texts:
        path = text.get("font_path")
        if isinstance(path, str) and _path_exists(path):
            existing[path] += 1
    if existing:
        return existing.most_common(1)[0][0]
    if _path_exists(DEFAULT_CAPCUT_FONT):
        return DEFAULT_CAPCUT_FONT
    return DEFAULT_CAPCUT_FONT


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _create_backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.before_text_fix_{ts}.bak")
    backup_path.write_bytes(path.read_bytes())
    return backup_path


def _collect_text_segment_pairs(
    data: dict[str, Any],
) -> list[tuple[int, dict[str, Any], str]]:
    texts = data.get("materials", {}).get("texts", [])
    text_ids = {t.get("id") for t in texts if isinstance(t, dict)}
    pairs: list[tuple[int, dict[str, Any], str]] = []
    for track_idx, track in enumerate(data.get("tracks", [])):
        if not isinstance(track, dict):
            continue
        for seg in track.get("segments") or []:
            if not isinstance(seg, dict):
                continue
            material_id = seg.get("material_id")
            if isinstance(material_id, str) and material_id in text_ids:
                pairs.append((track_idx, seg, material_id))
    return pairs


def repair_capcut_texts(
    draft_content_path: Path,
    *,
    group_by_track: bool = True,
    normalize_fixed_box: bool = True,
) -> dict[str, int]:
    data = _load_json(draft_content_path)
    materials = data.get("materials", {})
    material_texts = materials.get("texts", [])
    if not isinstance(material_texts, list):
        raise ValueError("materials.texts is not a list")

    fallback_font = _pick_fallback_font(material_texts)
    text_map = {
        t.get("id"): t
        for t in material_texts
        if isinstance(t, dict) and isinstance(t.get("id"), str)
    }
    segment_pairs = _collect_text_segment_pairs(data)

    style_font_patched = 0
    style_size_patched = 0
    material_font_patched = 0
    fixed_box_normalized = 0
    grouped_segments = 0
    grouped_texts = 0

    # Patch material text style/font/size first.
    for text in material_texts:
        if not isinstance(text, dict):
            continue
        content_obj = _parse_content(text.get("content"))
        font_size = float(text.get("font_size", 15.0))

        if text.get("font_path") != fallback_font:
            text["font_path"] = fallback_font
            material_font_patched += 1

        if normalize_fixed_box:
            fw = text.get("fixed_width")
            fh = text.get("fixed_height")
            if isinstance(fw, (int, float)) and fw > 0:
                text["fixed_width"] = -1.0
                fixed_box_normalized += 1
            if isinstance(fh, (int, float)) and fh > 0:
                text["fixed_height"] = -1.0
                fixed_box_normalized += 1

        if content_obj and isinstance(content_obj.get("styles"), list):
            styles = content_obj.get("styles") or []
            for style in styles:
                if not isinstance(style, dict):
                    continue
                font_obj = style.get("font")
                if not isinstance(font_obj, dict):
                    font_obj = {}
                    style["font"] = font_obj
                style_path = font_obj.get("path")
                if not isinstance(style_path, str) or not _path_exists(style_path):
                    font_obj["path"] = fallback_font
                    style_font_patched += 1
                size_val = style.get("size")
                if size_val is None or abs(float(size_val) - font_size) > 1e-6:
                    style["size"] = font_size
                    style_size_patched += 1
            text["content"] = _serialize_content(content_obj)

    # Optional grouping by track for text segments/materials.
    if group_by_track:
        track_group_ids: dict[int, str] = {}
        seen_text_ids: set[str] = set()
        for track_idx, seg, material_id in segment_pairs:
            gid = track_group_ids.get(track_idx)
            if gid is None:
                gid = str(uuid.uuid4()).upper()
                track_group_ids[track_idx] = gid
            if seg.get("group_id") != gid:
                seg["group_id"] = gid
                grouped_segments += 1
            text = text_map.get(material_id)
            if text is not None and material_id not in seen_text_ids:
                if text.get("group_id") != gid:
                    text["group_id"] = gid
                    grouped_texts += 1
                seen_text_ids.add(material_id)

    _save_json(draft_content_path, data)
    return {
        "texts_total": len(material_texts),
        "segments_total": len(segment_pairs),
        "style_font_patched": style_font_patched,
        "style_size_patched": style_size_patched,
        "material_font_patched": material_font_patched,
        "fixed_box_normalized": fixed_box_normalized,
        "grouped_segments": grouped_segments,
        "grouped_texts": grouped_texts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix CapCut draft text materials (font/size/fixed-box) and optional group_id by track."
    )
    parser.add_argument(
        "--draft-content",
        required=True,
        help="Path to draft_content.json",
    )
    parser.add_argument(
        "--no-group-by-track",
        action="store_true",
        help="Do not assign shared group_id for text items in the same track.",
    )
    parser.add_argument(
        "--no-normalize-fixed-box",
        action="store_true",
        help="Do not reset fixed_width/fixed_height to -1.0",
    )
    args = parser.parse_args()

    draft_content_path = Path(args.draft_content)
    if not draft_content_path.exists():
        raise FileNotFoundError(f"draft_content not found: {draft_content_path}")

    backup_path = _create_backup(draft_content_path)
    stats = repair_capcut_texts(
        draft_content_path,
        group_by_track=not args.no_group_by_track,
        normalize_fixed_box=not args.no_normalize_fixed_box,
    )

    print(f"Backup: {backup_path}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
