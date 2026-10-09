"""Manifest-driven wrapper around the paper figure renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .scripts.render_figures import render_all


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(raw: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def load_manifest(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read paper_reproduction/manifest.yaml") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported paper manifest: {path}")
    if not isinstance(payload.get("figures"), list) or len(payload["figures"]) != 4:
        raise ValueError("paper manifest must define exactly four public figures")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        data = handle.read()
    # CSV inputs are text contracts.  Git may check them out with LF or CRLF
    # depending on the platform, so hash their canonical LF representation.
    if path.suffix.lower() in {".csv", ".tsv"}:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    digest.update(data)
    return digest.hexdigest()


def verify_manifest(path: Path) -> dict:
    manifest = load_manifest(path)
    data_dir = _resolve(manifest.get("data_dir", "plot_data"), path.parent)
    figures = []
    for spec in manifest["figures"]:
        source = data_dir / spec["source"]
        if not source.is_file():
            raise FileNotFoundError(f"missing aggregate input: {source}")
        import pandas as pd

        frame = pd.read_csv(source)
        missing = sorted(set(spec.get("required_columns", [])) - set(frame.columns))
        if missing:
            raise ValueError(f"{source.name} is missing columns: {missing}")
        expected_hash = spec.get("sha256")
        actual_hash = _sha256(source)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"{source.name} hash drift: {actual_hash} != {expected_hash}")
        checks = spec.get("numeric_checks", {})
        if "min_rows" in checks and len(frame) < int(checks["min_rows"]):
            raise ValueError(f"{source.name} has too few rows")
        if "max_rows" in checks and len(frame) > int(checks["max_rows"]):
            raise ValueError(f"{source.name} has too many rows")
        for column in checks.get("rate_columns", []):
            values = frame[column].astype(float)
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"{source.name} contains an out-of-range rate in {column}")
        figures.append(
            {
                "id": spec["id"],
                "source": str(source.relative_to(PROJECT_ROOT)),
                "sha256": actual_hash,
                "rows": int(len(frame)),
                "columns": list(frame.columns),
            }
        )
    return {"manifest": str(path), "schema_version": 1, "figures": figures}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild the four public paper figures")
    parser.add_argument("--manifest", default="paper_reproduction/manifest.yaml")
    parser.add_argument("--only", default="", help="Comma-separated figure IDs")
    parser.add_argument("--output-dir", default="results/paper_figures")
    parser.add_argument("--verify", action="store_true", help="Verify aggregate inputs without rendering")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_path = _resolve(args.manifest)
        manifest = load_manifest(manifest_path)
        report = verify_manifest(manifest_path)
        if args.verify:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        selected_ids = {item.strip() for item in args.only.split(",") if item.strip()}
        known_ids = {spec["id"] for spec in manifest["figures"]}
        unknown = selected_ids - known_ids
        if unknown:
            raise ValueError(f"unknown figure IDs: {sorted(unknown)}")
        renderer_names = {spec["id"]: spec["renderer"] for spec in manifest["figures"]}
        selected_renderers = {renderer_names[item] for item in selected_ids}
        data_dir = _resolve(manifest.get("data_dir", "plot_data"), manifest_path.parent)
        output_dir = _resolve(args.output_dir)
        paths = render_all(data_dir, output_dir, selected_renderers)
        print(json.dumps({"verified": report, "outputs": [str(path) for path in paths]}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
