"""Selective extraction of the locked 20-category subset from the local
Adaptiope archive, plus the inventory (per-file SHA256, size) that the
duplicate policy and split generator both depend on.

Written to discover the archive's actual internal layout at runtime
(list every member under each domain's stated prefix) rather than assume
a file naming convention, since the archive was never opened locally
before this ran.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "protocol.yaml"


def load_protocol() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class InventoryEntry:
    domain: str
    category: str
    archive_member: str
    filename: str
    size_bytes: int
    sha256: str


def _iter_domain_members(zf: zipfile.ZipFile, prefix: str, categories: list[str]) -> list[str]:
    """Every archive member under prefix/<category>/ for a category in
    the locked list, for any category directory naming (case and
    whitespace can vary between an archive's own naming and a protocol
    file's), matched by exact category name as the immediate subfolder.
    """
    wanted = set(categories)
    members = []
    for name in zf.namelist():
        if not name.startswith(prefix + "/") or name.endswith("/"):
            continue
        rest = name[len(prefix) + 1:]
        parts = rest.split("/")
        if len(parts) != 2:
            continue
        category, filename = parts
        if category in wanted:
            members.append(name)
    return members


def extract_and_inventory(
    zip_path: Path, output_root: Path, protocol: dict | None = None,
) -> list[InventoryEntry]:
    protocol = protocol or load_protocol()
    categories = protocol["categories"]
    domains = protocol["dataset"]["domains"]

    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[InventoryEntry] = []

    with zipfile.ZipFile(zip_path) as zf:
        for domain_name, domain_cfg in domains.items():
            prefix = domain_cfg["archive_prefix"]
            members = _iter_domain_members(zf, prefix, categories)
            if not members:
                raise RuntimeError(
                    f"no archive members found under prefix {prefix!r} for domain "
                    f"{domain_name!r} - the protocol's archive_prefix does not match "
                    f"this archive's actual layout, inspect zf.namelist() directly"
                )
            for member in members:
                category, filename = member[len(prefix) + 1:].split("/")
                dest_dir = output_root / domain_name / category
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / filename

                data = zf.read(member)
                dest_path.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                entries.append(InventoryEntry(
                    domain=domain_name, category=category, archive_member=member,
                    filename=filename, size_bytes=len(data), sha256=digest,
                ))

    return entries


def write_inventory(entries: list[InventoryEntry], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(entries[0]).keys()) if entries else [])
        writer.writeheader()
        for e in entries:
            writer.writerow(asdict(e))


def verify_counts(entries: list[InventoryEntry], protocol: dict) -> dict:
    """Per (domain, category) image count against the protocol's expected
    count. Returns the summary rather than raising, since a real archive
    can legitimately deviate slightly from a paper's stated per-category
    count and that needs to be seen and decided on, not hidden."""
    from collections import Counter

    counts = Counter((e.domain, e.category) for e in entries)
    summary = {"ok": [], "mismatched": []}
    for domain_name, domain_cfg in protocol["dataset"]["domains"].items():
        expected = domain_cfg["expected_images_per_category"]
        for category in protocol["categories"]:
            actual = counts[(domain_name, category)]
            row = {"domain": domain_name, "category": category, "expected": expected, "actual": actual}
            (summary["ok"] if actual == expected else summary["mismatched"]).append(row)
    return summary


def find_duplicate_groups(entries: list[InventoryEntry]) -> dict[str, list[InventoryEntry]]:
    """Groups of entries (any domain/category) sharing an exact sha256.

    The locked duplicate policy is atomic_exact_sha256: a duplicate group
    must be kept together in whichever split it lands in, never split
    across train/val/test or across adaptation_pool/shifted_test, since
    splitting an exact duplicate would let the same image sit in both a
    training set and the test set that is supposed to be held out from it.
    """
    by_hash: dict[str, list[InventoryEntry]] = {}
    for e in entries:
        by_hash.setdefault(e.sha256, []).append(e)
    return {h: es for h, es in by_hash.items() if len(es) > 1}
