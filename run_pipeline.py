"""Single entry point: extraction through the recovery curve, end to
end. Each phase checks for the previous phase's output before running,
so a resumed or partial run does not redo finished work, and a phase can
be run alone with the module it lives in for debugging one step.

Usage: .venv/bin/python run_pipeline.py [phase]
  phase omitted or "all": run every phase in order
  phase one of: extract, split, train, eval, recovery, saliency
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from studio_shift import seed_everything  # noqa: E402
from studio_shift.extract import (  # noqa: E402
    extract_and_inventory,
    find_duplicate_groups,
    load_protocol,
    verify_counts,
    write_inventory,
)
from studio_shift.splits import build_all_splits, split_report, write_manifests  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "Adaptiope.zip"
DATA_PROCESSED = ROOT / "data" / "processed" / "Adaptiope"
ARTIFACTS = ROOT / "artifacts"
MANIFESTS_DIR = ARTIFACTS / "manifests"
CHECKPOINT_PATH = ARTIFACTS / "checkpoints" / "clean_classifier.pt"


def phase_extract(protocol: dict) -> None:
    inventory_json = ROOT / protocol["artifacts"]["inventory_json"]
    inventory_csv = ROOT / protocol["artifacts"]["inventory_table"]

    if inventory_json.exists():
        print(f"[extract] {inventory_json} already exists, skipping")
        return

    print(f"[extract] reading {DATA_RAW}")
    t0 = time.time()
    entries = extract_and_inventory(DATA_RAW, DATA_PROCESSED, protocol)
    print(f"[extract] extracted {len(entries)} files in {time.time() - t0:.1f}s")

    summary = verify_counts(entries, protocol)
    if summary["mismatched"]:
        print(f"[extract] WARNING: {len(summary['mismatched'])} (domain, category) pairs "
              f"do not match the expected per-category count:")
        for row in summary["mismatched"]:
            print(f"  {row}")
    else:
        print("[extract] every (domain, category) matches the expected count")

    dup_groups = find_duplicate_groups(entries)
    print(f"[extract] {len(dup_groups)} exact-duplicate groups found across the extracted set")

    write_inventory(entries, inventory_json, inventory_csv)
    with open(ARTIFACTS / "duplicate_groups.json", "w") as f:
        json.dump({h: [e.archive_member for e in es] for h, es in dup_groups.items()}, f, indent=2)
    with open(ARTIFACTS / "count_verification.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[extract] wrote {inventory_json}")


def phase_split(protocol: dict) -> None:
    manifest_files = ["product_train", "product_val", "product_test", "adaptation_pool", "shifted_test"]
    if all((MANIFESTS_DIR / f"{name}.json").exists() for name in manifest_files):
        print("[split] all manifests already exist, skipping")
        return

    with open(ARTIFACTS / "inventory.json") as f:
        raw_entries = json.load(f)
    from studio_shift.extract import InventoryEntry
    entries = [InventoryEntry(**e) for e in raw_entries]

    splits = build_all_splits(entries, protocol)
    report = split_report(splits, protocol)
    with open(ARTIFACTS / "tables" / "split_report.json", "w") as f:
        json.dump(report, f, indent=2)

    mismatches = [r for r in report if any(
        r[f"{k}"] != r[f"{k}_target"]
        for k in ["product_train", "product_val", "product_test", "adaptation_pool", "shifted_test"]
    )]
    if mismatches:
        print(f"[split] {len(mismatches)} categories deviate from target split sizes (see split_report.json)")
    else:
        print("[split] every category matches its target split sizes exactly")

    write_manifests(splits, DATA_PROCESSED, MANIFESTS_DIR)
    print(f"[split] wrote manifests to {MANIFESTS_DIR}")


def phase_train(protocol: dict) -> None:
    if CHECKPOINT_PATH.exists():
        print(f"[train] {CHECKPOINT_PATH} already exists, skipping")
        return

    from studio_shift.train import train_clean_classifier

    print("[train] training classifier head on product_train, selecting on product_val")
    lock = train_clean_classifier(MANIFESTS_DIR, CHECKPOINT_PATH, protocol)
    print(f"[train] done: val_acc={lock['best_val_acc']:.3f} "
          f"epochs={lock['epochs_run']} time={lock['training_seconds']:.1f}s device={lock['device']}")
    print(f"[train] checkpoint sha256={lock['sha256']}")


def phase_eval(protocol: dict) -> None:
    from studio_shift.evaluate import (
        bootstrap_class_balanced_ci,
        class_balanced_accuracy,
        confidence_gap_analysis,
        per_category_accuracy,
        run_eval,
    )

    categories = protocol["categories"]
    model_cfg = protocol["model"]
    root_seed = protocol["project"]["root_seed"]

    out_path = ARTIFACTS / "headline_result.json"
    if out_path.exists():
        print(f"[eval] {out_path} already exists, skipping")
        return

    print("[eval] scoring clean (product_test)")
    clean_preds = run_eval(MANIFESTS_DIR / "product_test.json", CHECKPOINT_PATH, categories,
                            model_cfg["backbone"], model_cfg["revision"])
    print("[eval] scoring shifted (shifted_test)")
    shifted_preds = run_eval(MANIFESTS_DIR / "shifted_test.json", CHECKPOINT_PATH, categories,
                              model_cfg["backbone"], model_cfg["revision"])

    clean_acc = class_balanced_accuracy(clean_preds)
    shifted_acc = class_balanced_accuracy(shifted_preds)
    shifted_ci = bootstrap_class_balanced_ci(shifted_preds, n_boot=2000, root_seed=root_seed,
                                              namespace="shifted_test_bootstrap")
    clean_ci = bootstrap_class_balanced_ci(clean_preds, n_boot=2000, root_seed=root_seed,
                                            namespace="clean_test_bootstrap")

    result = {
        "clean_accuracy": clean_acc,
        "clean_accuracy_ci": clean_ci,
        "shifted_accuracy": shifted_acc,
        "shifted_accuracy_ci": shifted_ci,
        "gap": clean_acc - shifted_acc,
        "clean_per_category": per_category_accuracy(clean_preds),
        "shifted_per_category": per_category_accuracy(shifted_preds),
        "shifted_confidence_gap": confidence_gap_analysis(shifted_preds),
        "clean_confidence_gap": confidence_gap_analysis(clean_preds),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[eval] clean={clean_acc:.3f} shifted={shifted_acc:.3f} gap={result['gap']:.3f}")
    print(f"[eval] shifted 95% CI: [{shifted_ci['ci_low']:.3f}, {shifted_ci['ci_high']:.3f}]")


def phase_recovery(protocol: dict) -> None:
    from studio_shift.recovery import run_recovery_curve, summarize_recovery_curve

    out_path = ARTIFACTS / "recovery_curve.json"
    progress_path = ARTIFACTS / "recovery_progress.jsonl"
    if out_path.exists():
        print(f"[recovery] {out_path} already exists, skipping")
        return

    print("[recovery] running the labelled-recovery sweep, this is the slow phase")
    print(f"[recovery] each completed cycle is saved to {progress_path} immediately, "
          f"so killing and rerunning this phase resumes rather than restarts")
    t0 = time.time()
    results = run_recovery_curve(MANIFESTS_DIR, CHECKPOINT_PATH, protocol, progress_path=progress_path)
    summary = summarize_recovery_curve(results)

    with open(out_path, "w") as f:
        json.dump({"raw": results, "summary": summary}, f, indent=2)

    print(f"[recovery] done in {time.time() - t0:.1f}s")
    for row in summary:
        print(f"  labels_per_class={row['labels_per_class']:3d}  "
              f"mean_acc={row['mean_accuracy']:.3f}  range=[{row['min_accuracy']:.3f}, {row['max_accuracy']:.3f}]")


def phase_saliency(protocol: dict) -> None:
    import torch

    from studio_shift.dataset import ManifestImageDataset
    from studio_shift.evaluate import load_locked_model, predict_all
    from studio_shift.model import load_backbone_and_processor, make_transform, pick_device
    from studio_shift.saliency import GradCAM, save_overlay
    from PIL import Image

    figures_dir = ARTIFACTS / "figures" / "saliency"
    if figures_dir.exists() and any(figures_dir.iterdir()):
        print(f"[saliency] {figures_dir} already has output, skipping")
        return

    categories = protocol["categories"]
    model_cfg = protocol["model"]
    device = pick_device()

    _, processor = load_backbone_and_processor(model_cfg["backbone"], model_cfg["revision"])
    transform = make_transform(processor)
    shifted_ds = ManifestImageDataset(MANIFESTS_DIR / "shifted_test.json", categories, transform)

    from torch.utils.data import DataLoader
    loader = DataLoader(shifted_ds, batch_size=16, shuffle=False)
    model = load_locked_model(CHECKPOINT_PATH, model_cfg["backbone"], model_cfg["revision"], categories, device)
    predictions = predict_all(model, loader, categories, device)

    # One miss per true category (up to 12 categories), not just the
    # first 12 misses overall: the shifted_test manifest is grouped by
    # category, so taking the first 12 misses unfiltered landed entirely
    # within the first category alphabetically and said nothing about
    # any other category's failures.
    miss_indices = []
    seen_categories = set()
    for i, p in enumerate(predictions):
        if p.correct or p.true_category in seen_categories:
            continue
        miss_indices.append(i)
        seen_categories.add(p.true_category)
        if len(miss_indices) >= 12:
            break
    print(f"[saliency] generating Grad-CAM for {len(miss_indices)} shifted-domain misses "
          f"across {len(seen_categories)} categories")

    cam = GradCAM(model)
    for idx in miss_indices:
        row = shifted_ds.rows[idx]
        row_path = Path(row["path"])
        original = Image.open(row_path if row_path.is_absolute() else ROOT / row_path).convert("RGB")
        pixel_values = transform(original).unsqueeze(0).to(device)
        predicted_index = categories.index(predictions[idx].predicted_category)
        cam_map, target_class = cam.compute(pixel_values, target_class=predicted_index)
        out_path = figures_dir / f"{Path(row['path']).stem}_true-{row['category']}_pred-{predictions[idx].predicted_category}.png"
        save_overlay(original, cam_map, out_path)

    print(f"[saliency] wrote {len(miss_indices)} overlays to {figures_dir}")


PHASES = {
    "extract": phase_extract,
    "split": phase_split,
    "train": phase_train,
    "eval": phase_eval,
    "recovery": phase_recovery,
    "saliency": phase_saliency,
}


def main() -> None:
    seed_everything()
    protocol = load_protocol()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "tables").mkdir(parents=True, exist_ok=True)

    requested = sys.argv[1] if len(sys.argv) > 1 else "all"
    if requested == "all":
        order = ["extract", "split", "train", "eval", "recovery", "saliency"]
    elif requested in PHASES:
        order = [requested]
    else:
        print(f"unknown phase {requested!r}, choose from: all, {', '.join(PHASES)}")
        sys.exit(1)

    for phase_name in order:
        print(f"\n=== phase: {phase_name} ===")
        PHASES[phase_name](protocol)


if __name__ == "__main__":
    main()
