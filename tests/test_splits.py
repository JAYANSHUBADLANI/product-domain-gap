from studio_shift.extract import InventoryEntry
from studio_shift.splits import (
    build_all_splits,
    build_product_split,
    build_real_life_split,
    split_report,
)


def _make_entries(domain: str, category: str, n: int, dup_pairs: int = 0) -> list[InventoryEntry]:
    """n distinct-hash entries, plus dup_pairs additional entries that
    each reuse the hash of an earlier one, to exercise the atomic
    duplicate-group placement."""
    entries = [
        InventoryEntry(domain=domain, category=category, archive_member=f"m{i}",
                        filename=f"{i}.jpg", size_bytes=100, sha256=f"hash{i}")
        for i in range(n)
    ]
    for i in range(dup_pairs):
        entries.append(InventoryEntry(domain=domain, category=category, archive_member=f"dup{i}",
                                       filename=f"dup{i}.jpg", size_bytes=100, sha256=f"hash{i}"))
    return entries


def test_product_split_sizes_with_no_duplicates():
    entries = _make_entries("product_images", "widget", 100)
    result = build_product_split(entries, "widget", n_train=70, n_val=10, n_test=20, root_seed=1)
    assert len(result["train"]) == 70
    assert len(result["val"]) == 10
    assert len(result["test"]) == 20


def test_product_split_is_a_partition_with_no_overlap():
    entries = _make_entries("product_images", "widget", 100)
    result = build_product_split(entries, "widget", n_train=70, n_val=10, n_test=20, root_seed=1)
    train_files = {e.filename for e in result["train"]}
    val_files = {e.filename for e in result["val"]}
    test_files = {e.filename for e in result["test"]}
    assert train_files.isdisjoint(val_files)
    assert train_files.isdisjoint(test_files)
    assert val_files.isdisjoint(test_files)
    assert len(train_files | val_files | test_files) == 100


def test_product_split_is_deterministic_across_reruns():
    entries = _make_entries("product_images", "widget", 100)
    a = build_product_split(entries, "widget", n_train=70, n_val=10, n_test=20, root_seed=42)
    b = build_product_split(entries, "widget", n_train=70, n_val=10, n_test=20, root_seed=42)
    assert [e.filename for e in a["train"]] == [e.filename for e in b["train"]]
    assert [e.filename for e in a["val"]] == [e.filename for e in b["val"]]
    assert [e.filename for e in a["test"]] == [e.filename for e in b["test"]]


def test_product_split_differs_with_a_different_seed():
    entries = _make_entries("product_images", "widget", 100)
    a = build_product_split(entries, "widget", n_train=70, n_val=10, n_test=20, root_seed=1)
    b = build_product_split(entries, "widget", n_train=70, n_val=10, n_test=20, root_seed=2)
    assert [e.filename for e in a["train"]] != [e.filename for e in b["train"]]


def test_duplicate_group_never_splits_across_train_and_test():
    # 98 unique images plus one duplicate pair (hash97 shared by two
    # files): the real bug this guards against is a naive per-file
    # shuffle placing one half of an exact duplicate in train and the
    # other in test, leaking the test image into training.
    entries = _make_entries("product_images", "widget", 98, dup_pairs=1)
    result = build_product_split(entries, "widget", n_train=70, n_val=10, n_test=20, root_seed=1)

    hash_to_split = {}
    for split_name, items in result.items():
        for e in items:
            hash_to_split.setdefault(e.sha256, set()).add(split_name)

    for h, splits_seen in hash_to_split.items():
        assert len(splits_seen) == 1, f"hash {h} appeared in multiple splits: {splits_seen}"


def test_real_life_split_sizes_and_disjoint():
    entries = _make_entries("real_life", "widget", 100)
    result = build_real_life_split(entries, "widget", n_pool=50, n_shifted_test=50, root_seed=1)
    assert len(result["adaptation_pool"]) == 50
    assert len(result["shifted_test"]) == 50
    pool_files = {e.filename for e in result["adaptation_pool"]}
    test_files = {e.filename for e in result["shifted_test"]}
    assert pool_files.isdisjoint(test_files)


def test_build_all_splits_covers_every_category():
    protocol = {
        "project": {"root_seed": 7},
        "categories": ["widget", "gadget"],
        "splits": {
            "product_images": {"product_train": 70, "product_val": 10, "product_test": 20},
            "real_life": {"adaptation_pool": 50, "shifted_test": 50},
        },
    }
    entries = (
        _make_entries("product_images", "widget", 100)
        + _make_entries("product_images", "gadget", 100)
        + _make_entries("real_life", "widget", 100)
        + _make_entries("real_life", "gadget", 100)
    )
    result = build_all_splits(entries, protocol)
    assert set(result["product_images"]) == {"widget", "gadget"}
    assert set(result["real_life"]) == {"widget", "gadget"}


def test_split_report_flags_target_vs_actual():
    protocol = {
        "categories": ["widget"],
        "splits": {
            "product_images": {"product_train": 70, "product_val": 10, "product_test": 20},
            "real_life": {"adaptation_pool": 50, "shifted_test": 50},
        },
    }
    splits = {
        "product_images": {"widget": {
            "train": _make_entries("product_images", "widget", 70),
            "val": _make_entries("product_images", "widget", 10),
            "test": _make_entries("product_images", "widget", 20),
        }},
        "real_life": {"widget": {
            "adaptation_pool": _make_entries("real_life", "widget", 50),
            "shifted_test": _make_entries("real_life", "widget", 50),
        }},
    }
    [row] = split_report(splits, protocol)
    assert row["product_train"] == row["product_train_target"] == 70
    assert row["shifted_test"] == row["shifted_test_target"] == 50
