import zipfile

import pytest

from studio_shift.extract import (
    extract_and_inventory,
    find_duplicate_groups,
    verify_counts,
)


@pytest.fixture
def tiny_archive(tmp_path):
    """A miniature archive with the same layout Adaptiope is assumed to
    have: <prefix>/<category>/<file>.jpg for two domains, two categories,
    including one exact duplicate and one file under a category NOT in
    the locked list, to check both are handled correctly.
    """
    zip_path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Adaptiope/product_images/widget/1.jpg", b"widget-image-a")
        zf.writestr("Adaptiope/product_images/widget/2.jpg", b"widget-image-b")
        zf.writestr("Adaptiope/product_images/gadget/1.jpg", b"gadget-image-a")
        zf.writestr("Adaptiope/product_images/not_a_locked_category/1.jpg", b"should-be-ignored")
        zf.writestr("Adaptiope/real_life/widget/1.jpg", b"widget-image-a")  # exact duplicate of product one
        zf.writestr("Adaptiope/real_life/widget/2.jpg", b"real-widget-b")
        zf.writestr("Adaptiope/real_life/gadget/1.jpg", b"real-gadget-a")
    return zip_path


@pytest.fixture
def protocol():
    return {
        "categories": ["widget", "gadget"],
        "dataset": {
            "domains": {
                "product_images": {"archive_prefix": "Adaptiope/product_images", "expected_images_per_category": 2},
                "real_life": {"archive_prefix": "Adaptiope/real_life", "expected_images_per_category": 2},
            }
        },
    }


def test_extract_only_pulls_locked_categories(tiny_archive, protocol, tmp_path):
    output_root = tmp_path / "extracted"
    entries = extract_and_inventory(tiny_archive, output_root, protocol)
    categories_seen = {e.category for e in entries}
    assert categories_seen == {"widget", "gadget"}
    assert "not_a_locked_category" not in categories_seen


def test_extract_writes_files_to_domain_category_structure(tiny_archive, protocol, tmp_path):
    output_root = tmp_path / "extracted"
    extract_and_inventory(tiny_archive, output_root, protocol)
    assert (output_root / "product_images" / "widget" / "1.jpg").read_bytes() == b"widget-image-a"
    assert (output_root / "real_life" / "gadget" / "1.jpg").read_bytes() == b"real-gadget-a"


def test_extract_computes_correct_sha256(tiny_archive, protocol, tmp_path):
    import hashlib

    output_root = tmp_path / "extracted"
    entries = extract_and_inventory(tiny_archive, output_root, protocol)
    widget_1 = next(e for e in entries if e.domain == "product_images" and e.filename == "1.jpg" and e.category == "widget")
    assert widget_1.sha256 == hashlib.sha256(b"widget-image-a").hexdigest()


def test_extract_raises_if_prefix_matches_nothing(tiny_archive, tmp_path):
    protocol = {
        "categories": ["widget"],
        "dataset": {"domains": {
            "product_images": {"archive_prefix": "wrong/prefix", "expected_images_per_category": 2},
        }},
    }
    with pytest.raises(RuntimeError, match="no archive members found"):
        extract_and_inventory(tiny_archive, tmp_path / "out", protocol)


def test_verify_counts_flags_a_category_below_expected(tiny_archive, protocol, tmp_path):
    entries = extract_and_inventory(tiny_archive, tmp_path / "out", protocol)
    summary = verify_counts(entries, protocol)
    # gadget only has 1 image per domain here, expected is 2
    mismatched_categories = {row["category"] for row in summary["mismatched"]}
    assert "gadget" in mismatched_categories
    ok_categories_at_full_count = {row["category"] for row in summary["ok"]}
    assert "widget" in ok_categories_at_full_count


def test_find_duplicate_groups_finds_the_cross_domain_duplicate(tiny_archive, protocol, tmp_path):
    entries = extract_and_inventory(tiny_archive, tmp_path / "out", protocol)
    dup_groups = find_duplicate_groups(entries)
    assert len(dup_groups) == 1
    [group] = dup_groups.values()
    domains_in_group = {e.domain for e in group}
    assert domains_in_group == {"product_images", "real_life"}
