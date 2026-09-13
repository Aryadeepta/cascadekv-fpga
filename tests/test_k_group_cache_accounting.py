"""Frozen architectural storage accounting for the K-group routing cache."""


def test_k_group_cache_logical_storage_accounting() -> None:
    groups_per_key = 8
    k_code_bytes_per_group = 8
    scale_bytes_per_group = 2
    bytes_per_group = k_code_bytes_per_group + scale_bytes_per_group
    bytes_per_key = groups_per_key * bytes_per_group
    candidate_count = 256

    assert bytes_per_group == 10
    assert bytes_per_key == 80
    assert candidate_count * bytes_per_key == 20_480
