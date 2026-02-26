from __future__ import annotations

import pytest


def test_fixed_temperature_schedule_has_six_equal_values():
    from skedl.pipeline.sampling import build_temperature_schedule

    schedule = build_temperature_schedule(
        num_cots=6,
        mode="fixed",
        temperature=0.7,
        delta=0.2,
    )

    assert schedule == [0.7] * 6


def test_mixed_temperature_schedule_uses_offsets_and_clamps():
    from skedl.pipeline.sampling import build_temperature_schedule

    schedule = build_temperature_schedule(
        num_cots=6,
        mode="mixed",
        temperature=0.1,
        delta=0.2,
        min_temperature=0.0,
        max_temperature=1.0,
    )

    assert schedule == [0.1, 0.1, 0.1, 0.0, 0.3, 0.5]


def test_non_default_num_cots_is_supported_with_fixed_mode():
    from skedl.pipeline.sampling import build_temperature_schedule

    schedule = build_temperature_schedule(
        num_cots=4,
        mode="fixed",
        temperature=0.6,
        delta=0.1,
    )

    assert schedule == [0.6, 0.6, 0.6, 0.6]


def test_mixed_mode_requires_six_cots_in_v1():
    from skedl.pipeline.sampling import build_temperature_schedule

    with pytest.raises(ValueError, match="num_cots=6"):
        build_temperature_schedule(
            num_cots=5,
            mode="mixed",
            temperature=0.7,
            delta=0.2,
        )
