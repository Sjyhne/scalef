from argparse import Namespace

from optimize import resolve_hash_resolutions


def _args(**kwargs):
    base = dict(
        lr_height=68,
        lr_width=209,
        lr_size=0,
        hash_max_resolution=0,
        hash_max_resolution_h=0,
        hash_max_resolution_w=0,
        hash_max_resolution_mult=1.0,
        hash_base_resolution=0,
    )
    base.update(kwargs)
    return Namespace(**base)


def test_default_max_resolution_tracks_lr_grid():
    base_h, max_h, base_w, max_w = resolve_hash_resolutions(_args())
    assert (max_h, max_w) == (68, 209)
    assert (base_h, base_w) == (17, 52)


def test_multiplier_scales_max_resolution_to_hr_grid():
    base_h, max_h, base_w, max_w = resolve_hash_resolutions(
        _args(hash_max_resolution_mult=4.0)
    )
    assert (max_h, max_w) == (272, 836)
    # Coarsest level lands on the LR grid, finest on HR.
    assert (base_h, base_w) == (68, 209)


def test_multiplier_of_one_is_a_noop():
    assert resolve_hash_resolutions(_args(hash_max_resolution_mult=1.0)) == (
        resolve_hash_resolutions(_args())
    )


def test_explicit_base_resolution_overrides_auto():
    base_h, _, base_w, _ = resolve_hash_resolutions(_args(hash_base_resolution=12))
    assert (base_h, base_w) == (12, 12)


def test_explicit_per_axis_max_still_scaled_by_multiplier():
    _, max_h, _, max_w = resolve_hash_resolutions(
        _args(hash_max_resolution_h=100, hash_max_resolution_w=200, hash_max_resolution_mult=2.0)
    )
    assert (max_h, max_w) == (200, 400)
