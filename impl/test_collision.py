"""Checks for the collision geometry. Run: python test_collision.py"""

from collision import Box, CollisionConfig, Forecast, HitLatch, centre_in_margin_frame, fit_line, forecast_from_window


def test_ols_known_line():
    times = [100.0, 105.0, 110.0, 115.0]
    values = [10.0, 20.0, 30.0, 40.0]  # slope 2, value(t) = 2*(t-100)+10 = 2t - 190
    line = fit_line(times, values)
    assert abs(line.slope - 2.0) < 1e-6, line
    assert abs(line.at(100) - 10) < 1e-6
    assert abs(line.at(115) - 40) < 1e-6


def test_margin_axes_are_independent():
    # 854x480. Horizontal 1.2 → extra 0.1 W each side. Vertical 1.5 → extra 0.25 H.
    assert centre_in_margin_frame(0, 0, 854, 480, 1.2, 1.5)
    assert centre_in_margin_frame(854, 480, 854, 480, 1.2, 1.5)
    assert centre_in_margin_frame(-0.09 * 854, 240, 854, 480, 1.2, 1.5)
    assert not centre_in_margin_frame(-0.15 * 854, 240, 854, 480, 1.2, 1.5)
    # Horizontal 0.8 shrinks the width: centre must sit in the middle 80%.
    assert centre_in_margin_frame(0.15 * 854, 240, 854, 480, 0.8, 1.5)
    assert not centre_in_margin_frame(0.05 * 854, 240, 854, 480, 0.8, 1.5)
    # Vertical 1.5 still allows a bit above/below the picture.
    assert centre_in_margin_frame(427, -0.2 * 480, 854, 480, 0.8, 1.5)
    assert not centre_in_margin_frame(427, -0.3 * 480, 854, 480, 0.8, 1.5)


def _window(boxes_at):
    return [(t, box, "car") for t, box in boxes_at]


def test_head_on_growth_is_collision():
    cfg = CollisionConfig(frame_gap=15, min_samples=3, min_growth_ratio=1.01, max_ttc_seconds=10)
    # Width 200 → 320 over 15 frames, centred. Screen 400 wide → fills in ~15 more frames.
    samples = _window(
        [
            (100, Box(100, 140, 300, 260)),
            (107, Box(80, 130, 320, 270)),
            (115, Box(40, 120, 360, 280)),
        ]
    )
    f = forecast_from_window(samples, 115, 30.0, 400, 400, cfg, 1, "car")
    assert f.will_collide, f.reason
    assert f.ttc_seconds is not None and 0 < f.ttc_seconds < 2.0, f.ttc_seconds
    assert abs(f.impact_cx - 200) < 30, f.impact_cx


def test_past_fill_uses_current_centre():
    cfg = CollisionConfig(frame_gap=15, min_samples=3, min_growth_ratio=1.0, max_ttc_seconds=10)
    # Already wider than the 200px screen; centre stays in the middle.
    samples = _window(
        [
            (100, Box(-20, 40, 220, 160)),
            (107, Box(-30, 35, 230, 165)),
            (115, Box(-40, 30, 240, 170)),
        ]
    )
    f = forecast_from_window(samples, 115, 30.0, 200, 200, cfg, 2, "car")
    assert f.ttc_seconds == 0.0, f.ttc_seconds
    assert f.will_collide, f.reason
    # Must not jump back to the fill-crossing time (would shift the centre).
    assert 90 < f.impact_cx < 110, f.impact_cx


def test_side_passer_is_not_collision():
    cfg = CollisionConfig(frame_gap=15, min_samples=3, min_growth_ratio=1.01, max_ttc_seconds=10)
    samples = _window(
        [
            (100, Box(300, 140, 340, 200)),
            (107, Box(360, 140, 410, 210)),
            (115, Box(430, 140, 490, 220)),
        ]
    )
    f = forecast_from_window(samples, 115, 30.0, 400, 400, cfg, 3, "car")
    assert not f.will_collide, f.reason


def _raw(track_id: int, hit: bool) -> Forecast:
    return Forecast(track_id, "car", hit, "geo", 0.4, 100.0, 100.0)


def test_confirm_needs_k_consecutive_hits():
    latch = HitLatch(3)
    first = latch.apply([_raw(1, True)], {1})[0]
    assert first.raw_collide and not first.will_collide
    assert first.hit_streak == 1
    second = latch.apply([_raw(1, True)], {1})[0]
    assert not second.will_collide and second.hit_streak == 2
    third = latch.apply([_raw(1, True)], {1})[0]
    assert third.will_collide and third.hit_streak == 3


def test_miss_breaks_confirm_streak():
    latch = HitLatch(3)
    latch.apply([_raw(1, True)], {1})
    latch.apply([_raw(1, True)], {1})
    miss = latch.apply([_raw(1, False)], {1})[0]
    assert not miss.will_collide and miss.hit_streak == 0
    again = latch.apply([_raw(1, True)], {1})[0]
    assert not again.will_collide and again.hit_streak == 1


def test_dropout_resets_confirm_streak():
    latch = HitLatch(3)
    latch.apply([_raw(1, True)], {1})
    latch.apply([_raw(1, True)], {1})
    latch.apply([], set())
    again = latch.apply([_raw(1, True)], {1})[0]
    assert not again.will_collide and again.hit_streak == 1


def test_live_without_forecast_resets_streak():
    latch = HitLatch(3)
    latch.apply([_raw(1, True)], {1})
    latch.apply([_raw(1, True)], {1})
    latch.apply([], {1})
    again = latch.apply([_raw(1, True)], {1})[0]
    assert not again.will_collide and again.hit_streak == 1


def test_confirm_hits_one_is_immediate():
    latch = HitLatch(1)
    hit = latch.apply([_raw(7, True)], {7})[0]
    assert hit.will_collide and hit.raw_collide
    miss = latch.apply([_raw(7, False)], {7})[0]
    assert not miss.will_collide and not miss.raw_collide


if __name__ == "__main__":
    test_ols_known_line()
    test_margin_axes_are_independent()
    test_head_on_growth_is_collision()
    test_past_fill_uses_current_centre()
    test_side_passer_is_not_collision()
    test_confirm_needs_k_consecutive_hits()
    test_miss_breaks_confirm_streak()
    test_dropout_resets_confirm_streak()
    test_live_without_forecast_resets_streak()
    test_confirm_hits_one_is_immediate()
    print("all collision checks passed")
