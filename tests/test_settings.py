import unittest

from core.settings import apply_setting_change, safe_float, safe_int


class SettingsParsingTests(unittest.TestCase):
    def test_safe_float_invalid_empty_string(self) -> None:
        value = safe_float(("ui_scale_multiplier", ""), 1.0, min_value=0.5, max_value=2.5)
        self.assertEqual(value, 1.0)

    def test_safe_float_parses_trimmed_value(self) -> None:
        value = safe_float(("ui_scale_multiplier", " 1.25 "), 1.0, min_value=0.5, max_value=2.5)
        self.assertEqual(value, 1.25)

    def test_safe_float_invalid_string(self) -> None:
        value = safe_float(("ui_scale_multiplier", "abc"), 1.0, min_value=0.5, max_value=2.5)
        self.assertEqual(value, 1.0)

    def test_safe_int_clamps_low(self) -> None:
        value = safe_int(("ui_font_size", "8"), 12, min_value=9, max_value=32)
        self.assertEqual(value, 9)

    def test_safe_int_clamps_high(self) -> None:
        value = safe_int(("ui_font_size", "100"), 12, min_value=9, max_value=32)
        self.assertEqual(value, 32)

    def test_safe_int_invalid_empty_string(self) -> None:
        value = safe_int(("ui_font_size", ""), 12, min_value=9, max_value=32)
        self.assertEqual(value, 12)

    def test_safe_int_invalid_none(self) -> None:
        value = safe_int(("ui_font_size", None), 12, min_value=9, max_value=32)
        self.assertEqual(value, 12)

    def test_apply_setting_change_ignores_invalid(self) -> None:
        saved_values: list[object] = []

        def save(value: object) -> None:
            saved_values.append(value)

        last_good = 12
        new_value, saved = apply_setting_change(
            "abc",
            key="ui_font_size",
            cast=int,
            last_good=last_good,
            save=save,
        )

        self.assertFalse(saved)
        self.assertEqual(new_value, last_good)
        self.assertEqual(saved_values, [])


if __name__ == "__main__":
    unittest.main()
