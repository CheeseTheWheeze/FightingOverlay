import os
import sys
import unittest
from unittest import mock

from apps.control_center import main as control_main


class ControlCenterStartupTests(unittest.TestCase):
    def test_main_safe_returns_nonzero_when_opencv_missing(self) -> None:
        original_argv = sys.argv[:]
        original_env = os.environ.copy()
        sys.argv = ["main.py"]
        os.environ["FIGHTINGOVERLAY_SUPPRESS_UI"] = "1"
        try:
            with mock.patch("apps.control_center.main.get_cv2", side_effect=ImportError("no cv2")):
                result = control_main.main_safe()
        finally:
            sys.argv = original_argv
            os.environ.clear()
            os.environ.update(original_env)
        self.assertNotEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
