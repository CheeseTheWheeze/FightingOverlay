import os
import subprocess
import sys
import unittest


class ControlCenterSmokeTests(unittest.TestCase):
    def test_ui_smoke(self) -> None:
        if os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self.skipTest("UI smoke test requires a display (set DISPLAY/WAYLAND_DISPLAY).")
        if os.environ.get("FIGHTINGOVERLAY_UI_SMOKE") != "1":
            self.skipTest("Set FIGHTINGOVERLAY_UI_SMOKE=1 to enable UI smoke test.")
        env = os.environ.copy()
        env["FIGHTINGOVERLAY_UI_SMOKE"] = "1"
        command = [sys.executable, "apps/control_center/main.py", "--ui-smoke-test"]
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            self.fail(
                "UI smoke test failed.\nstdout:\n"
                f"{result.stdout}\n\nstderr:\n{result.stderr}"
            )


if __name__ == "__main__":
    unittest.main()
