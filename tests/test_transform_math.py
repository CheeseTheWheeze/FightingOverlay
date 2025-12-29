import unittest

from core.pipeline import _build_transform, _convert_xy


class TransformMathTests(unittest.TestCase):
    def test_letterbox_mapping(self) -> None:
        track = {
            "track_id": "t1",
            "source": {
                "coord_space": "pixels_in_infer_canvas",
                "transform_kind": "LETTERBOX",
                "infer_width": 1280,
                "infer_height": 1280,
                "resized_width": 1280,
                "resized_height": 720,
                "pad_left": 0,
                "pad_right": 0,
                "pad_top": 280,
                "pad_bottom": 280,
                "crop_x": 0,
                "crop_y": 0,
                "crop_w": 1920,
                "crop_h": 1080,
            },
        }
        transform = _build_transform(track, 1920, 1080)
        x, y = _convert_xy(640, 640, transform)
        self.assertAlmostEqual(x, 960, delta=1.0)
        self.assertAlmostEqual(y, 540, delta=1.0)

    def test_crop_mapping(self) -> None:
        track = {
            "track_id": "t1",
            "source": {
                "coord_space": "pixels_in_resized_content",
                "transform_kind": "CROP",
                "infer_width": 640,
                "infer_height": 360,
                "resized_width": 640,
                "resized_height": 360,
                "pad_left": 0,
                "pad_right": 0,
                "pad_top": 0,
                "pad_bottom": 0,
                "crop_x": 200,
                "crop_y": 100,
                "crop_w": 320,
                "crop_h": 180,
            },
        }
        transform = _build_transform(track, 1920, 1080)
        x, y = _convert_xy(320, 180, transform)
        self.assertAlmostEqual(x, 360, delta=0.5)
        self.assertAlmostEqual(y, 190, delta=0.5)


if __name__ == "__main__":
    unittest.main()
