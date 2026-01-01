from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk


@dataclass(frozen=True)
class Joint:
    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class FightSkeleton:
    joints: dict[str, Joint]

    def get(self, name: str) -> Joint | None:
        return self.joints.get(name)


@dataclass(frozen=True)
class AttackPoint:
    name: str
    x: float
    y: float


JOINT_CONNECTIONS = [
    ("head", "neck"),
    ("neck", "shoulder_left"),
    ("neck", "shoulder_right"),
    ("shoulder_left", "elbow_left"),
    ("elbow_left", "wrist_left"),
    ("shoulder_right", "elbow_right"),
    ("elbow_right", "wrist_right"),
    ("neck", "pelvis"),
    ("pelvis", "hip_left"),
    ("pelvis", "hip_right"),
    ("hip_left", "knee_left"),
    ("knee_left", "ankle_left"),
    ("hip_right", "knee_right"),
    ("knee_right", "ankle_right"),
]

DERIVED_LINES = [
    ("neck", "pelvis"),
    ("shoulder_left", "shoulder_right"),
    ("hip_left", "hip_right"),
]


def _landmark_xy(landmark, width: int, height: int) -> tuple[float, float]:
    return landmark.x * width, landmark.y * height


def _midpoint(a: Joint, b: Joint, name: str) -> Joint:
    return Joint(
        name=name,
        x=(a.x + b.x) / 2.0,
        y=(a.y + b.y) / 2.0,
        confidence=min(a.confidence, b.confidence),
    )


def _reduce_to_fight_skeleton(
    pose_landmarks: Iterable[mp.framework.formats.landmark_pb2.NormalizedLandmark],
    width: int,
    height: int,
) -> FightSkeleton:
    landmarks = list(pose_landmarks)
    joint_map: dict[str, Joint] = {}

    def add_joint(name: str, landmark_index: int) -> None:
        landmark = landmarks[landmark_index]
        x, y = _landmark_xy(landmark, width, height)
        joint_map[name] = Joint(name=name, x=x, y=y, confidence=landmark.visibility)

    add_joint("head", mp.solutions.pose.PoseLandmark.NOSE.value)
    add_joint("shoulder_left", mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value)
    add_joint("shoulder_right", mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value)
    add_joint("elbow_left", mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value)
    add_joint("elbow_right", mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value)
    add_joint("wrist_left", mp.solutions.pose.PoseLandmark.LEFT_WRIST.value)
    add_joint("wrist_right", mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value)
    add_joint("hip_left", mp.solutions.pose.PoseLandmark.LEFT_HIP.value)
    add_joint("hip_right", mp.solutions.pose.PoseLandmark.RIGHT_HIP.value)
    add_joint("knee_left", mp.solutions.pose.PoseLandmark.LEFT_KNEE.value)
    add_joint("knee_right", mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value)
    add_joint("ankle_left", mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value)
    add_joint("ankle_right", mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value)

    joint_map["neck"] = _midpoint(joint_map["shoulder_left"], joint_map["shoulder_right"], "neck")
    joint_map["pelvis"] = _midpoint(joint_map["hip_left"], joint_map["hip_right"], "pelvis")

    return FightSkeleton(joints=joint_map)


def _extend_point(start: Joint, end: Joint, extension: float) -> tuple[float, float]:
    dx = end.x - start.x
    dy = end.y - start.y
    length = math.hypot(dx, dy)
    if length == 0:
        return end.x, end.y
    scale = extension / length
    return end.x + dx * scale, end.y + dy * scale


def _compute_attack_points(skeleton: FightSkeleton) -> list[AttackPoint]:
    points: list[AttackPoint] = []

    for side in ("left", "right"):
        elbow = skeleton.get(f"elbow_{side}")
        wrist = skeleton.get(f"wrist_{side}")
        knee = skeleton.get(f"knee_{side}")
        ankle = skeleton.get(f"ankle_{side}")
        if elbow and wrist:
            extension = 0.3 * math.hypot(wrist.x - elbow.x, wrist.y - elbow.y)
            fx, fy = _extend_point(elbow, wrist, extension)
            points.append(AttackPoint(name=f"fist_{side}", x=fx, y=fy))
            points.append(AttackPoint(name=f"elbow_{side}", x=elbow.x, y=elbow.y))
        if knee:
            points.append(AttackPoint(name=f"knee_{side}", x=knee.x, y=knee.y))
        if knee and ankle:
            points.append(
                AttackPoint(
                    name=f"shin_{side}",
                    x=(knee.x + ankle.x) / 2.0,
                    y=(knee.y + ankle.y) / 2.0,
                )
            )

    head = skeleton.get("head")
    neck = skeleton.get("neck")
    pelvis = skeleton.get("pelvis")
    if head:
        points.append(AttackPoint(name="head", x=head.x, y=head.y))
    if neck:
        points.append(AttackPoint(name="neck", x=neck.x, y=neck.y))
    if pelvis:
        points.append(AttackPoint(name="hip", x=pelvis.x, y=pelvis.y))

    return points


def _draw_overlay(frame: np.ndarray, skeleton: FightSkeleton, attack_points: list[AttackPoint]) -> None:
    for joint_a, joint_b in JOINT_CONNECTIONS:
        a = skeleton.get(joint_a)
        b = skeleton.get(joint_b)
        if not a or not b:
            continue
        cv2.line(frame, (int(a.x), int(a.y)), (int(b.x), int(b.y)), (0, 255, 255), 2)

    for joint_a, joint_b in DERIVED_LINES:
        a = skeleton.get(joint_a)
        b = skeleton.get(joint_b)
        if not a or not b:
            continue
        cv2.line(frame, (int(a.x), int(a.y)), (int(b.x), int(b.y)), (255, 255, 0), 1)

    for joint in skeleton.joints.values():
        cv2.circle(frame, (int(joint.x), int(joint.y)), 4, (0, 200, 255), -1)

    for point in attack_points:
        cv2.circle(frame, (int(point.x), int(point.y)), 6, (0, 0, 255), -1)


def _resize_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width == 0 or height == 0:
        return frame
    scale = min(target_width / width, target_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


class FightOverlayApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Fighting Overlay - MVP")
        self.video_path: Path | None = None
        self.capture: cv2.VideoCapture | None = None
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.is_playing = False
        self.overlay_enabled = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        load_button = ttk.Button(toolbar, text="Load Video", command=self.load_video)
        load_button.pack(side=tk.LEFT, padx=4)

        play_button = ttk.Button(toolbar, text="Analyze / Play", command=self.toggle_play)
        play_button.pack(side=tk.LEFT, padx=4)

        overlay_toggle = ttk.Checkbutton(
            toolbar,
            text="Show Overlay",
            variable=self.overlay_enabled,
        )
        overlay_toggle.pack(side=tk.LEFT, padx=8)

        self.status_label = ttk.Label(toolbar, text="No video loaded")
        self.status_label.pack(side=tk.LEFT, padx=8)

        self.video_panel = ttk.Label(self.root)
        self.video_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def load_video(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video Files", "*.mp4;*.mov;*.avi;*.mkv"), ("All Files", "*.*")],
        )
        if not file_path:
            return
        self.video_path = Path(file_path)
        self.status_label.configure(text=f"Loaded: {self.video_path.name}")
        self._open_capture()
        self._display_first_frame()

    def _open_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
        if not self.video_path:
            return
        self.capture = cv2.VideoCapture(str(self.video_path))

    def _display_first_frame(self) -> None:
        if not self.capture:
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.capture.read()
        if not ok:
            return
        self._update_panel(frame)

    def toggle_play(self) -> None:
        if not self.video_path:
            self.status_label.configure(text="Load a video to analyze")
            return
        if not self.capture:
            self._open_capture()
        if not self.capture:
            return
        if not self.is_playing:
            self.is_playing = True
            self.status_label.configure(text="Analyzing...")
            self._play_next_frame()
        else:
            self.is_playing = False
            self.status_label.configure(text="Paused")

    def _play_next_frame(self) -> None:
        if not self.is_playing or not self.capture:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.is_playing = False
            self.status_label.configure(text="Playback complete")
            return

        display_frame = frame.copy()
        if self.overlay_enabled.get():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            if results.pose_landmarks:
                skeleton = _reduce_to_fight_skeleton(
                    results.pose_landmarks.landmark,
                    width=frame.shape[1],
                    height=frame.shape[0],
                )
                attack_points = _compute_attack_points(skeleton)
                _draw_overlay(display_frame, skeleton, attack_points)

        self._update_panel(display_frame)

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps and fps > 0 else 33
        self.root.after(delay, self._play_next_frame)

    def _update_panel(self, frame: np.ndarray) -> None:
        target_width = self.video_panel.winfo_width() or 960
        target_height = self.video_panel.winfo_height() or 540
        resized = _resize_frame(frame, target_width, target_height)
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=image)
        self.video_panel.configure(image=photo)
        self.video_panel.image = photo

    def shutdown(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.pose.close()


def main() -> int:
    root = tk.Tk()
    app = FightOverlayApp(root)

    def on_close() -> None:
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.geometry("1024x768")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
