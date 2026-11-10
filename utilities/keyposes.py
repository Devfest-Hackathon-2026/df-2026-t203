"""Select motion key poses from a per-frame energy signal."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema, find_peaks


JOINT_WEIGHTS = np.array([
    0.1,  # 0  - nose
    0.0,  # 1  - left eye (inner)
    0.0,  # 2  - left eye
    0.0,  # 3  - left eye (outer)
    0.0,  # 4  - right eye (inner)
    0.0,  # 5  - right eye
    0.0,  # 6  - right eye (outer)
    0.0,  # 7  - left ear
    0.0,  # 8  - right ear
    0.0,  # 9  - mouth (left)
    0.0,  # 10 - mouth (right)
    1.5,  # 11 - left shoulder
    1.5,  # 12 - right shoulder
    1.2,  # 13 - left elbow
    1.2,  # 14 - right elbow
    1.4,  # 15 - left wrist
    1.4,  # 16 - right wrist
    0.3,  # 17 - left pinky
    0.3,  # 18 - right pinky
    0.3,  # 19 - left index
    0.3,  # 20 - right index
    0.3,  # 21 - left thumb
    0.3,  # 22 - right thumb
    1.5,  # 23 - left hip
    1.5,  # 24 - right hip
    1.2,  # 25 - left knee
    1.2,  # 26 - right knee
    1.2,  # 27 - left ankle
    1.2,  # 28 - right ankle
    0.5,  # 29 - left heel
    0.5,  # 30 - right heel
    0.3,  # 31 - left foot index
    0.3,  # 32 - right foot index
])


def normalize_energy(energy):
    minimum = energy.min()
    energy_range = energy.max() - minimum
    if energy_range < 1e-9:
        return np.zeros_like(energy)
    return (energy - minimum) / energy_range


def keypose_indices(energy, fps, include_initial=True):
    """Return frame indices at the valleys after prominent movement peaks."""
    peaks, _ = find_peaks(energy, prominence=0.08, distance=max(int(fps * 0.15), 2))
    valleys = argrelextrema(energy, np.less, order=max(int(fps * 0.15), 2))[0]
    minimum_gap = max(int(fps * 0.2), 3)
    selected = []
    for peak in peaks:
        next_valley = valleys.searchsorted(peak, side="right")
        valley = valleys[next_valley] if next_valley < len(valleys) else peak
        # Energy[i] describes the movement into frame i + 1.
        frame = int(valley) + 1
        if not selected or frame - selected[-1] >= minimum_gap:
            selected.append(frame)
    if include_initial and (not selected or selected[0] != 1):
        selected.insert(0, 1)
    return selected


@lru_cache(maxsize=16)
def recording_keypose_times(path: Path, modified_ns: int) -> tuple[float, ...]:
    """Find held poses on the original video timeline using saved world landmarks."""
    data = json.loads(path.read_text(encoding="utf-8"))
    fps = float(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        return ()

    times = []
    track = []
    first_frame = 0

    def select_poses():
        if len(track) < 3:
            return
        distances = np.linalg.norm(np.diff(np.asarray(track), axis=0), axis=2)
        motion = distances @ JOINT_WEIGHTS / JOINT_WEIGHTS.sum()
        energy = normalize_energy(gaussian_filter1d(motion, sigma=max(1.0, fps * 0.2)))
        for index in keypose_indices(energy, fps, include_initial=not times):
            timestamp = (first_frame + index) / fps
            if not times or timestamp - times[-1] >= 0.2:
                times.append(timestamp)

    for index, frame in enumerate(data["frames"]):
        points = frame.get("pose_world")
        if points is not None:
            points = np.asarray(points, dtype=float)
            if points.ndim == 2 and points.shape[0] == len(JOINT_WEIGHTS) and points.shape[1] >= 3:
                xyz = points[:, :3]
                if np.isfinite(xyz).all():
                    if not track:
                        first_frame = index
                    track.append(xyz)
                    continue
        # Gaps must not look like sudden movement between nonconsecutive frames.
        select_poses()
        track.clear()
    select_poses()
    return tuple(times)
