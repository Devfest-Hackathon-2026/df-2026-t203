"""
DTW-based movement scoring for evaluating transitions between keyposes.
Compares student movement sequences against teacher sequences using
Dynamic Time Warping with per-limb breakdown.
"""

import numpy as np
from scipy.spatial.distance import euclidean

# Try to import fastdtw, fall back to a simple implementation
try:
    from fastdtw import fastdtw
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False


# Limb groups for per-limb DTW scoring
LIMB_GROUPS = {
    "right_arm": [12, 14, 16],   # shoulder, elbow, wrist
    "left_arm":  [11, 13, 15],
    "right_leg": [24, 26, 28],   # hip, knee, ankle
    "left_leg":  [23, 25, 27],
    "torso":     [11, 12, 23, 24],  # shoulders + hips
}

# Landmark indices for major body joints (excluding face/fingers)
BODY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def _simple_dtw(seq1, seq2, dist_func=euclidean):
    """
    Simple O(N*M) DTW implementation as fallback when fastdtw is not installed.
    
    :param seq1: First sequence (list of vectors)
    :param seq2: Second sequence (list of vectors)
    :param dist_func: Distance function between two vectors
    :returns: (distance, path)
    """
    n, m = len(seq1), len(seq2)
    cost_matrix = np.full((n + 1, m + 1), np.inf)
    cost_matrix[0, 0] = 0.0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = dist_func(seq1[i - 1], seq2[j - 1])
            cost_matrix[i, j] = d + min(
                cost_matrix[i - 1, j],      # insertion
                cost_matrix[i, j - 1],      # deletion
                cost_matrix[i - 1, j - 1],  # match
            )
    
    # Backtrack to find path
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (cost_matrix[i - 1, j - 1], i - 1, j - 1),
                (cost_matrix[i - 1, j], i - 1, j),
                (cost_matrix[i, j - 1], i, j - 1),
            ]
            _, i, j = min(candidates, key=lambda x: x[0])
    path.reverse()
    
    return cost_matrix[n, m], path


def dtw_distance(seq1, seq2):
    """
    Compute DTW distance between two sequences.
    Uses fastdtw if available, otherwise falls back to simple DTW.
    
    :param seq1: First sequence as list/array of vectors
    :param seq2: Second sequence as list/array of vectors
    :returns: (normalized_distance, path)
    """
    if len(seq1) < 2 or len(seq2) < 2:
        return float('inf'), []
    
    if HAS_FASTDTW:
        distance, path = fastdtw(seq1, seq2, dist=euclidean)
    else:
        distance, path = _simple_dtw(seq1, seq2)
    
    # Normalize by path length so longer sequences aren't penalized
    normalized = distance / max(len(path), 1)
    return normalized, path


def extract_limb_sequence(pose_sequence, landmark_indices):
    """
    Extract a sub-sequence containing only the specified landmark indices.
    
    :param pose_sequence: List of pose dicts, each mapping landmark_id -> [x, y, z, ...]
    :param landmark_indices: List of landmark indices to extract
    :returns: List of flattened coordinate vectors for those landmarks
    """
    result = []
    for pose in pose_sequence:
        coords = []
        for idx in landmark_indices:
            key = str(idx) if str(idx) in pose else idx
            if key in pose:
                val = pose[key]
                coords.extend(val[:3])  # x, y, z only
            else:
                coords.extend([0.0, 0.0, 0.0])
        result.append(np.array(coords))
    return result


def score_transition(teacher_sequence, student_sequence):
    """
    Score how well the student performed a movement transition.
    
    Returns an overall score and per-limb breakdown.
    
    :param teacher_sequence: List of teacher pose dicts between two keyposes
    :param student_sequence: List of student pose dicts captured during the transition
    :returns: dict with 'overall_score' (0-100, higher is better) and 
              'limb_scores' dict mapping limb names to scores
    """
    if len(teacher_sequence) < 2 or len(student_sequence) < 2:
        return {"overall_score": 0.0, "limb_scores": {}, "raw_distances": {}}
    
    limb_scores = {}
    raw_distances = {}
    
    for limb_name, indices in LIMB_GROUPS.items():
        teacher_limb = extract_limb_sequence(teacher_sequence, indices)
        student_limb = extract_limb_sequence(student_sequence, indices)
        
        dist, _ = dtw_distance(teacher_limb, student_limb)
        raw_distances[limb_name] = dist
        
        # Convert distance to a 0-100 score
        # Use a sigmoid-like mapping: score = 100 * exp(-k * dist)
        # k controls sensitivity -- tuned for normalized pose coordinates
        k = 2.0
        score = 100.0 * np.exp(-k * dist)
        limb_scores[limb_name] = round(score, 1)
    
    # Overall score is the weighted mean of limb scores
    weights = {
        "right_arm": 1.0, "left_arm": 1.0,
        "right_leg": 1.0, "left_leg": 1.0,
        "torso": 1.5,  # torso alignment is most important
    }
    weighted_sum = sum(limb_scores[k] * weights.get(k, 1.0) for k in limb_scores)
    total_weight = sum(weights.get(k, 1.0) for k in limb_scores)
    overall = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0
    
    return {
        "overall_score": overall,
        "limb_scores": limb_scores,
        "raw_distances": raw_distances,
    }


def format_score_display(score_result):
    """
    Format the DTW score result into a human-readable string for overlay display.
    
    :param score_result: Output from score_transition()
    :returns: List of strings for display
    """
    lines = [f"Movement Score: {score_result['overall_score']:.0f}%"]
    
    # Sort limbs for consistent display order
    display_order = ["right_arm", "left_arm", "right_leg", "left_leg", "torso"]
    display_names = {
        "right_arm": "R.Arm", "left_arm": "L.Arm",
        "right_leg": "R.Leg", "left_leg": "L.Leg",
        "torso": "Torso"
    }
    
    for limb in display_order:
        if limb in score_result["limb_scores"]:
            score = score_result["limb_scores"][limb]
            name = display_names.get(limb, limb)
            # Create a simple bar visualization
            bar_len = int(score / 10)
            bar = "\u2588" * bar_len + "\u2591" * (10 - bar_len)
            check = " \u2713" if score >= 80 else ""
            lines.append(f"  {name}: {bar} {score:.0f}%{check}")
    
    return lines

