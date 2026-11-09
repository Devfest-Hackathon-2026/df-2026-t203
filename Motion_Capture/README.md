# new-pipeline-mocap — one video in, one Blender animation out

A trial motion-capture pipeline: it takes a **single** video from
[src/avatar/videos/](../videos/), tracks the signer with MediaPipe, and produces a
Blender scene where `src/avatar/avatar-models/avatar_model.fbx` performs the same motion,
baked to keyframes. That model is not committed (`*.fbx` is git-ignored); place it in that
folder locally, or point `--model` at your copy.

```
video.mp4 ──▶ extract_landmarks.py ──▶ landmarks.json ──▶ postprocess_landmarks.py ──▶ landmarks_filtered.json
            (python 3.11 + mediapipe)                    (skellyforge: freemocap's
                                                          smoothing and gap filling)
                                                                                                 │
              ┌──────────────────────────────────────────────────────────────────────────────────┘
              ▼
        stabilize_hands.py ──▶ landmarks_stabilized.json ──▶ blender_retarget.py ──▶ Water.blend
        (occlusion handling,                                 (blender, headless)      (+ .fbx / .glb)
         joint limits, one euro)
```

## Run it

```bash
python src/avatar/new-pipeline-mocap/run_pipeline.py --video src/avatar/videos/Water.mp4
```

`run_pipeline.py` runs all three stages and imports none of their dependencies itself —
it finds an interpreter with mediapipe, one with skellyforge, and an installed Blender,
and shells out to each. Everything lands in `output/<video name>/`.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--preview` | Also render a few PNG frames, to check the result without opening Blender |
| `--debug-video` | Also write the annotated video showing what MediaPipe actually tracked |
| `--export-fbx` / `--export-glb` | Also export the animated avatar (`.glb` is what the web player loads) |
| `--mirror` | Flip the capture left/right (see *Handedness* below) |
| `--skip-extract` | Re-run from an existing `landmarks.json` instead of tracking again |
| `--no-postprocess` | Skip the smoothing and gap filling and retarget the raw landmarks |
| `--no-stabilize` | Skip stage 2.5 — no occlusion handling and no joint limits (see *Hands the tracker is guessing at*) |
| `--debug-hands` | Also write `debug_hands.mp4`: confidence, hold state and rigidity, frame-synced to the video |
| `--stabilize-config other.yaml` | Tune stage 2.5 from a different config file |
| `--filter-cutoff 4` | Butterworth cutoff in Hz (default 6.0, freemocap's own default). Lower is smoother and blunter |
| `--handshape-in-gaps interpolate` | Morph the fingers across an untracked gap instead of holding the last measured handshape |
| `--no-trim` | Animate the whole clip, including the lead-in before the hands are in view |
| `--max-frames 60` | Only process the first N frames, for a quick test |
| `--depth-scale 0.7` | Flatten MediaPipe's depth axis, the noisiest one, towards the camera plane |
| `--no-hands` / `--no-head` | Leave those bones in the rest pose |
| `--no-roll-calibration` | Keep MediaPipe's head roll as measured (see *Limits*) |

Any stage also runs on its own:

```bash
.venv311/Scripts/python.exe src/avatar/new-pipeline-mocap/extract_landmarks.py --video ... --out ...
.venv/Scripts/python.exe    src/avatar/new-pipeline-mocap/postprocess_landmarks.py --landmarks ... --out ...
.venv/Scripts/python.exe    src/avatar/new-pipeline-mocap/stabilize_hands.py --landmarks ... --out ...
blender --background --python src/avatar/new-pipeline-mocap/blender_retarget.py -- --landmarks ... --model ... --blend-out ...
```

Two tools sit alongside them, both needing only numpy and cv2:

```bash
# Draw what the stabiliser decided, frame by frame, against the source video.
.venv/Scripts/python.exe src/avatar/new-pipeline-mocap/debug_overlay.py     --video ... --debug stabilize_debug.json --before landmarks.json     --after landmarks_stabilized.json --out debug_hands.mp4

# Before/after statistics, as a regression check.
.venv/Scripts/python.exe src/avatar/new-pipeline-mocap/measure_stabilization.py     --raw landmarks.json --filtered landmarks_filtered.json --stabilized landmarks_stabilized.json
```

## What it needs

* **Blender** — found automatically under `C:/Program Files/Blender Foundation/*`, or
  pass `--blender`, or set `BLENDER_EXE`; with several installs the highest version wins.
  Verified against Blender 5.2 LTS; the one
  version-sensitive call (bone selection) handles the 4.x API too.
* **A Python with mediapipe** — in this repo that is `.venv311`, *not* the default
  `.venv`, because mediapipe 0.10.9 has no wheel for Python 3.14. Found automatically,
  or pass `--python`, or set `MEDIAPIPE_PYTHON`. See [requirements.txt](requirements.txt).
* **A Python with skellyforge** — freemocap's post-processing library, for stage 2. It is
  installed in this repo's `.venv`, need not be the same interpreter as mediapipe's, and
  stage 2 can be skipped entirely with `--no-postprocess`:

  ```bash
  .venv/Scripts/python.exe -m pip install "skellyforge @ git+https://github.com/freemocap/skellyforge@main"
  ```
* **A Mixamo-rigged model.** `avatar_model.fbx` is one (its bones carry a `mixamorig4:`
  prefix, which the script detects). A rig with different bone names fails at import
  with the list of bones it could not find. A rig without finger bones (Mixamo's "No
  Fingers" option) fails the same way unless run with `--no-hands`.

## How the retarget works

Every driven bone is solved as a **world rotation**, never as a position. The avatar is
not the person in the video and has different limb proportions, so matching joint
*positions* would put the hands in the wrong place; matching joint *directions*
transfers the pose and lets the avatar keep its own body.

* **Direction bones** — clavicles, upper arms, forearms, all finger bones. The measured
  direction is one landmark minus another (elbow − shoulder, fingertip − knuckle), and
  the bone swings from its rest direction onto it. The parent's rotation is applied
  first so a finger rolls with the palm.
* **Basis bones** — hips, upper chest, head, both hands. An orthonormal frame is built
  from the landmarks and from the rig's rest pose, and the bone takes
  `B_frame · B_rest⁻¹ · R_rest`. This solves roll as well as direction, which is why the
  hands are done this way: palm orientation carries meaning in JSL.

Each rotation is keyframed onto an empty, the bone gets a world-space Copy Rotation
constraint, and the whole thing is baked to plain keyframes so it survives export.
Neck and lower chest follow their neighbours at partial influence so a turn is shared
across joints rather than snapping at one.

## How the smoothing works (stage 2)

Stage 2 is freemocap's own post-processing, called through
[skellyforge](https://github.com/freemocap/skellyforge) — the same two functions, in the
same order, that freemocap calls from its `skeleton_from_mediapipe_observations.py`:
linear interpolation across gaps, then a 4th-order Butterworth low-pass at 6 Hz run
through `filtfilt`, so it has no phase lag. The sampling rate is taken from the video's
real frame rate; leaving skellyforge's 30 Hz default on 60 fps footage would put the true
cutoff at 12 Hz and pass most of what the filter exists to remove. freemocap fixed that
same bug in its own pipeline.

Five things happen around those two calls, each because the sample clip showed the need:

* **Hands are split into orientation and shape before filtering.** MediaPipe reports hand
  landmarks relative to the wrist, so turning a hand over rotates all 21 points about the
  origin. Low-pass filtering those coordinates averages opposite vectors and shrinks the
  hand towards a point — on this clip a genuine 180° wrist turn came out as a collapsed
  hand whose orientation then swung wildly, a flip the filter *created*. So each hand
  becomes a palm orientation plus the finger shape in palm-local coordinates; both filter
  cleanly, and the hand is rebuilt afterwards. The body needs none of this, being already
  in a hip-centred frame that does not spin.
* **Hand gaps are filled before skellyforge sees them.** A straight line between two
  handshapes is not a handshape: halfway through this clip's 14-frame gap, interpolated
  fingers had curled to 54% of their length and twisted back out, none of it measured. So
  while a hand is untracked its fingers **hold the last shape they were seen in**, and its
  palm turns along the shortest arc towards where it reappears rather than having each
  axis interpolated separately. The final `--gap-return-blend` seconds of a gap (0.15 s,
  and never more than half the gap) ease the held shape into the pose the hand is about to
  be measured in, so tracking resumes without a snap. `--handshape-in-gaps interpolate`
  restores the old behaviour.
* **The clip is trimmed to where the hands are in view.** A take usually opens with the
  signer standing at rest, hands down or clasped, before they start. There is no handshape
  to use in those frames, so the avatar was posing them from the first shape measured
  later — a pose the signer never held, which is what made the opening look wrong. The
  animation now starts at the first frame with a hand in view and ends at the last;
  on the sample clip that drops 0.78 s of lead-in. One hand is enough to start, since not
  every sign is two-handed. `--no-trim` keeps everything.
* **Orientation spikes are dropped, not smoothed.** Where the signer's hands overlap,
  MediaPipe occasionally reconstructs one badly for a frame or two: the palm swings 150°
  and comes straight back. Filtering only smears such a frame over its neighbours, so it
  is deleted and the gap filling draws across it. A rotation that does *not* return is
  kept, however large — that is what real motion looks like.
* **Palm rotation is rate-limited** to `--max-wrist-speed`, 1500 °/s by default. On this
  clip the right palm jumps 129° between two frames — nearly 8000 °/s, which no wrist
  does, but it does not come back either, so the spike check correctly leaves it alone.
  The limit follows such a jump over several frames instead of snapping through it.

What is *not* used is freemocap's 3D solver. It triangulates several calibrated cameras,
and its single-camera fallback (`project_2d_batch_to_3d`) maps image pixels onto the Y=0
plane — no depth at all. MediaPipe's world landmarks are the better source of 3D from one
camera, so the depth estimate stays MediaPipe's and the smoothing comes from freemocap.

### What it measured on the sample clip

Per-frame bone rotation on `Water.mp4` (224 frames, 60 fps), raw against post-processed.
"Jitter" is the mean frame-to-frame change in rotation speed; "worst step" is the largest
single-frame jump. These are one clip's numbers, measured, not a guarantee:

| Bone | Median step | Worst step | Jitter |
| --- | --- | --- | --- |
| RightHand | 2.86° → 1.89° | 147.4° → 25.0° | 6.26 → 1.19 |
| RightHandIndex1 | 3.96° → 2.76° | 165.5° → 35.6° | 6.49 → 1.75 |
| LeftHand | 1.43° → 0.92° | 61.9° → 25.1° | 1.93 → 0.55 |
| RightForeArm | 0.80° → 0.75° | 7.7° → 6.1° | 0.43 → 0.21 |
| Head | 0.19° → 0.17° | 1.7° → 1.0° | 0.11 → 0.04 |

Those numbers were measured before the out-of-frame rejection and the trim, on the full
224-frame clip; both stages changed which frames exist, so treat them as a comparison of
smoothing on identical input rather than as current output statistics.

## Hands the tracker is guessing at (stage 2.5)

When a hand turns side-on to the camera, or one finger passes behind another, MediaPipe
keeps reporting all 21 landmarks and gives no sign that it is guessing. The fingers snap
into bends no hand makes. Stage 2's low-pass filter cannot fix that — it averages a bad
frame with its neighbours, which spreads the error rather than removing it. Measured on
`Water.mp4`, the Butterworth left the fraction of joint angles outside anatomical range
slightly *worse* than it found it: 45.3% → 46.8% on the right hand.

Stage 2.5 works in **joint-angle space**, which nothing else in this pipeline does. Each
hand is decomposed per frame into a palm orientation plus 20 joint angles (per finger: a
2-DOF root joint, then two hinges), the angles are cleaned, and the 21 landmarks are
rebuilt from them. Because the angles are measured in the palm's own frame, a finger whose
angles are being held still rides the tracked wrist — it stops articulating without
freezing in space.

### There is no confidence score to read

**MediaPipe reports `visibility` and `presence` as exactly `0.0` on every hand landmark**,
image-space and world. That is measured, not assumed. So the confidence this stage
thresholds is derived from four signals, weighted in [stabilize.yaml](stabilize.yaml):

| Signal | What it catches | Measured on `Water.mp4` |
| --- | --- | --- |
| **Rigidity residual** | Bones do not change length. Spread against the hand's own median is direct evidence the solver fitted noisy 2D to impossible 3D. | Per detection: median 9.6%, p90 37.7%, max 85.1% |
| **Out-of-frame fraction** | Landmarks outside the picture are extrapolated by MediaPipe, not seen. | The signer's left hand sits below the bottom edge for most of the take |
| **Palm facing** | A palm edge-on to the camera has almost no depth information. | Separates well on one hand and barely at all on the other — a contributing term, not a gate |
| **Handedness score** | Collapses towards 0.5 exactly where the hands overlap. | Minimum 0.52 |

Rigidity is weighted so that it can condemn a frame on its own: the other three weights sum
to less than the threshold on purpose, because a hand can be fully in shot, square to the
camera, confidently classified — and still reconstructed as nonsense. That is the case this
stage exists for. If you re-tune the weights, keep that property.

Being edge-on is deliberately *not* enough on its own. On frame 208 of the sample clip the
signer's right hand is fully edge-on (`palm facing 0.00`) while its rigidity residual is 2%:
the handshape is foreshortened but self-consistent, and all five fingers stay live. On
frame 81 the same hand is at 35% residual and four of five fingers are held.

### What happens to a finger that goes unreliable

1. **Hysteresis.** Three consecutive low-confidence frames before holding, four good ones
   before trusting live data again. Different numbers on purpose: recovering eagerly
   re-introduces the flicker the hold was there to remove.
2. **Hold, then extrapolate, then decay.** The last good angles and angular velocity are
   cached. For the first 200 ms the finger dead-reckons forward, so it keeps moving
   plausibly instead of looking dead. Past that it decays towards a relaxed pose, because
   unbounded extrapolation drifts into nonsense.
3. **Blend back.** Eight frames of smoothstep from the held pose to the live one. The real
   hand has usually moved while it was occluded, so snapping to live data pops.
4. **Anatomical limits, every frame.** Per-joint min/max on flexion and abduction, applied
   whether or not the finger is occluded — most unnatural bending happens while tracking is
   nominally fine.
5. **One Euro filter.** Adaptive smoothing on the angles: heavy when a joint is nearly
   still, light when it moves fast. It runs last and cannot undo the clamp, because each
   output is a convex blend of two values that were already inside the limits.

Every threshold above lives in [stabilize.yaml](stabilize.yaml) with a comment explaining
what moving it does. Nothing is hardcoded.

### What stage 2.5 measured on the sample clips

`Water.mp4`, comparing the same frames — only those where MediaPipe genuinely detected the
hand, so filled frames do not dilute the numbers. One clip, measured, not a guarantee:

> These figures, and the closed-fist figures below, were measured before two later fixes:
> bone lengths are now the median of each frame's lengths rather than the lengths of the
> median pose, and the thumb's palm-frame neutral spread and abduction limits are now moved
> onto the calibrated zero. They have not been re-measured since, so the stabilised columns
> may no longer match. Re-run `measure_stabilization.py` before quoting them.

| Right hand (163 frames) | raw | stage 2 only | stabilised |
| --- | --- | --- | --- |
| Joint angles outside anatomical range | 45.3% | 46.8% | **0.0%** |
| Frames with any violation | 100% | 100% | **0.0%** |
| Worst overshoot | 173.6° | 174.5° | 0.1° |
| Angle jitter (mean 2nd difference) | 27.37° | 17.35° | **2.03°** |
| Angle step (mean 1st difference) | 15.90° | 9.94° | 2.37° |
| Rigidity residual — median / p90 / max | 11.4% / 50.5% / 65.3% | 12.4% / 15.7% / 21.6% | 0% / 0% / 0% |

Across all six clips in [../videos/](../videos/) — eleven tracked hands, three different
signers, comparing only genuinely detected frames:

| Clip | Hand | Frames | Violations raw → stabilised | Jitter raw → stabilised | Fingers held |
| --- | --- | --- | --- | --- | --- |
| Earthquake | Left | 67 | 25.2% → 0.0% | 35.12° → 2.05° | 67% |
| Earthquake | Right | 70 | 19.1% → 0.0% | 18.96° → 1.77° | 64% |
| Evacuation | Left | 68 | 7.2% → 0.0% | 4.78° → 1.13° | 0% |
| Evacuation | Right | 68 | 12.8% → 0.0% | 14.01° → 2.49° | 10% |
| Headache | Right | 97 | 11.9% → 0.0% | 9.45° → 1.70° | 0% |
| Scared | Left | 112 | 7.3% → 0.0% | 3.99° → 0.47° | 20% |
| Scared | Right | 126 | 11.6% → 0.0% | 3.04° → 0.78° | 4% |
| Water | Left | 22 | 12.7% → 0.0% | 9.84° → 1.05° | 94% |
| Water | Right | 163 | 45.3% → 0.0% | 27.37° → 2.03° | 61% |
| fire | Left | 64 | 7.9% → 0.0% | 3.18° → 0.52° | 0% |
| fire | Right | 67 | 8.6% → 0.0% | 6.95° → 1.67° | 0% |
| **mean** | | | **15.4% → 0.0%** | **12.43° → 1.42°** | 29% |

Both columns are measured with the same abduction calibration the stabiliser used, which is
why the raw column is lower than it would be against a naive zero — measuring the two sides
against different definitions of "no spread" would flatter the output.

`Headache` has no left hand at all — it stays out of shot for the whole take, and the stage
reports that rather than inventing one. `Water` is the outlier for holding: its right hand is
edge-on for most of the clip and its left rests below the bottom of frame.

Read that table in order. **The joint-limit rows are the honest headline** — they count
poses no hand can make, they are measured identically on both sides, and nothing about the
reconstruction forces them to improve. The 0.2° worst overshoot is not the filter: it is the
5-decimal rounding of metre coordinates when landmarks are written to JSON, worth up to
0.17° on the shortest bones.

**The rigidity residual of the stabilised column is ~0 by construction and proves nothing.**
The hand is rebuilt by forward kinematics on fixed bone lengths, so it cannot come out any
other way whatever the input was. It is in the table because the *before* column is real and
is what drives the detection. Do not quote the after column as evidence of anything.

Regenerate the whole table with `measure_stabilization.py`.

### Closed fists

A fist is the hardest handshape for a monocular tracker: the fingertips tuck behind the
knuckles, so more than half the finger landmarks are inferred rather than seen. Measured on
`Evacuation.mp4` and `Scared.mp4`, both of which hold two fists to camera:

| | Evacuation L | Evacuation R | Scared L | Scared R |
| --- | --- | --- | --- | --- |
| Frames reading as a fist | 40/68 | 49/68 | 106/112 | 101/126 |
| Still a fist afterwards | 39/40 | 45/49 | **106/106** | 100/101 |
| Median curl, raw → stabilised | 58.1° → 58.4° | 61.4° → 61.7° | 76.6° → 76.6° | 75.0° → 75.1° |
| Fingers held on fist frames | 0.00 | 0.00 | 0.57 | 0.00 |
| Fingers held elsewhere | 0.00 | 1.79 | 0.83 | 0.48 |

Fists come through intact, and the last two rows say why: a fist is held *less* than the rest
of the clip. It is a compact, self-consistent shape with a low rigidity residual, so the
confidence score likes it and the fingers stay live. The failure this stage exists for is a
hand caught mid-motion or edge-on, not a hand holding a firm shape.

That was not a given. Two things had to be got right, and one of them was wrong at first:

* **The MCP ceiling was too low.** Fist frames sit at a median of 74–86° of MCP flexion with
  a p95 of 91–99°, so the original flat 90° limit clipped a real fist on 23–27% of them. The
  ceiling now follows the ulnar gradient (95° index and middle, 100° ring and pinky) and the
  median curl passes through unchanged.
* **A long hold would open a fist.** The decay target is a relaxed hand at 23° of curl,
  against a fist's 90°. Nothing in the clips above holds long enough for that to bite — the
  longest is 13 frames — but a fist occluded for a second would visibly open. That is the
  intended behaviour, since the alternative is asserting a handshape nobody measured, but it
  is worth knowing before trusting a long gap.

### Looking at it rather than trusting it

`--debug-hands` writes `debug_hands.mp4`: the source video, the raw and stabilised
handshapes drawn side by side as wireframes, per-finger confidence and live/hold/blend
state, and a scrolling chart of confidence and rigidity against the threshold, with the
hold and blend runs marked underneath. Aggregate numbers cannot tell you *which* frames
were caught — this can.

## Hands that are only half in shot

MediaPipe reports all 21 landmarks for a hand that is partly outside the picture,
extrapolating the ones it cannot see, and it does so with no signal that it is guessing.
On the sample clip the signer's resting hand sits below the bottom edge for the entire
take: detected on almost every frame, with 57% of its landmarks outside the image and a
bounding box that never moves. Posed from that, the avatar's hand splayed into a shape
nobody performed and held it for the whole clip.

Stage 1 therefore rejects a detection when more than `--max-out-of-frame` of its landmarks
(30% by default) fall outside the picture, and says how many it rejected — 145 of 330 on
this clip. A hand nobody could see gets no keyframes at all and keeps the rig's rest pose,
which is both better looking and more honest than inventing one. Raise the threshold for
footage framed tightly on the hands.

## Handedness

Which hand is which is decided by matching each detected hand to the nearer **pose**
wrist in image space, not by MediaPipe's handedness classifier. That classifier flips on
individual frames exactly where it matters — when the signer's hands meet, with scores
around 0.55 — and a swapped handshape is not something a later stage can undo. On the
sample clip the geometric match overrides it on 3 detections, and those 3 produced the
largest snaps in the raw output. The winning distance there is a few pixels against a
couple of hundred for the alternative, so it is not a close call.

Two flags remain, for what geometry cannot settle:

* `--selfie` — the footage really is mirrored. This now affects only the classifier
  fallback, which is used when the pose wrists are unavailable.
* `--mirror` — flip the whole capture left/right at retarget time.

If the avatar signs with the wrong hand, `--mirror` is the flag to reach for. Check the
result against `--debug-video` before deciding.

## Limits — read before trusting any output

* **Nothing this produces is reviewed JSL.** It is unverified motion copied off a video.
  Anything shown in the UI from it must be labelled 未監修, per
  [CLAUDE.md](../../../CLAUDE.md).
* **No face, mouth shape or gaze.** Mouth shapes are part of JSL, so the output is not a
  complete sign on its own.
* **No root motion.** MediaPipe's world landmarks are hip-centred, so one uncalibrated
  camera gives no usable global translation. The avatar stays planted.
* **No legs, no spine curvature.** MediaPipe has no landmark between hips and shoulders,
  so the torso bends at the hips and the upper chest and nowhere else.
* **The head roll is calibrated, not measured.** MediaPipe's *world* landmarks carry a
  constant roll bias about the camera axis that its image-space landmarks do not — on
  the sample clip the ear axis reads 16° off level in 3D while the same ears are within
  a couple of degrees of level in the image. The retarget subtracts the clip's median
  roll, which keeps relative head tilt but assumes the signer's average head over the
  clip is upright. `--no-roll-calibration` turns that off.
* **Depth is the weak axis.** Monocular depth error shows up as elbows that swing too far
  from the body and as noise in torso twist. `--depth-scale` below 1.0 trades that noise
  for a flatter, more camera-facing pose.
* **Filled gaps are invented, not measured.** Where a hand was not tracked the avatar
  holds the last handshape it saw and its arm carries on — on the sample clip that covers
  a one-second gap. Holding is the most honest option available (it repeats a real
  measurement rather than inventing a new one), but the avatar is still signing frames
  nobody performed. Every frame records which hands were actually seen, under
  `hands_measured` in `landmarks_filtered.json`, and stage 2 prints how much it filled.
* **The filter blunts fast finger detail.** 6 Hz is freemocap's default for whole-body
  capture, and fingers move faster than a torso. If handshapes look mushy, raise
  `--filter-cutoff`; if the result still jitters, lower it. With stage 2.5 enabled the
  Butterworth is kept off the finger shape entirely and the One Euro filter smooths the
  joint angles instead — two low-pass filters over one signal cannot be tuned separately.
* **Held fingers are invented, and there are a lot of them.** On the sample clip the
  signer's left hand is held on 90% of finger-frames and the right on 50%. A held finger is
  not being measured: it is dead-reckoned and then decayed towards a relaxed pose.
  `stabilize_debug.json` records the state of every finger on every frame, and
  `hands_reliable` in the output says which hands were trusted at all.
* **The abduction zero is calibrated per clip, not anatomical.** Finger spread is measured
  from each finger's own neutral, taken as the median over that clip's better-tracked frames
  — without it the knuckle arch reads as a permanent splay and abduction was being clamped on
  25–60% of angle-frames. Calibrating drops that to 4–23%. But the zero is then *that clip's*
  typical spread: on a clip that is mostly one handshape it describes that handshape, so the
  offsets are not a measurement of the signer's neutral hand. They are clamped to the
  anatomical range, because on a badly-tracked hand the median pose is itself unreliable —
  one hand in the sample set produced a 56° "resting" splay before that guard was added.
  The four fingers' limits and neutral spread read against this calibrated zero; the thumb's
  are written in the palm frame, so they are shifted by the thumb's offset before use.
* **The joint limits are unverified.** They are textbook clinical range-of-motion figures
  for a typical adult hand, not measurements of any signer in this repository, and no
  reviewer has checked them. The thumb's two proximal angles are measured in the palm frame
  rather than clinically — see the note in [stabilize.yaml](stabilize.yaml).
* **Stage 2.5 rebuilds the hand rather than adjusting it.** Bone lengths and palm size come
  out exactly constant, taken from each hand's better-tracked frames: bone lengths as the
  median of each frame's own lengths, the palm as its median shape. That
  removes a large part of what looked anatomically impossible, but it also means the output
  carries no information about how long the signer's bones actually were on any given
  frame.
* **No accuracy figure has been measured for this pipeline.** Do not quote one.

## Relationship to freemocap

Stage 2 calls freemocap's post-processing library directly, as described above. That
library is [skellyforge](https://github.com/freemocap/skellyforge), installed with pip
into this repo's `.venv` — it is the only part of freemocap this pipeline depends on, and
it needs nothing else from the project:

```bash
.venv/Scripts/python.exe -m pip install "skellyforge @ git+https://github.com/freemocap/skellyforge@main"
```

The rest of the [freemocap](https://github.com/freemocap/freemocap) project is a
multi-camera system: it wants a recording folder of synchronised videos plus a
calibration, and its Blender add-on builds its *own* generic rig rather than animating a
supplied character. Neither fits "one video onto `avatar_model`", so the tracking and the
retarget are ours. What they borrow is freemocap's overall shape — detect, then solve,
then hand a finished file to Blender in a subprocess — and its trick of injecting
arguments after `--` on the Blender command line.

A full clone of freemocap was vendored at `../freemocap/` while that comparison was being
made. It was never committed, nothing here ever imported it, and it has been deleted; the
upstream repository is the link above if it is needed again.

Two further freemocap pieces were looked at and left out. Its **rigid-bone enforcement**
(`skeleton_rigidifier.py`) holds segment lengths constant, but our retarget reads
directions only and normalises them, so fixed lengths would change little — and it pulls
in skellytracker. Its **realtime point gate** needs that same dependency, which is why
the spike check lives here instead.

## Output folder

`output/` is git-ignored. It holds landmark data derived from footage of a real person,
which must never be committed — see [CLAUDE.md](../../../CLAUDE.md).
