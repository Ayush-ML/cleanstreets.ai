# CleanStreets AI

**An edge-deployable, rule-based computer vision system for real-time littering detection.**

CleanStreets AI watches a live camera feed, detects when a person picks up, holds, and abandons an object, and flags the sequence as a candidate littering incident — displaying the full detection state live on-screen and raising a desktop notification the moment an incident is confirmed. The system is built entirely on consumer-grade hardware (a standard laptop and its built-in webcam), using no paid APIs, no custom training data, and no cloud dependency of any kind.

---

## Demo

<video controls width="720" src="data/Demo.mp4">
  Your viewer does not support embedded video. Watch it directly at <a href="data/Demo.mp4">data/Demo.mp4</a>.
</video>

📹 **[Watch the full demo recording](data/Demo.mp4)**

The recording shows `main.py` running live: real-time bounding boxes around detected people and objects, wrist points overlaid whenever hand-landmark data is available, an object's box turning yellow and labeled `[HELD]` once it's confirmed as being held, a running status line (objects detected, wrists tracked, objects held, drops in progress), and — the moment a full littering sequence is confirmed — an on-screen "INCIDENT TRIGGERED" banner alongside a desktop toast notification reporting the detection confidence and timestamp.

*(GitHub and most Git-based hosts render `.mp4` files with an inline player automatically when linked directly from a Markdown file in the repository. If viewing this file outside of such a host, open `data/Demo.mp4` directly in a media player.)*

---

## Table of Contents

- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [Design Philosophy](#design-philosophy)
- [System Architecture](#system-architecture)
- [Detection Pipeline, Stage by Stage](#detection-pipeline-stage-by-stage)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [What You'll See](#what-youll-see)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Related Work](#related-work)
- [Future Work](#future-work)
- [Hardware Notes](#hardware-notes)
- [License](#license)

---

## Motivation

Littering enforcement today relies almost entirely on either physical patrols or continuous human monitoring of CCTV footage. Both approaches scale poorly: patrols cover a tiny fraction of any area at any given time, and manual footage review is labor-intensive precisely because the overwhelming majority of any camera feed contains no relevant activity at all. CleanStreets AI explores a different model: an AI system that watches continuously and only calls attention to a moment — via a live on-screen flag and a desktop notification — when a genuine candidate littering sequence has actually been detected.

## Problem Statement

Although littering is prohibited in many jurisdictions, enforcement is difficult because continuous monitoring is labor-intensive, most surveillance footage contains no relevant events, and human operators cannot efficiently monitor many cameras simultaneously. The system proposed here automatically detects suspicious littering events in real time and surfaces them immediately, rather than requiring a human to watch continuously for something that occurs rarely.

## Design Philosophy

Several deliberate constraints shaped every design decision in this project:

- **No fully autonomous enforcement.** The system never issues fines, identifies individuals, or makes a final legal determination. It surfaces a detected event; a human decides what to do about it.
- **No custom training data required.** Rather than training a dedicated action-recognition model (which would require a labeled dataset of littering behavior that does not readily exist), the system uses a pretrained, general-purpose object detector combined with a hand-tracking model, and reasons about the *sequence* of events using hand-tuned, interpretable rules. This trades some theoretical accuracy ceiling for zero data-collection cost and full explainability — every decision the system makes can be traced back to a specific, human-readable rule.
- **Runs on hardware you already own.** The entire system runs on a standard consumer laptop and its built-in webcam. No dedicated edge hardware, no cloud GPU, no paid inference API.
- **Graceful degradation over hard failure.** Every optional signal in the pipeline (hand-landmark data, in particular) is designed to degrade gracefully to a simpler fallback rather than block the system entirely when it isn't available for a given frame.
- **Transparent, visible reasoning.** The live view doesn't just show a final incident flag — it continuously displays the system's intermediate state (what's currently held, how many drop candidates are in progress) so its behavior is observable and debuggable in real time, not a black box.

---

## System Architecture

```
                     ┌─────────────────────────┐
                     │   Live Camera (webcam)   │
                     └────────────┬────────────┘
                                  │  raw frames
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │         YOLO26 Detection + ByteTrack Tracking      │
        │   (persons + litter-relevant object classes)       │
        └───────────────────────┬─────────────────────────┘
                                  │  tracked Objects
                    ┌────────────┴────────────┐
                    ▼                          ▼
        ┌───────────────────────┐   ┌───────────────────────┐
        │  MediaPipe HandLandmarker │   │  Rule-Based Event      │
        │  (async LIVE_STREAM mode) │──▶│  State Machine          │
        └───────────────────────┘   │  (hold → drop → settle │
                                     │   → person exits)       │
                                     └───────────┬───────────┘
                                                  │  confirmed incident
                                                  ▼
                                     ┌───────────────────────┐
                                     │  Live Overlay (OpenCV)  │
                                     │  + Desktop Notification │
                                     │       (plyer)           │
                                     └───────────────────────┘
```

`main.py` runs this entire loop directly in the foreground: each frame is captured, detected, tracked, checked against the event state machine, and rendered with a full diagnostic overlay in a single `cv2.imshow` window — with no background threading or separate review UI. When an incident is confirmed, the frame is annotated on-screen, the incident is printed to the console, and a desktop toast notification is raised via `plyer`.

---

## Detection Pipeline, Stage by Stage

### 1. Object Detection & Tracking

Every frame is passed through a YOLO26 detector, restricted via a curated class filter to persons and a specific set of litter-relevant object categories (bottles, cups, bowls, food containers, and related items) rather than the full 80-class COCO vocabulary. The detector is exported to ONNX and INT8-quantized for faster CPU inference.

Detection alone is not sufficient — the system needs to reason about the *same* person and the *same* object across many consecutive frames, not just isolated per-frame detections. Ultralytics' built-in `.track()` method (backed by ByteTrack) assigns a persistent tracker ID to every detected person and object, which every downstream stage relies on, and which `main.py` displays directly on-screen next to each bounding box.

### 2. Hand Landmark Detection

Bounding-box overlap alone is a coarse signal for "is this person holding this object" — a person's torso box can easily overlap nearby litter they are simply walking past, without ever touching it. To get a tighter signal, the system runs Google's MediaPipe HandLandmarker in asynchronous `LIVE_STREAM` mode alongside the detector, extracting wrist keypoints for each detected hand and matching them to the correct tracked person by spatial containment. `main.py` draws every detected wrist point directly on the live feed, making it possible to visually confirm whether hand-tracking data is actually arriving on any given frame.

Because this result arrives asynchronously and is not guaranteed to be available on every single frame (a hand may be out of frame, occluded, or the result may simply not have arrived yet for the current timestamp), the system is explicitly designed so that **missing wrist data is a routine, expected condition — not an error.** Whenever wrist data isn't available for a given person on a given frame, the holding-detection logic falls back to bounding-box IoU automatically. This fallback is a structural part of the design, exercised on a meaningful fraction of frames in normal operation, not a rare edge case.

### 3. Holding State Machine

Holding is not decided from a single frame. Each (person, object) pair is tracked through an explicit state machine with **symmetric debouncing** in both directions:

- **Entering "held":** the pair must be confirmed as touching (via wrist proximity or bounding-box IoU) for several consecutive frames before the object is considered genuinely held — a single coincidental frame of contact while walking past an object does not count.
- **Leaving "held":** once confirmed as held, the pair must be confirmed as *not* touching for several consecutive frames before being treated as released — preventing a single noisy frame (a tracker jitter, a brief hand movement, momentary bounding-box misalignment) from ending a real hold prematurely.
- **Occlusion tolerance:** if the object is not detected at all for a short run of frames while it is being held — the most common real-world case being a hand partially or fully covering the very object it is holding — the hold state is *paused*, not reset. Progress toward the hold or release thresholds is preserved across the gap, up to a configurable grace period, after which the state is finally discarded if the object still hasn't reappeared.

Held objects are highlighted with a distinct on-screen color and an explicit `[HELD]` label in real time, directly reflecting this internal state.

### 4. Drop Confirmation

Once a hold is confirmed released, the system opens a drop candidate for that (person, object) pair and tracks the object's motion over the following frames:

- **Sustained, normalized descent.** Rather than reacting to a single frame's pixel delta (vulnerable to tracker noise), downward motion is averaged over a short window of recent positions and normalized against the object's own bounding-box size, keeping the check scale-invariant regardless of how close or far the object is from the camera.
- **Edge exclusion.** A drop candidate that begins near the frame's border is discarded outright — object tracking is least reliable at frame boundaries, and this is exactly where coincidental separations (e.g., wind moving debris at the edge of the monitored area) are most likely to produce a false signal.
- **Settling, not just landing.** After a confirmed descent, the object must remain within a small positional radius of a fixed anchor point for a sustained number of frames before being treated as genuinely settled. This specifically distinguishes a real, physically at-rest object from something that is still moving — such as debris disturbed by wind, which keeps shifting and therefore never accumulates enough consecutive "stayed still" frames to be confirmed.
- **Person exit.** A drop is only finalized into a confirmed incident once the associated person has also been absent from the scene for a sustained period. This distinguishes an object someone has set down and abandoned from one they are still actively standing near and attending to.
- **Occlusion tolerance during settling.** The same grace-period tolerance used during the hold phase applies here — if the settled object briefly leaves the frame or is occluded, the drop candidate is not immediately discarded, only after the gap exceeds the configured tolerance.

The live status line at the bottom of the feed continuously reports how many drop candidates are currently in progress, independent of whether any have yet been confirmed as a full incident.

### 5. Incident Confirmation & Notification

The moment the full sequence completes, `main.py`:
- Draws a large **"INCIDENT TRIGGERED"** banner directly on the video frame.
- Prints the incident's person ID, object ID, class name, and confidence to the console.
- Raises a **desktop toast notification** (via `plyer`), reporting the detection confidence and timestamp, so the event is noticeable even if the video window isn't the active focus.

No incident is currently persisted to disk, saved as a clip, or queued for later review — `main.py` is a real-time observation and alerting tool, not a review/audit system.

---

## Project Structure

```
cleanstreets.ai/
├── src/
│   ├── core/
│   │   ├── config.py          # All tunable parameters, in one place
│   │   ├── types.py           # Shared, dependency-free type aliases
│   │   └── utils.py           # Shared data structures & helper functions
│   ├── camera/
│   │   └── capture.py         # Live camera frame source
│   └── models/
│       ├── objects.py         # Tracked-object data schema
│       ├── detector.py        # YOLO detection + tracking, ONNX export/cache
│       ├── pose_est.py        # MediaPipe hand-landmark estimation
│       └── events.py          # The rule-based event state machine
├── main.py                    # Entry point — live detection loop, on-screen overlay, desktop alerts
├── models/                    # Cached model weights (downloaded on first run)
├── data/
│   └── Demo.mp4                # Recorded demo walkthrough
├── requirements.txt
└── setup.cfg
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Object detection & tracking | YOLO26 (Ultralytics) → ONNX, INT8 | Real-time person/object detection with persistent tracking |
| Hand landmark detection | MediaPipe HandLandmarker | Wrist-proximity-based holding detection |
| Event logic | Pure Python, rule-based state machine | Interpretable, zero-training-data decision logic |
| Live display | OpenCV (`cv2.imshow`) | Real-time bounding-box, wrist-point, and status overlay |
| Desktop alerts | `plyer` | Cross-platform toast notification on incident confirmation |

---

## Setup & Installation

**1. Clone the repository and navigate to the project root.**

**2. Create and activate a virtual environment:**

```bash
python -m venv venv
```

On Windows (PowerShell):
```powershell
venv\Scripts\activate
```
> If PowerShell blocks this with a `running scripts is disabled on this system` error, either run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once, or activate via `venv\Scripts\activate.bat` instead, which is unaffected by this restriction.

On macOS/Linux:
```bash
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. First run.** On first launch, the YOLO detector is exported to ONNX and INT8-quantized, and MediaPipe's hand-landmarker model bundle is downloaded — both are cached locally afterward, so this cost is only paid once. This step requires an internet connection.

> Windows will likely prompt for **camera access permission** the first time the app opens the webcam — this must be granted for the live feed to function.

---

## Running the Application

```bash
python main.py
```

A window titled **"CleanStreets AI — Diagnostic Preview"** opens, showing the live camera feed with the full detection overlay. Press **`q`** with the window focused to quit.

---

## What You'll See

- **Green boxes** — detected people, labeled with tracker ID, class name, and confidence.
- **Orange boxes** — detected litter-relevant objects, not currently held.
- **Yellow boxes, labeled `[HELD]`** — objects the event state machine currently considers genuinely held by a tracked person.
- **Red dots** — individual detected wrist points, labeled by the tracker ID of the person they were matched to.
- **Status line** (bottom of frame) — live counts of objects detected, wrists currently tracked, objects currently held, and drop candidates currently in progress.
- **"INCIDENT TRIGGERED" banner** — appears the instant a full littering sequence is confirmed, alongside a console log line and a desktop notification.

---

## Configuration Reference

All tunable parameters live in `src/core/config.py`. A selection of the most significant ones:

| Parameter | Purpose |
|---|---|
| `MODEL_NAME` | Path to the YOLO weights used for detection |
| `CONFIDENCE`, `IOU` | Detection confidence and NMS IoU thresholds |
| `CLASSES` | The filtered set of COCO class IDs (persons + litter-relevant objects) the detector reports |
| `OVERLAP_THRESHOLD` | Minimum bounding-box IoU to count as "touching," used as the fallback when wrist data is unavailable |
| `WRIST_PROXIMITY_RATIO` | Wrist-to-object-center distance threshold, as a ratio of the object's own bounding-box diagonal |
| `MIN_HOLD_FRAMES` | Consecutive confirmed-touching frames required before an object is considered genuinely held |
| `RELEASE_CONFIRM_FRAMES` | Consecutive confirmed-separated frames required before a hold is considered genuinely released |
| `OCCLUSION_GRACE_FRAMES` | How many consecutive frames of missing detection are tolerated, during either holding or settling, before state is discarded |
| `EDGE_MARGIN_PERCENT` | Frame-border margin within which a new drop candidate is excluded outright |
| `DOWNWARD_WIDNOW_FRAMES` | Number of recent object positions averaged when computing descent |
| `DESCENT_THRESHOLD` | Minimum normalized average downward motion to count as a genuine descent |
| `STILLNESS_RADIUS_PERCENT` | Maximum positional drift (as a percentage of frame width) still counted as "settled" |
| `SETTLED_CONFIRMATION_FRAMES` | Consecutive settled frames required before an object is confirmed at rest |
| `PERSON_EXIT_FRAMES` | Consecutive frames the associated person must be absent before an incident is finalized |
| `CLASS_VOTE_FRAMES` | Size of the rolling window used for majority-vote class-label stabilization |
| `STALE_HISTORY_FRAMES` | How long an object's class history is retained after it's last seen, before being pruned |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Occurs when running a script from a subdirectory rather than the project root. Run `python main.py` from the project root itself.

**`ImportError: cannot import name 'X' from partially initialized module` (circular import)**
Occurs if two modules import from each other. Shared, dependency-free types should live in their own leaf module (`types.py`) that both sides import from, rather than importing from each other directly.

**Model export re-runs on every restart**
Ultralytics appends a suffix (e.g. `_int8`) to exported filenames when quantization is used. If the cache-check logic looks for the un-suffixed filename, it will never find a match and will silently re-export from scratch every run.

**`running scripts is disabled on this system` (PowerShell)**
A default PowerShell security restriction unrelated to the project itself — see the Setup section above for the fix.

**No desktop notification appears**
Confirm `plyer` is installed and that your OS's notification permissions allow Python/the terminal to raise toast notifications — this is an OS-level setting independent of the application code.

---

## Known Limitations

- **Stock, non-fine-tuned detector.** The system uses general-purpose, COCO-pretrained YOLO weights rather than a detector fine-tuned specifically on litter imagery. Detection accuracy on ambiguous, small, or heavily occluded objects is correspondingly limited. Fine-tuning on a litter-specific dataset (e.g., TACO) was investigated as a natural next step but was scoped out of this prototype in favor of validating the full pipeline first.
- **No persistence.** Incidents are surfaced live (on-screen banner, console log, desktop notification) but are not currently saved to disk, recorded as clips, or queued for later human review.
- **Single camera, single line of sight.** An object fully occluded from the only available camera angle cannot be tracked, regardless of software-level tolerance for brief occlusion.
- **Geometric, not learned, decision logic.** Holding and dropping are determined by tuned geometric thresholds rather than a model trained on labeled real-world footage. This is interpretable and requires no training data, but has a lower theoretical accuracy ceiling than a properly trained classifier would.
- **fps-dependent timing thresholds.** Several thresholds are expressed as consecutive-frame counts rather than real time durations, meaning their real-world meaning shifts if the actual achieved processing rate drifts from what they were tuned against.
- **No re-hold cancellation.** An object briefly picked back up (e.g., someone retrieving something they set down momentarily) is not explicitly distinguished from genuine abandonment except by the person-exit timer eventually elapsing.

---

## Related Work

This project's core approach — combining object detection with hand-keypoint proximity to determine when an item has been released, then applying rule-based temporal logic to confirm abandonment — has independent precedent in the literature. Kim & Cho (*Sensors*, 2022) describe a closely comparable architecture (YOLO for object detection, OpenPose for wrist-point extraction, DeepSORT for tracking, and Euclidean hand-to-object distance for dumping detection) for identifying illegal garbage dumping by pedestrians, reporting 97% accuracy. This project was developed independently, using MediaPipe in place of OpenPose and ByteTrack in place of DeepSORT, with additional emphasis on runtime robustness: symmetric hold/release debouncing, occlusion-tolerant state transitions, and explicit hardening against false positives such as wind-disturbed debris and frame-edge tracking instability.

More broadly, AI-based litter and illegal-dumping detection is an active area both commercially (deployed municipal systems exist internationally) and within India specifically, where several city-level government deployments and academic projects already exist. The problem domain itself is well-established; this project's specific contribution is a fully rule-based, training-data-free architecture designed for deployment on ordinary consumer hardware.

---

## Future Work

- **Incident persistence and review** — saving confirmed incidents as clips with structured metadata, and building a review interface, rather than only surfacing them live.
- **Fine-tuning the detector** on a litter-specific dataset (TACO, and similar public litter-detection datasets) to improve accuracy on ambiguous or small objects beyond what stock COCO weights provide.
- **Migrating fixed frame-count thresholds to real time durations**, deriving the actual frame counts from measured achieved fps at runtime, so timing behavior remains consistent regardless of processing-rate fluctuations.
- **Re-hold cancellation**, so an object picked back up before an incident is finalized cancels the drop candidate immediately rather than relying solely on the person-exit timer.
- **Background-subtraction cross-verification** as a secondary confidence signal for settled-object confirmation, complementing (not replacing) the existing bounding-box-based logic, since it does not depend on tracker identity continuity.
- **Edge deployment** to dedicated hardware (e.g., Raspberry Pi or NVIDIA Jetson class devices), extending the original project scope beyond the current laptop-based prototype, with ONNX export already in place to support this transition.

---

## Hardware Notes

The system was developed and demonstrated entirely on a standard consumer laptop (AMD Ryzen 5 5500U, 8GB RAM) using its built-in webcam, deliberately validating that the approach does not require dedicated or expensive hardware. Various upgrade paths — external global-shutter USB cameras, network/IP cameras, dedicated GPU or edge-AI compute — were evaluated for their impact on accuracy and are documented as the natural next steps for a production deployment, but none were required to build or demonstrate the working prototype.

---

## License

Apache 2.0