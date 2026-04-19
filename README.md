# Project Mazerunner

An autonomous maze-solving system for RC cars using camera-based computer vision and A* pathfinding. The car sees the maze through a camera, builds a map in real time, plans the shortest route to the exit, and drives itself through.

---

## How it works

```
Camera → Vision Pipeline → Occupancy Map → A* Pathfinder → Motor Commands
```

1. The camera captures a live video feed at 30fps
2. Each frame is processed through a 5-step vision pipeline to detect walls
3. Frames are accumulated into a stable occupancy grid (map of the maze)
4. A* finds the shortest path from the car's position to the exit
5. The path is smoothed into motor commands and sent to the RC car

---

## Project structure

```
mazerunner/
├── mazerunner_vision.py         # Phase 1 — camera capture & wall detection
├── mazerunner_pathfinding.py    # Phase 2 — occupancy mapping & A* pathfinding
├── mazerunner_main.py           # Integration — runs Phase 1 + Phase 2 together
└── README.md
```

> Phase 3 (motor control) and Phase 4 (hardware integration) are in progress.

---

## Requirements

### Software

- Python 3.8+
- OpenCV
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```

### Hardware (when deploying to RC car)

- Raspberry Pi 4 (2GB RAM minimum)
- Raspberry Pi Camera Module 3 or USB webcam
- RC car chassis with motor driver
- Arduino (for motor control in Phase 3)

---

## Quickstart

### Test on your laptop (no RC car needed)

All three files must be in the same folder. Run the integrated system:

```bash
python mazerunner_main.py
```

Point your laptop's webcam at a hand-drawn maze or a printed maze sheet. Three windows will open:

| Window | Shows |
|---|---|
| Mazerunner — integrated | Live feed with grid overlay and planned path |
| Edges | Canny edge detection output |
| Wall mask | Binary wall/floor mask |

To test pathfinding without a camera, run Phase 2 standalone — it uses a built-in simulated maze and prints the result as ASCII art:

```bash
python mazerunner_pathfinding.py
```

### Keyboard controls

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Reset occupancy map (use when moving to a new maze) |
| `g` | Set goal to centre of frame (quick testing) |

---

## Configuration

Each phase has its own config class at the top of the file. The most important settings:

### Vision (`VisionConfig` in `mazerunner_vision.py`)

| Setting | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Camera device index |
| `FRAME_WIDTH` | `320` | Capture width in pixels |
| `FRAME_HEIGHT` | `240` | Capture height in pixels |
| `BLUR_KERNEL` | `(5, 5)` | Gaussian blur size — increase if noisy |
| `CANNY_LOW` | `50` | Lower edge detection threshold |
| `CANNY_HIGH` | `150` | Upper edge detection threshold |
| `MIN_CONTOUR_AREA` | `500` | Ignore contours smaller than this |

### Pathfinding (`PathfindingConfig` in `mazerunner_pathfinding.py`)

| Setting | Default | Description |
|---|---|---|
| `GRID_COLS` | `20` | Horizontal grid cells |
| `GRID_ROWS` | `15` | Vertical grid cells |
| `ALLOW_DIAGONAL` | `False` | Allow diagonal movement |

### Integration (`IntegrationConfig` in `mazerunner_main.py`)

| Setting | Default | Description |
|---|---|---|
| `FRAMES_BEFORE_PATHFIND` | `10` | Frames to accumulate before first path plan |
| `REPLAN_INTERVAL` | `15` | Replan path every N frames |
| `START` | `(1, 1)` | Car start position in grid coords |
| `GOAL` | `(18, 13)` | Maze exit position in grid coords |

---

## Tuning tips

**Wall detection is picking up the floor texture**
Increase `BLUR_KERNEL` from `(5, 5)` to `(7, 7)` in `VisionConfig`.

**Wall detection breaks in different lighting**
Switch from Otsu thresholding to adaptive thresholding in `_threshold()`:
```python
binary = cv2.adaptiveThreshold(
    blurred, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2
)
```

**Pipeline is too slow on Raspberry Pi**
Reduce resolution to `160×120` in `VisionConfig`. Target latency is under 30ms per frame on a Pi 4.

**Path is too jittery / replanning too often**
Increase `REPLAN_INTERVAL` or `FRAMES_BEFORE_PATHFIND` in `IntegrationConfig`.

**No path found**
- Check `START` and `GOAL` are not inside wall cells (dark cells in the overlay)
- Lower `MIN_CONTOUR_AREA` if small walls are not being detected
- Check the wall mask window — if the floor and walls look inverted, flip `THRESH_BINARY_INV` to `THRESH_BINARY`

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1 — Vision & Perception | Done | Camera capture, wall detection, passage map |
| Phase 2 — Mapping & Pathfinding | Done | Occupancy grid, A* pathfinder, path smoother |
| Phase 3 — Motion Control | In progress | Motor driver, PID control, turn commands |
| Phase 4 — Hardware Integration | Planned | Mount camera, power management, latency tuning |
| Phase 5 — Testing & Optimization | Planned | Real maze trials, lighting robustness, speed tuning |

---

## How the vision pipeline works

```
Raw frame (BGR)
    → Grayscale         cv2.cvtColor()
    → Gaussian blur     cv2.GaussianBlur()       — removes noise
    → Threshold         cv2.threshold()          — black/white mask
    → Edge detection    cv2.Canny()              — finds wall boundaries
    → Contours          cv2.findContours()       — walls as shapes
    → Passage map       boolean grid             — open vs wall cells
```

## How A* pathfinding works

A* finds the shortest route through the passage map grid. It combines two values for every cell:

- `g` — the actual cost to reach this cell from the start
- `h` — the estimated cost from this cell to the goal (Manhattan distance)

It always expands the cell with the lowest `g + h` first, which guides it efficiently toward the goal without exploring the whole grid. The resulting path is then smoothed — consecutive steps in the same direction are collapsed into a single waypoint, reducing the number of motor commands needed.

---

## License

MIT License — free to use, modify, and distribute.
