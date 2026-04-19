"""
Project Mazerunner — Phase 1: Vision & Perception Pipeline
==========================================================
Captures frames from a Pi Camera / USB cam, processes them through
a grayscale → blur → threshold → edge → contour pipeline, and outputs
a binary passage map ready for Phase 2 pathfinding.

Requirements:
    pip install opencv-python numpy

On Raspberry Pi with Pi Camera Module:
    pip install picamera2
"""

import cv2
import numpy as np
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class VisionConfig:
    # Camera
    CAMERA_INDEX = 0          # 0 for Pi Camera or first USB cam
    FRAME_WIDTH  = 320
    FRAME_HEIGHT = 240
    FPS_TARGET   = 30

    # Preprocessing
    BLUR_KERNEL  = (5, 5)     # Must be odd. Larger = smoother but slower.
    BLUR_SIGMA   = 0           # 0 lets OpenCV auto-calculate from kernel size

    # Thresholding — Otsu auto-picks the cutoff, so these rarely need tuning
    THRESH_MAX   = 255
    THRESH_FLAGS = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

    # Canny edge detection — tune these for your lighting
    CANNY_LOW    = 50
    CANNY_HIGH   = 150

    # Contour filtering
    MIN_CONTOUR_AREA = 500    # Ignore tiny blobs (noise, dust, reflections)

    # Debug — set True to show live windows during development
    SHOW_DEBUG_WINDOWS = True


# ---------------------------------------------------------------------------
# Vision Pipeline
# ---------------------------------------------------------------------------

class MazeVisionPipeline:
    """
    Full per-frame processing pipeline.

    Usage:
        pipeline = MazeVisionPipeline()
        passage_map = pipeline.process(frame)
    """

    def __init__(self, config: VisionConfig = None):
        self.cfg = config or VisionConfig()

    def process(self, frame: np.ndarray) -> dict:
        """
        Run a raw BGR frame through the full pipeline.

        Returns a dict with intermediate images and the final passage map,
        useful for debugging and for feeding into Phase 2.
        """
        t0 = time.perf_counter()

        gray      = self._to_grayscale(frame)
        blurred   = self._blur(gray)
        binary    = self._threshold(blurred)
        edges     = self._detect_edges(blurred)
        contours  = self._extract_contours(edges)
        wall_mask = self._build_wall_mask(contours, frame.shape)
        passage_map = self._build_passage_map(wall_mask)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "gray":        gray,
            "blurred":     blurred,
            "binary":      binary,
            "edges":       edges,
            "contours":    contours,
            "wall_mask":   wall_mask,
            "passage_map": passage_map,  # 2D bool array: True = open passage
            "latency_ms":  elapsed_ms,
        }

    # --- Pipeline steps ---

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to grayscale. Halves the data to process."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _blur(self, gray: np.ndarray) -> np.ndarray:
        """
        Gaussian blur to suppress noise and surface texture.
        Larger kernel = smoother walls but less sharp edges.
        Start with (5,5); increase to (7,7) if you see speckle noise.
        """
        return cv2.GaussianBlur(gray, self.cfg.BLUR_KERNEL, self.cfg.BLUR_SIGMA)

    def _threshold(self, blurred: np.ndarray) -> np.ndarray:
        """
        Otsu thresholding — automatically finds the best black/white cutoff.
        THRESH_BINARY_INV makes walls white (255) and open floor black (0).
        Switch to cv2.adaptiveThreshold() if lighting is very uneven.
        """
        _, binary = cv2.threshold(
            blurred, 0, self.cfg.THRESH_MAX, self.cfg.THRESH_FLAGS
        )
        return binary

    def _detect_edges(self, blurred: np.ndarray) -> np.ndarray:
        """
        Canny edge detector — finds sharp brightness transitions (wall edges).
        Tune CANNY_LOW and CANNY_HIGH in VisionConfig for your environment.
        Rule of thumb: HIGH is ~2-3x LOW.
        """
        return cv2.Canny(blurred, self.cfg.CANNY_LOW, self.cfg.CANNY_HIGH)

    def _extract_contours(self, edges: np.ndarray) -> list:
        """
        Find closed contours from edge image.
        RETR_EXTERNAL grabs only outermost contours (good for simple mazes).
        Use RETR_TREE if your maze has nested wall structures.
        """
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Filter out tiny contours (noise, reflections, dust)
        return [c for c in contours if cv2.contourArea(c) > self.cfg.MIN_CONTOUR_AREA]

    def _build_wall_mask(self, contours: list, frame_shape: tuple) -> np.ndarray:
        """
        Draw filled contours onto a blank mask.
        White pixels (255) = walls. Black pixels (0) = open space.
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
        return mask

    def _build_passage_map(self, wall_mask: np.ndarray) -> np.ndarray:
        """
        Convert pixel mask to a coarse boolean grid for pathfinding.
        Each cell in the grid represents a small region of the frame.
        True = open passage, False = wall.

        The grid resolution (20x15 for a 320x240 frame) is a starting point.
        Increase for finer navigation, decrease for speed.
        """
        grid_cols, grid_rows = 20, 15
        cell_w = wall_mask.shape[1] // grid_cols
        cell_h = wall_mask.shape[0] // grid_rows

        grid = np.zeros((grid_rows, grid_cols), dtype=bool)
        for r in range(grid_rows):
            for c in range(grid_cols):
                cell = wall_mask[
                    r * cell_h:(r + 1) * cell_h,
                    c * cell_w:(c + 1) * cell_w
                ]
                # Cell is open if fewer than 20% of its pixels are walls
                wall_ratio = np.count_nonzero(cell) / cell.size
                grid[r, c] = wall_ratio < 0.2

        return grid


# ---------------------------------------------------------------------------
# Debug visualiser
# ---------------------------------------------------------------------------

def draw_debug_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    """
    Overlay contours and latency onto the original frame for live debugging.
    Press 'q' to quit the debug window.
    """
    vis = frame.copy()
    cv2.drawContours(vis, result["contours"], -1, (0, 255, 0), 2)
    label = f"Pipeline: {result['latency_ms']:.1f}ms"
    cv2.putText(vis, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


# ---------------------------------------------------------------------------
# Camera capture loop
# ---------------------------------------------------------------------------

class MazeCamera:
    """Thin wrapper around cv2.VideoCapture with Pi-friendly defaults."""

    def __init__(self, config: VisionConfig = None):
        self.cfg = config or VisionConfig()
        self.cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          self.cfg.FPS_TARGET)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self.cfg.CAMERA_INDEX}. "
                "Check that the camera is connected and not in use."
            )

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def release(self):
        self.cap.release()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg      = VisionConfig()
    camera   = MazeCamera(cfg)
    pipeline = MazeVisionPipeline(cfg)

    print("Mazerunner Vision Pipeline running. Press 'q' to quit.")

    try:
        while True:
            frame  = camera.read()
            result = pipeline.process(frame)

            # Log latency — target is under 30ms on a Pi 4
            print(f"Latency: {result['latency_ms']:.1f}ms | "
                  f"Contours: {len(result['contours'])} | "
                  f"Passages: {result['passage_map'].sum()} open cells")

            if cfg.SHOW_DEBUG_WINDOWS:
                overlay = draw_debug_overlay(frame, result)
                cv2.imshow("Mazerunner — live",    overlay)
                cv2.imshow("Edges",                result["edges"])
                cv2.imshow("Wall mask",            result["wall_mask"])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


if __name__ == "__main__":
    main()
