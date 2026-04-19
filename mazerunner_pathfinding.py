"""
Project Mazerunner — Phase 2: Mapping & Pathfinding
=====================================================
Takes the boolean passage_map produced by Phase 1 and finds the
shortest path from the car's current position to the maze exit
using the A* algorithm.

Run standalone (simulated map) or import MazeSolver into your main loop.

Requirements:
    pip install numpy
    (No extra dependencies — A* is implemented from scratch)
"""

import numpy as np
import heapq
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class PathfindingConfig:
    # Grid dimensions — must match VisionConfig grid in Phase 1
    GRID_COLS = 20
    GRID_ROWS = 15

    # Allow diagonal movement between cells?
    # True = smoother paths, slightly more complex turns for the car
    # False = only up/down/left/right, easier for motor control
    ALLOW_DIAGONAL = False

    # Debug
    SHOW_DEBUG = True


# ---------------------------------------------------------------------------
# A* Pathfinder
# ---------------------------------------------------------------------------

class AStarPathfinder:
    """
    Finds the shortest path through a boolean grid using A*.

    True  = open cell (passable)
    False = wall cell (blocked)

    Usage:
        finder = AStarPathfinder()
        path = finder.find_path(passage_map, start=(0,0), goal=(19,14))
    """

    def __init__(self, config: PathfindingConfig = None):
        self.cfg = config or PathfindingConfig()

    def find_path(
        self,
        grid: np.ndarray,
        start: tuple,
        goal: tuple
    ) -> list:
        """
        Run A* from start to goal on the boolean grid.

        start, goal — (col, row) tuples in grid coordinates.

        Returns a list of (col, row) tuples from start to goal (inclusive),
        or an empty list if no path exists.
        """
        if not self._is_passable(grid, start):
            print(f"[Pathfinder] Start {start} is inside a wall.")
            return []
        if not self._is_passable(grid, goal):
            print(f"[Pathfinder] Goal {goal} is inside a wall.")
            return []

        # Each entry in the open heap: (f_score, g_score, position)
        open_heap = []
        heapq.heappush(open_heap, (0, 0, start))

        # Track where each cell was reached from
        came_from = {start: None}

        # g_score = actual cost from start to this cell
        g_score = {start: 0}

        while open_heap:
            _, g, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct_path(came_from, goal)

            # Skip if we've already found a better route to this cell
            if g > g_score.get(current, float("inf")):
                continue

            for neighbor in self._neighbors(grid, current):
                # Cost of moving to neighbor (diagonal = slightly more)
                move_cost = 1.4 if self._is_diagonal(current, neighbor) else 1.0
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    came_from[neighbor] = current
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

        # Open heap exhausted — no path found
        print("[Pathfinder] No path found between start and goal.")
        return []

    # --- Internal helpers ---

    def _heuristic(self, pos: tuple, goal: tuple) -> float:
        """
        Manhattan distance heuristic (for 4-directional movement).
        Switch to Euclidean if ALLOW_DIAGONAL is True.
        """
        if self.cfg.ALLOW_DIAGONAL:
            return ((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2) ** 0.5
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def _neighbors(self, grid: np.ndarray, pos: tuple) -> list:
        """
        Return all valid, passable neighbors of pos.
        4 directions by default; 8 if ALLOW_DIAGONAL is True.
        """
        col, row = pos
        directions = [
            ( 0, -1),  # up
            ( 0,  1),  # down
            (-1,  0),  # left
            ( 1,  0),  # right
        ]
        if self.cfg.ALLOW_DIAGONAL:
            directions += [(-1, -1), (1, -1), (-1, 1), (1, 1)]

        result = []
        for dc, dr in directions:
            neighbor = (col + dc, row + dr)
            if self._is_passable(grid, neighbor):
                result.append(neighbor)
        return result

    def _is_passable(self, grid: np.ndarray, pos: tuple) -> bool:
        """True if pos is inside the grid and not a wall."""
        col, row = pos
        rows, cols = grid.shape
        if not (0 <= col < cols and 0 <= row < rows):
            return False
        return bool(grid[row, col])

    def _is_diagonal(self, a: tuple, b: tuple) -> bool:
        return a[0] != b[0] and a[1] != b[1]

    def _reconstruct_path(self, came_from: dict, goal: tuple) -> list:
        """Walk back through came_from to build the path from start to goal."""
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Occupancy Grid Builder
# ---------------------------------------------------------------------------

class OccupancyGridBuilder:
    """
    Maintains a persistent map of the maze that accumulates across frames.

    Phase 1 gives a fresh passage_map every frame, but individual frames
    can have noise. This class merges multiple frames together so the map
    becomes more reliable over time.

    Usage:
        builder = OccupancyGridBuilder()
        builder.update(passage_map)   # call each frame
        stable_map = builder.get_stable_map()
    """

    def __init__(self, config: PathfindingConfig = None):
        self.cfg = config or PathfindingConfig()
        rows, cols = self.cfg.GRID_ROWS, self.cfg.GRID_COLS

        # Confidence counters — how many times each cell was seen as open
        self._open_count  = np.zeros((rows, cols), dtype=int)
        self._total_count = np.zeros((rows, cols), dtype=int)
        self._frame_count = 0

    def update(self, passage_map: np.ndarray):
        """Merge a new passage_map from Phase 1 into the accumulated map."""
        self._open_count  += passage_map.astype(int)
        self._total_count += 1
        self._frame_count += 1

    def get_stable_map(self, confidence: float = 0.6) -> np.ndarray:
        """
        Return a boolean grid where a cell is marked open only if it was
        seen as open in at least `confidence` fraction of frames.

        Higher confidence = fewer false passages (safer but slower to map).
        Lower confidence = maps faster but may include noisy open cells.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                self._total_count > 0,
                self._open_count / self._total_count,
                0.0
            )
        return ratio >= confidence

    def reset(self):
        """Clear accumulated map — call when entering a new maze."""
        self._open_count[:] = 0
        self._total_count[:] = 0
        self._frame_count = 0

    @property
    def frame_count(self):
        return self._frame_count


# ---------------------------------------------------------------------------
# Path smoother
# ---------------------------------------------------------------------------

def smooth_path(path: list) -> list:
    """
    Remove redundant waypoints from a path.

    A* on a grid produces paths with many small steps in the same direction.
    This collapses them into straight-line segments, reducing the number of
    motor commands the car needs to execute.

    Example: [(0,0),(1,0),(2,0),(3,0),(3,1)] → [(0,0),(3,0),(3,1)]
    """
    if len(path) < 3:
        return path

    smoothed = [path[0]]
    for i in range(1, len(path) - 1):
        prev = smoothed[-1]
        curr = path[i]
        next_ = path[i + 1]

        # Direction from prev to curr
        d1 = (curr[0] - prev[0], curr[1] - prev[1])
        # Direction from curr to next
        d2 = (next_[0] - curr[0], next_[1] - curr[1])

        # Only keep curr if direction changes (i.e. it's a turn)
        if d1 != d2:
            smoothed.append(curr)

    smoothed.append(path[-1])
    return smoothed


# ---------------------------------------------------------------------------
# Debug visualiser
# ---------------------------------------------------------------------------

def print_map(grid: np.ndarray, path: list = None, start: tuple = None, goal: tuple = None):
    """
    Print the grid to the terminal as ASCII art.
    Useful when you don't have a display (SSH into Pi, for example).

    Legend:
        .  = open cell
        #  = wall
        *  = path
        S  = start
        G  = goal
    """
    rows, cols = grid.shape
    path_set = set(path) if path else set()

    lines = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            pos = (c, r)
            if pos == start:
                row_chars.append("S")
            elif pos == goal:
                row_chars.append("G")
            elif pos in path_set:
                row_chars.append("*")
            elif grid[r, c]:
                row_chars.append(".")
            else:
                row_chars.append("#")
        lines.append(" ".join(row_chars))

    print("\n".join(lines))
    print()


# ---------------------------------------------------------------------------
# Simulated maze for standalone testing
# ---------------------------------------------------------------------------

def make_test_maze() -> np.ndarray:
    """
    Build a simple 20x15 test maze without needing a camera.
    True = open, False = wall.

    Use this to verify pathfinding logic on your laptop
    before connecting Phase 1.
    """
    grid = np.ones((15, 20), dtype=bool)  # start fully open

    # Outer walls
    grid[0,  :] = False
    grid[14, :] = False
    grid[:,  0] = False
    grid[:, 19] = False

    # Internal walls (horizontal barriers with gaps)
    grid[3,  2:10] = False
    grid[3,  12:18] = False

    grid[7,  1:8]  = False
    grid[7,  10:19] = False

    grid[11, 3:12] = False
    grid[11, 14:18] = False

    return grid


# ---------------------------------------------------------------------------
# Entry point — standalone test
# ---------------------------------------------------------------------------

def main():
    cfg     = PathfindingConfig()
    finder  = AStarPathfinder(cfg)
    builder = OccupancyGridBuilder(cfg)

    # Simulate feeding 10 identical frames into the occupancy builder
    test_map = make_test_maze()
    for _ in range(10):
        builder.update(test_map)

    stable_map = builder.get_stable_map(confidence=0.6)

    start = (1, 1)    # top-left open cell
    goal  = (18, 13)  # bottom-right open cell

    print("=== Mazerunner Phase 2 — Pathfinding ===\n")
    print("Maze map (before pathfinding):")
    print_map(stable_map, start=start, goal=goal)

    t0   = time.perf_counter()
    path = finder.find_path(stable_map, start, goal)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if path:
        smoothed = smooth_path(path)
        print(f"Path found in {elapsed_ms:.2f}ms")
        print(f"Raw waypoints:      {len(path)}")
        print(f"Smoothed waypoints: {len(smoothed)}  (motor commands needed)\n")
        print("Maze map (path shown as *):")
        print_map(stable_map, path=smoothed, start=start, goal=goal)
        print("Smoothed waypoints (col, row):")
        print(smoothed)
    else:
        print("No path found.")


if __name__ == "__main__":
    main()
