import time
import API
import sys

def log(string):
    sys.stderr.write("{}\n".format(string))
    sys.stderr.flush()

DIRECTIONS = ["NORTH", "EAST", "SOUTH", "WEST"]
MOVE_MAP = {
    "NORTH": (0, 1),
    "EAST": (1, 0),
    "SOUTH": (0, -1),
    "WEST": (-1, 0)
}

def is_goal(x, y):
    goal_cells = [(7, 7), (8, 7), (7, 8), (8, 8)]
    return (x, y) in goal_cells

def dfs(x, y, current_direction, visited, move_counter, turn_counter, revisit_counter, collision_counter):
    log(f"Visiting ({x}, {y})")
    if (x, y) in visited:
        revisit_counter[0] += 1
    visited.add((x, y))
    move_counter[0] += 1

    if is_goal(x, y):
        log("Goal reached!")
        return True
    for _ in range(4):
        next_x = x + MOVE_MAP[current_direction][0]
        next_y = y + MOVE_MAP[current_direction][1]

        if (next_x, next_y) not in visited and not API.wallFront():
            log(f"Moving forward from ({x}, {y}) to ({next_x}, {next_y})")
            API.moveForward()
            move_counter[0] += 1
            if dfs(next_x, next_y, current_direction, visited, move_counter, turn_counter, revisit_counter, collision_counter):
                return True
        
            log("Backtracking")
            API.turnRight()
            API.turnRight()
            turn_counter[0] += 2
            if not API.wallFront():
                API.moveForward()
                move_counter[0] += 1
            API.turnRight()
            API.turnRight()
            turn_counter[0] += 2
        else:
            log("Collision detected!")
            collision_counter[0] += 1

        log("Turning right")
        API.turnRight()
        turn_counter[0] += 1
        current_direction = DIRECTIONS[(DIRECTIONS.index(current_direction) + 1) % 4]

    return False

def run_dfs():
    log("Running DFS...")
    API.setColor(0, 0, "G")
    API.setText(0, 0, "Start")

    start_x, start_y = 0, 0 
    log(f"Starting at ({start_x}, {start_y})")

    visited = set()
    visited.add((start_x, start_y))
    log(f"Visited set: {visited}")

    current_x, current_y = start_x, start_y
    current_direction = "NORTH"

    move_counter = [0]
    turn_counter = [0]
    revisit_counter = [0]
    collision_counter = [0]

    start_time = time.time()
    if dfs(start_x, start_y, current_direction, visited, move_counter, turn_counter, revisit_counter, collision_counter):
        log("DFS complete and goal reached!")
    else:
        log("DFS complete but goal not reachable.")
    end_time = time.time()

    # Fetching statistics from the simulator
    stats = {
        "total-distance": API.getStat("total-distance"),
        "total-turns": API.getStat("total-turns"),
        "best-run-distance": API.getStat("best-run-distance"),
        "best-run-turns": API.getStat("best-run-turns"),
        "current-run-distance": API.getStat("current-run-distance"),
        "current-run-turns": API.getStat("current-run-turns"),
        "total-effective-distance": API.getStat("total-effective-distance"),
        "best-run-effective-distance": API.getStat("best-run-effective-distance"),
        "current-run-effective-distance": API.getStat("current-run-effective-distance"),
        "score": API.getStat("score")
    }

    efficiency_ratio = move_counter[0] / stats["total-distance"] if stats["total-distance"] > 0 else 0

    for stat_name, stat_value in stats.items():
        log(f"{stat_name.replace('-', ' ').title()}: {stat_value}")

    log(f"Counted distance: {move_counter[0]}")
    log(f"Counted turns: {turn_counter[0]}")
    log(f"Revisited cells: {revisit_counter[0]}")
    log(f"Collision count: {collision_counter[0]}")
    log(f"Execution Time: {end_time - start_time} seconds")
    log(f"Efficiency Ratio: {efficiency_ratio}")

if __name__ == "__main__":
    run_dfs()