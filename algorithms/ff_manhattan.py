import API
import sys
from collections import deque
import heapq

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

current_orientation = NORTH

x, y = 0, 0

# Initialize wall arrays for a 16x16 maze
horizontal_walls = [[0] * 16 for _ in range(17)]
vertical_walls = [[0] * 17 for _ in range(16)]


# Counters for statistics
initial_run_cells = 0
return_run_cells = 0
final_run_cells = 0
explored_cells = set()


def log(string):
    sys.stderr.write("{}\n".format(string))
    sys.stderr.flush()


def log_stats():
    stats = [
        "total-distance", "total-turns", "best-run-distance", "best-run-turns",
        "current-run-distance", "current-run-turns", "total-effective-distance",
        "best-run-effective-distance", "current-run-effective-distance", "score"
    ]
    for stat in stats:
        value = API.getStat(stat)
        log(f"{stat}: {value}")


def turn_left():
    global current_orientation
    API.turnLeft()
    current_orientation = (current_orientation - 1) % 4
    log(f"Turned left. New orientation: {current_orientation}")

def turn_right():
    global current_orientation
    API.turnRight()
    current_orientation = (current_orientation + 1) % 4
    log(f"Turned right. New orientation: {current_orientation}")

def turn_around():
    global current_orientation
    API.turnLeft()
    API.turnLeft()
    current_orientation = (current_orientation + 2) % 4
    log(f"Turned around. New orientation: {current_orientation}")

def valid_position(x, y, width, height):
    return 0 <= x < width and 0 <= y < height

def calculate_manhattan_distances(goal_positions, maze_size):
    manhattan_distances = [[float('inf')] * maze_size for _ in range(maze_size)]
    for goal in goal_positions:
        for i in range(maze_size):
            for j in range(maze_size):
                manhattan_distances[i][j] = min(manhattan_distances[i][j], abs(goal[0] - i) + abs(goal[1] - j))
    return manhattan_distances

def flood_fill(maze, width, height, goal_cells, horizontal_walls, vertical_walls, explored_cells):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque(goal_cells)

    for gx, gy in goal_cells:
        maze[gy][gx] = 0

    while queue:
        x, y = queue.popleft()
        current_distance = maze[y][x]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if valid_position(nx, ny, width, height):
                if (dx == 0 and dy == 1 and not horizontal_walls[y + 1][x]) or \
                   (dx == 1 and dy == 0 and not vertical_walls[y][x + 1]) or \
                   (dx == 0 and dy == -1 and not horizontal_walls[y][x]) or \
                   (dx == -1 and dy == 0 and not vertical_walls[y][x]):
                    if maze[ny][nx] == float('inf'):
                        maze[ny][nx] = current_distance + 1
                        queue.append((nx, ny))
                        API.setText(nx, ny, str(int(maze[ny][nx])))


def set_virtual_walls_around_unexplored(width, height, horizontal_walls, vertical_walls, explored_cells):
    for x in range(width):
        for y in range(height):
            if (x, y) not in explored_cells:
                # Set virtual walls around unexplored cells
                if valid_position(x, y + 1, width, height) and (x, y + 1) in explored_cells:
                    horizontal_walls[y + 1][x] = 1
                    API.setWall(x, y, 'n')
                if valid_position(x + 1, y, width, height) and (x + 1, y) in explored_cells:
                    vertical_walls[y][x + 1] = 1
                    API.setWall(x, y, 'e')
                if valid_position(x, y - 1, width, height) and (x, y - 1) in explored_cells:
                    horizontal_walls[y][x] = 1
                    API.setWall(x, y, 's')
                if valid_position(x - 1, y, width, height) and (x - 1, y) in explored_cells:
                    vertical_walls[y][x] = 1
                    API.setWall(x, y, 'w')


def check_wall(direction):
    actual_direction = (current_orientation + direction) % 4
    if direction == 0:  # Front
        return API.wallFront()
    elif direction == 1:  # Right
        return API.wallRight()
    elif direction == 3:  # Left
        return API.wallLeft()

def update_walls(x, y, direction, has_wall, horizontal_walls, vertical_walls):
    actual_direction = (current_orientation + direction) % 4
    if actual_direction == 0:  # NORTH
        if has_wall:
            horizontal_walls[y + 1][x] = 1
            API.setWall(x, y, 'n')
            log(f"Added wall in cell ({x}, {y}, N)")
            if valid_position(x, y + 1, 16, 16):
                API.setWall(x, y + 1, 's')
                log(f"Added wall in cell ({x}, {y + 1}, S)")
    elif actual_direction == 1:  # EAST
        if has_wall:
            vertical_walls[y][x + 1] = 1
            API.setWall(x, y, 'e')
            log(f"Added wall in cell ({x}, {y}, E)")
            if valid_position(x + 1, y, 16, 16):
                API.setWall(x + 1, y, 'w')
                log(f"Added wall in cell ({x + 1}, {y}, W)")
    elif actual_direction == 2:  # SOUTH
        if has_wall:
            horizontal_walls[y][x] = 1
            API.setWall(x, y, 's')
            log(f"Added wall in cell ({x}, {y}, S)")
            if valid_position(x, y - 1, 16, 16):
                API.setWall(x, y - 1, 'n')
                log(f"Added wall in cell ({x}, {y - 1}, N)")
    elif actual_direction == 3:  # WEST
        if has_wall:
            vertical_walls[y][x] = 1
            API.setWall(x, y, 'w')
            log(f"Added wall in cell ({x}, {y}, W)")
            if valid_position(x - 1, y, 16, 16):
                API.setWall(x - 1, y, 'e')
                log(f"Added wall in cell ({x - 1}, {y}, E)")


def scan_and_update_walls(x, y, horizontal_walls, vertical_walls):
    global explored_cells
    explored_cells.add((x, y))
    directions = [0, 1, 3]  # NORTH, EAST, WEST
    log(f"Scanning walls at ({x}, {y}) with orientation {current_orientation}")
    for direction in directions:
        has_wall = check_wall(direction)
        log(f"Checking wall in direction {direction}: {has_wall}")
        update_walls(x, y, direction, has_wall, horizontal_walls, vertical_walls)
    log(f"Scanned walls at ({x}, {y}), orientation: {current_orientation}")


def can_move(x, y, direction, maze, horizontal_walls, vertical_walls):
    width, height = 16, 16  # Fixed size for the maze
    current_value = maze[y][x]
    
    if direction == 0:  # NORTH
        if valid_position(x, y + 1, width, height):
            can_move_north = horizontal_walls[y + 1][x] == 0 and maze[y + 1][x] < current_value
            log(f"Checking NORTH: can move to ({x}, {y + 1}): {can_move_north}")
            return can_move_north
        return False
    elif direction == 1:  # EAST
        if valid_position(x + 1, y, width, height):
            can_move_east = vertical_walls[y][x + 1] == 0 and maze[y][x + 1] < current_value
            log(f"Checking EAST: can move to ({x + 1}, {y}): {can_move_east}")
            return can_move_east
        return False
    elif direction == 2:  # SOUTH
        if valid_position(x, y - 1, width, height):
            can_move_south = horizontal_walls[y][x] == 0 and maze[y - 1][x] < current_value
            log(f"Checking SOUTH: can move to ({x}, {y - 1}): {can_move_south}")
            return can_move_south
        return False
    elif direction == 3:  # WEST
        if valid_position(x - 1, y, width, height):
            can_move_west = vertical_walls[y][x] == 0 and maze[y][x - 1] < current_value
            log(f"Checking WEST: can move to ({x - 1}, {y}): {can_move_west}")
            return can_move_west
        return False
    return False


def get_accessible_neighbors(x, y, maze, horizontal_walls, vertical_walls):
    neighbors = []
    width, height = len(maze[0]), len(maze)

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if valid_position(nx, ny, width, height):
            if (dx == 0 and dy == 1 and horizontal_walls[y + 1][x] == 0) or \
               (dx == 1 and dy == 0 and vertical_walls[y][x + 1] == 0) or \
               (dx == 0 and dy == -1 and horizontal_walls[y][x] == 0) or \
               (dx == -1 and dy == 0 and vertical_walls[y][x] == 0):
                neighbors.append((nx, ny))

    return neighbors


def recalculate_distances_from_goal(maze, horizontal_walls, vertical_walls, goal_cells):
    width, height = len(maze[0]), len(maze)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque(goal_cells)

    # Set all cells to infinity except goal cells
    for y in range(height):
        for x in range(width):
            maze[y][x] = float('inf')  # Initialize all cells to infinity

    for gx, gy in goal_cells:
        maze[gy][gx] = 0  # Initialize goal cells to 0

    while queue:
        x, y = queue.popleft()
        current_distance = maze[y][x]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if valid_position(nx, ny, width, height):
                if (dx == 0 and dy == 1 and horizontal_walls[y + 1][x] == 1):
                    continue  # There's a wall to the North
                if (dx == 1 and dy == 0 and vertical_walls[y][x + 1] == 1):
                    continue  # There's a wall to the East
                if (dx == 0 and dy == -1 and horizontal_walls[y][x] == 1):
                    continue  # There's a wall to the South
                if (dx == -1 and dy == 0 and vertical_walls[y][x] == 1):
                    continue  # There's a wall to the West

                if maze[ny][nx] == float('inf'):
                    maze[ny][nx] = current_distance + 1
                    queue.append((nx, ny))
                    API.setText(nx, ny, str(int(maze[ny][nx])))

    show(maze)


    
def print_flood_values(maze):
    for row in maze[::-1]:
        log(" ".join([str(cell) for cell in row]))

def move_to_lowest_neighbor(x, y, maze, horizontal_walls, vertical_walls, goal_cells, manhattan_distances, path=None, phase="initial"):
    global current_orientation, initial_run_cells, return_run_cells, final_run_cells  # Use the global orientation and counters
    neighbors = get_accessible_neighbors(x, y, maze, horizontal_walls, vertical_walls)
    pq = []

    log(f"Evaluating neighbors for move from ({x}, {y}): {neighbors}")
    for nx, ny in neighbors:
        flood_fill_value = maze[ny][nx]
        log(f"Neighbor ({nx}, {ny}) has flood fill value {flood_fill_value}")

    # Check if we are surrounded by higher flood fill values
    if all(maze[ny][nx] >= maze[y][x] for nx, ny in neighbors):
        log(f"Stuck at ({x}, {y}) with neighbors having higher flood fill values. Recalculating distances.")
        recalculate_distances_from_goal(maze, horizontal_walls, vertical_walls, goal_cells)
        for nx, ny in neighbors:
            flood_fill_value = maze[ny][nx]
            combined_value = flood_fill_value + manhattan_distances[ny][nx]
            log(f"Recalculated neighbor ({nx}, {ny}) combined value (flood fill: {flood_fill_value} + Manhattan: {manhattan_distances[ny][nx]}) = {combined_value}")
            heapq.heappush(pq, (combined_value, flood_fill_value, nx, ny))
    else:
        for nx, ny in neighbors:
            flood_fill_value = maze[ny][nx]
            combined_value = flood_fill_value + manhattan_distances[ny][nx]
            heapq.heappush(pq, (combined_value, flood_fill_value, nx, ny))

    if pq:
        _, _, next_x, next_y = heapq.heappop(pq)
        log(f"Moving from ({x}, {y}) to ({next_x}, {next_y}) with combined value {maze[next_y][next_x] + manhattan_distances[next_y][next_x]}")
    else:
        log(f"No valid moves available from ({x}, {y})")
        return x, y

    show(maze, highlight_cells=[(x, y), (next_x, next_y)])

    if phase == "initial":
        API.setColor(x, y, 'y')
    elif phase == "return":
        API.setColor(x, y, 'b')
    elif phase == "final":
        API.setColor(x, y, 'g')

    target_orientation = current_orientation
    if next_x == x and next_y == y + 1:  # Move North
        target_orientation = NORTH
    elif next_x == x + 1 and next_y == y:  # Move East
        target_orientation = EAST
    elif next_x == x and next_y == y - 1:  # Move South
        target_orientation = SOUTH
    elif next_x == x - 1 and next_y == y:  # Move West
        target_orientation = WEST

    while current_orientation != target_orientation:
        log(f"Current orientation: {current_orientation}, Target: {target_orientation}")
        if (target_orientation - current_orientation) % 4 == 1:
            turn_right()
        elif (target_orientation - current_orientation) % 4 == 3:
            turn_left()
        elif (target_orientation - current_orientation) % 4 == 2:
            turn_around()

    API.moveForward()
    if path is not None:
        path.append((next_x, next_y))
    if phase == "initial":
        initial_run_cells += 1
    elif phase == "return":
        return_run_cells += 1
    elif phase == "final":
        final_run_cells += 1

    x, y = next_x, next_y  # Update position

    log(f"Updated position after move: ({x}, {y}), orientation: {current_orientation}")
    scan_and_update_walls(x, y, horizontal_walls, vertical_walls)  # Scan walls after moving

    log("Flood fill values after move:")
    print_flood_values(maze)

    log("____________________")
    return x, y



def show(maze, highlight_cells=None):
    width, height = len(maze[0]), len(maze)
    for y in range(height):
        for x in range(width):
            if maze[y][x] == float('inf'):
                API.setText(x, y, 'inf')
            else:
                API.setText(x, y, str(int(maze[y][x])))

def run_ff_manhattan():
    global initial_run_cells, return_run_cells, final_run_cells, explored_cells

    width, height = 16, 16  # Fixed size for the maze
    maze = [[float('inf')] * width for _ in range(height)]

    # Initialize internal wall representations with boundary walls
    horizontal_walls = [[0] * 16 for _ in range(17)]
    vertical_walls = [[0] * 17 for _ in range(16)]
    for i in range(width):
        horizontal_walls[0][i] = 1
        horizontal_walls[height][i] = 1
        API.setWall(i, 0, 's')
        API.setWall(i, height - 1, 'n')
    for i in range(height):
        vertical_walls[i][0] = 1
        vertical_walls[i][width] = 1
        API.setWall(0, i, 'w')
        API.setWall(width - 1, i, 'e')

    log("Boundary walls initialized.")

    goal_cells = [(7, 7), (8, 7), (7, 8), (8, 8)]
    manhattan_distances = calculate_manhattan_distances(goal_cells, width)
    
    flood_fill(maze, width, height, goal_cells, horizontal_walls, vertical_walls, explored_cells)

    x, y = 0, 0
    while (x, y) not in goal_cells:
        log(f"Scanning and updating walls at ({x}, {y})")
        scan_and_update_walls(x, y, horizontal_walls, vertical_walls)
        
        log(f"Determining next move from ({x}, {y})")
        log(f"Current position: ({x}, {y}), orientation: {current_orientation}")
        x, y = move_to_lowest_neighbor(x, y, maze, horizontal_walls, vertical_walls, goal_cells, manhattan_distances, phase="initial")
        log(f"Moved to ({x}, {y})")

    log("Reached the goal. Re-flooding maze from the start point.")

    start_goal = [(0, 0)]
    flood_fill(maze, width, height, start_goal, horizontal_walls, vertical_walls, explored_cells)

    while (x, y) != (0, 0):
        log(f"Determining next move from ({x}, {y}) to return to start")
        x, y = move_to_lowest_neighbor(x, y, maze, horizontal_walls, vertical_walls, start_goal, manhattan_distances, phase="return")
        log(f"Moved to ({x}, {y})")

    log("Reached the start point. Preparing for the final run to the goal.")

    set_virtual_walls_around_unexplored(width, height, horizontal_walls, vertical_walls, explored_cells)

    flood_fill(maze, width, height, goal_cells, horizontal_walls, vertical_walls, explored_cells)

    path = [(0, 0)]
    while (x, y) not in goal_cells:
        log(f"Scanning and updating walls at ({x}, {y})")
        scan_and_update_walls(x, y, horizontal_walls, vertical_walls)
        
        log(f"Determining next move from ({x}, {y}) with path recording")
        log(f"Current position: ({x}, {y}), orientation: {current_orientation}")
        x, y = move_to_lowest_neighbor(x, y, maze, horizontal_walls, vertical_walls, goal_cells, manhattan_distances, path, phase="final")
        log(f"Moved to ({x}, {y})")

    log(f"Path to goal: {path}")

    log("Final distance map:")
    print_flood_values(maze)

    log_stats()
    log(f"Cells traversed in initial run: {initial_run_cells}")
    log(f"Cells traversed in return run: {return_run_cells}")
    log(f"Cells traversed in final run: {final_run_cells}")

if __name__ == "__main__":
    run_ff_manhattan()
