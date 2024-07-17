import API
import sys
from collections import deque

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

current_orientation = NORTH

x, y = 0, 0

horizontal_walls = [[0] * 6 for _ in range(7)]
vertical_walls = [[0] * 7 for _ in range(6)]

def log(string):
    """
    Log messages for debugging.
    """
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

def flood_fill(maze, width, height):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    goal_cells = [(2, 2), (3, 2), (2, 3), (3, 3)]
    queue = deque(goal_cells)

    while queue:
        x, y = queue.popleft()
        current_distance = maze[y][x]
        
        if current_distance != float('inf'):
            API.setText(x, y, str(int(current_distance)))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if valid_position(nx, ny, width, height) and maze[ny][nx] == float('inf'):
                maze[ny][nx] = current_distance + 1
                queue.append((nx, ny))

    # Show the initial flood fill result
    show(maze)

def check_wall(direction):
    actual_direction = (current_orientation + direction) % 4
    if direction == 0:  # Front
        return API.wallFront()
    elif direction == 1:  # Right
        return API.wallRight()
    elif direction == 3:  # Left
        return API.wallLeft()

def update_walls(x, y, direction, has_wall, horizontal_walls, vertical_walls):
    """
    Update the internal map with the detected walls and print debug statements.
    
    Args:
    x (int): The x-coordinate of the current cell.
    y (int): The y-coordinate of the current cell.
    direction (int): The direction of the wall (0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST).
    has_wall (bool): True if there is a wall, False otherwise.
    horizontal_walls (list): 2D list representing horizontal walls.
    vertical_walls (list): 2D list representing vertical walls.
    """
    actual_direction = (current_orientation + direction) % 4
    if actual_direction == 0:  # NORTH
        if has_wall:
            horizontal_walls[y + 1][x] = 1
            API.setWall(x, y, 'n')
            log(f"Added wall in cell ({x}, {y}, N)")
            if valid_position(x, y + 1, 6, 6):
                API.setWall(x, y + 1, 's')
                log(f"Added wall in cell ({x}, {y + 1}, S)")
    elif actual_direction == 1:  # EAST
        if has_wall:
            vertical_walls[y][x + 1] = 1
            API.setWall(x, y, 'e')
            log(f"Added wall in cell ({x}, {y}, E)")
            if valid_position(x + 1, y, 6, 6):
                API.setWall(x + 1, y, 'w')
                log(f"Added wall in cell ({x + 1}, {y}, W)")
    elif actual_direction == 2:  # SOUTH
        if has_wall:
            horizontal_walls[y][x] = 1
            API.setWall(x, y, 's')
            log(f"Added wall in cell ({x}, {y}, S)")
            if valid_position(x, y - 1, 6, 6):
                API.setWall(x, y - 1, 'n')
                log(f"Added wall in cell ({x}, {y - 1}, N)")
    elif actual_direction == 3:  # WEST
        if has_wall:
            vertical_walls[y][x] = 1
            API.setWall(x, y, 'w')
            log(f"Added wall in cell ({x}, {y}, W)")
            if valid_position(x - 1, y, 6, 6):
                API.setWall(x - 1, y, 'e')
                log(f"Added wall in cell ({x - 1}, {y}, E)")

def scan_and_update_walls(x, y, horizontal_walls, vertical_walls):
    directions = [0, 1, 3]  # NORTH, EAST, WEST
    log(f"Scanning walls at ({x}, {y}) with orientation {current_orientation}")
    for direction in directions:
        has_wall = check_wall(direction)
        log(f"Checking wall in direction {direction}: {has_wall}")
        update_walls(x, y, direction, has_wall, horizontal_walls, vertical_walls)
    log(f"Scanned walls at ({x}, {y}), orientation: {current_orientation}")

def can_move(x, y, direction, maze, horizontal_walls, vertical_walls):
    width, height = 6, 6  # Fixed size for the maze
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
    """
    Get accessible neighboring cells from the current position using the can_move function.
    
    Args:
    x (int): The x-coordinate of the current cell.
    y (int): The y-coordinate of the current cell.
    maze (list): 2D list representing the maze distances.
    horizontal_walls (list): 2D list representing horizontal walls.
    vertical_walls (list): 2D list representing vertical walls.
    
    Returns:
    list: A list of accessible neighboring cells as (x, y) tuples.
    """
    neighbors = []
    
    for direction in range(4):
        if can_move(x, y, direction, maze, horizontal_walls, vertical_walls):
            if direction == 0:  # NORTH
                neighbors.append((x, y + 1))
            elif direction == 1:  # EAST
                neighbors.append((x + 1, y))
            elif direction == 2:  # SOUTH
                neighbors.append((x, y - 1))
            elif direction == 3:  # WEST
                neighbors.append((x - 1, y))
    
    return neighbors


def move_to_lowest_neighbor(x, y, maze, horizontal_walls, vertical_walls, goal_cells):
    global current_orientation  # Use the global orientation
    neighbors = get_accessible_neighbors(x, y, maze, horizontal_walls, vertical_walls)
    lowest_value = float('inf')
    next_x, next_y = x, y

    log(f"Evaluating neighbors for move from ({x}, {y}): {neighbors}")
    for nx, ny in neighbors:
        log(f"Neighbor ({nx}, {ny}) has value {maze[ny][nx]}")
        if maze[ny][nx] < lowest_value:
            lowest_value = maze[ny][nx]
            next_x, next_y = nx, ny

    if lowest_value >= maze[y][x]:
        log(f"Stuck at ({x}, {y}). Recalculating distances.")
        recalculate_distances_from_goal(maze, horizontal_walls, vertical_walls, goal_cells)
        neighbors = get_accessible_neighbors(x, y, maze, horizontal_walls, vertical_walls)  # Re-evaluate neighbors after recalculating distances
        lowest_value = float('inf')
        for nx, ny in neighbors:
            log(f"Neighbor ({nx}, {ny}) has value {maze[ny][nx]}")
            if maze[ny][nx] < lowest_value:
                lowest_value = maze[ny][nx]
                next_x, next_y = nx, ny

    log(f"Moving from ({x}, {y}) to ({next_x}, {next_y}) with value {lowest_value}")
    show(maze, highlight_cells=[(x, y), (next_x, next_y)])

    # Determine the direction to move based on next_x and next_y
    target_orientation = current_orientation
    if next_x == x and next_y == y + 1:  # Move North
        target_orientation = NORTH
    elif next_x == x + 1 and next_y == y:  # Move East
        target_orientation = EAST
    elif next_x == x and next_y == y - 1:  # Move South
        target_orientation = SOUTH
    elif next_x == x - 1 and next_y == y:  # Move West
        target_orientation = WEST

    # Optimize turn to the target orientation
    while current_orientation != target_orientation:
        log(f"Current orientation: {current_orientation}, Target: {target_orientation}")
        # Determine shortest turn direction
        if (target_orientation - current_orientation) % 4 == 1:
            turn_right()
        elif (target_orientation - current_orientation) % 4 == 3:
            turn_left()
        elif (target_orientation - current_orientation) % 4 == 2:
            turn_around()

    API.moveForward()
    if next_x == x:
        y = next_y
    else:
        x = next_x

    log(f"Updated position after move: ({x}, {y}), orientation: {current_orientation}")
    log("____________________")
    return x, y



def recalculate_distances_from_goal(maze, horizontal_walls, vertical_walls, goal_cells):
    width, height = len(maze[0]), len(maze)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque(goal_cells)

    # Set all cells to infinity except goal cells
    for y in range(height):
        for x in range(width):
            if (x, y) not in goal_cells:
                maze[y][x] = float('inf')

    for gx, gy in goal_cells:
        maze[gy][gx] = 0

    while queue:
        x, y = queue.popleft()
        current_distance = maze[y][x]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if valid_position(nx, ny, width, height):
                if (dx, dy) == (0, 1) and horizontal_walls[y + 1][x] == 1:
                    continue  # There's a wall to the North
                if (dx, dy) == (1, 0) and vertical_walls[y][x + 1] == 1:
                    continue  # There's a wall to the East
                if (dx, dy) == (0, -1) and horizontal_walls[y][x] == 1:
                    continue  # There's a wall to the South
                if (dx, dy) == (-1, 0) and vertical_walls[y][x] == 1:
                    continue  # There's a wall to the West

                if maze[ny][nx] == float('inf'):
                    maze[ny][nx] = current_distance + 1
                    queue.append((nx, ny))
                    API.setText(nx, ny, str(int(maze[ny][nx])))

    show(maze)


def show(maze, highlight_cells=None):
    """
    Update the simulator display with the current distance values.
    Optionally highlight specific cells.

    Args:
    maze (list): The 2D list representing the maze distances.
    highlight_cells (list): List of (x, y) tuples to highlight. Default is None.
    """
    width, height = len(maze[0]), len(maze)
    for y in range(height):
        for x in range(width):
            # Update the distance value in the simulator
            API.setText(x, y, str(int(maze[y][x])))

def run_flood_fill_6():
    width, height = 6, 6  # Fixed size for the maze
    maze = [[float('inf')] * width for _ in range(height)]
    
    # Initialize internal wall representations with boundary walls
    horizontal_walls = [[0] * 6 for _ in range(7)]
    vertical_walls = [[0] * 7 for _ in range(6)]
    for i in range(width):
        horizontal_walls[0][i] = 1
        horizontal_walls[height][i] = 1
        API.setWall(i, 0, 's')  # Highlight bottom boundary wall
        API.setWall(i, height - 1, 'n')  # Highlight top boundary wall
    for i in range(height):
        vertical_walls[i][0] = 1
        vertical_walls[i][width] = 1
        API.setWall(0, i, 'w')  # Highlight left boundary wall
        API.setWall(width - 1, i, 'e')  # Highlight right boundary wall

    log("Boundary walls initialized.")

    goal_cells = [(2, 2), (3, 2), (2, 3), (3, 3)]
    for gx, gy in goal_cells:
        maze[gy][gx] = 0

    flood_fill(maze, width, height)

    x, y = 0, 0
    while (x, y) not in goal_cells:
        log(f"Scanning and updating walls at ({x}, {y})")
        scan_and_update_walls(x, y, horizontal_walls, vertical_walls)
        
        log(f"Determining next move from ({x}, {y})")
        log(f"Current position: ({x}, {y}), orientation: {current_orientation}")
        x, y = move_to_lowest_neighbor(x, y, maze, horizontal_walls, vertical_walls, goal_cells)
        log(f"Moved to ({x}, {y})")

    log("Final distance map:")
    for row in maze[::-1]:
        log(" ".join([str(cell) for cell in row]))

    # Log stats at the end of the run
    log_stats()

if __name__ == "__main__":
    run_flood_fill_6()
