import API
import sys
import heapq

# Constants for directions
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
direction_map = {NORTH: 'n', EAST: 'e', SOUTH: 's', WEST: 'w'}
current_orientation = NORTH

# Mouse position
x, y = 0, 0

# Maze dimensions
maze_width, maze_height = 6, 6
horizontal_walls = [[0] * maze_width for _ in range(maze_height + 1)]
vertical_walls = [[0] * (maze_width + 1) for _ in range(maze_height)]

# Tracking explored cells
explored_cells = set()

# Logging function
def log(string):
    sys.stderr.write("{}\n".format(string))
    sys.stderr.flush()

# Turn and move functions
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

def move_forward():
    global x, y, current_orientation
    log(f"Attempting to move forward. Current position: ({x}, {y}), orientation: {current_orientation}")
    API.moveForward()

    # Update position based on current orientation
    if current_orientation == NORTH:
        y += 1
    elif current_orientation == EAST:
        x += 1
    elif current_orientation == SOUTH:
        y -= 1
    elif current_orientation == WEST:
        x -= 1

    log(f"Moved forward to ({x}, {y}). Updated position: ({x}, {y})")
    log(f"New orientation: {current_orientation}")

# D* Lite Algorithm
class DStarLite:
    def __init__(self, start, goals, graph):
        self.start = start
        self.goals = goals
        self.graph = graph
        self.k_m = 0
        self.g = {v: float('inf') for v in graph.get_all_nodes()}
        self.rhs = {v: float('inf') for v in graph.get_all_nodes()}
        self.priority_queue = PriorityQueue()

        self.initialize(start, goals)

    def initialize(self, start, goals):
        log("Initializing D* Lite")
        for goal in goals:
            self.rhs[goal] = 0
            self.priority_queue.put(goal, self.calculate_key(goal))

        log("D* Lite initialization complete.")
        for goal in goals:
            log(f"g[{goal}] = {self.g[goal]}")
            log(f"rhs[{goal}] = {self.rhs[goal]}")
            log(f"Added {goal} to the priority queue with key {self.calculate_key(goal)}")
            API.setText(goal[0], goal[1], str(int(self.rhs[goal])))
            log(f"Post-init check - rhs[{goal}] = {self.rhs[goal]}")

    def calculate_key(self, vertex):
        g_rhs_min = min(self.g[vertex], self.rhs[vertex])
        return (g_rhs_min + self.heuristic(self.start, vertex) + self.k_m, g_rhs_min)

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def update_vertex(self, vertex):
        if vertex not in self.goals:
            # Calculate rhs as the minimum g value of the neighbors plus cost
            old_rhs = self.rhs[vertex]
            self.rhs[vertex] = min(
                [self.g[neighbor] + self.graph.cost(vertex, neighbor) for neighbor in self.graph.get_neighbors(vertex)]
            )
        
        # Remove the vertex from the priority queue if it exists
        self.priority_queue.remove(vertex)
        
        # If g != rhs, reinsert with updated priority
        if self.g[vertex] != self.rhs[vertex]:
            key = self.calculate_key(vertex)
            self.priority_queue.put(vertex, key)
            log(f"Updated vertex {vertex}: g = {self.g[vertex]}, rhs = {self.rhs[vertex]}")
            log(f"Added {vertex} to the priority queue with key {key}")
            log("---------------------------------------------------")
        else:
            log(f"Vertex {vertex} not added to queue: g = {self.g[vertex]}, rhs = {self.rhs[vertex]}")
        
        self.priority_queue.log_queue()  # Log the current state of the priority queue

        # Update the display with the new rhs value in the simulator
        if self.rhs[vertex] == float('inf'):
            API.setText(vertex[0], vertex[1], "inf")
        else:
            API.setText(vertex[0], vertex[1], str(int(self.rhs[vertex])))



    def compute_shortest_path(self):
        log("Starting to compute the shortest path...")
        while not self.priority_queue.empty() and (self.priority_queue.top_key() < self.calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]):
            u = self.priority_queue.get()
            log(f"Processing node {u} with g[{u}] = {self.g[u]} and rhs[{u}] = {self.rhs[u]}")
            self.priority_queue.log_queue() # Extract the node with the smallest key

            if self.g[u] > self.rhs[u]:  # Overconsistent case
                self.g[u] = self.rhs[u]  # Relax the g value
                for s in self.graph.get_neighbors(u): # Update neighbors
                    log("-----------------------------------------")
                    log(f"Before updating rhs[{s}] = {self.rhs[s]}")
                    self.update_vertex(s)
                    log(f"After updating rhs[{s}] = {self.rhs[s]}")
            else:  # Underconsistent case
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.graph.get_neighbors(u):
                    log(f"Before updating rhs[{s}] = {self.rhs[s]}")
                    self.update_vertex(s)
                    log(f"After updating rhs[{s}] = {self.rhs[s]}")

            log(f"Updated node {u} with new g[{u}] = {self.g[u]} and rhs[{u}] = {self.rhs[u]}")

        log("Shortest path computation complete.")

# Maze Graph Structure
class MazeGraph:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.graph = {}
        self.initialize_graph()
        self.initialize_full_connections()

    def initialize_graph(self):
        for x in range(self.width):
            for y in range(self.height):
                self.graph[(x, y)] = []

    def initialize_full_connections(self):
        for x in range(self.width):
            for y in range(self.height):
                if y < self.height - 1:
                    self.add_edge((x, y), (x, y + 1))  # Connect to north
                if y > 0:
                    self.add_edge((x, y), (x, y - 1))  # Connect to south
                if x < self.width - 1:
                    self.add_edge((x, y), (x + 1, y))  # Connect to east
                if x > 0:
                    self.add_edge((x, y), (x - 1, y))  # Connect to west

    def add_edge(self, u, v):
        if v not in self.graph[u]:
            self.graph[u].append(v)
        if u not in self.graph[v]:
            self.graph[v].append(u)

    def remove_edge(self, u, v):
        if v in self.graph[u]:
            self.graph[u].remove(v)
        if u in self.graph[v]:
            self.graph[v].remove(u)

    def get_neighbors(self, node):
        return self.graph[node]

    def is_connected(self, u, v):
        return v in self.graph[u]

    def cost(self, u, v):
        if not self.is_connected(u, v):
            return float('inf')
        return 1

    def get_all_nodes(self):
        return self.graph.keys()

# Priority Queue Implementation
class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

    def top_key(self):
        return self.elements[0][0] if self.elements else (float('inf'), float('inf'))

    def remove(self, item):
        for i in range(len(self.elements)):
            if self.elements[i][1] == item:
                del self.elements[i]
                heapq.heapify(self.elements)
                break

    def update(self, item, priority):
        self.remove(item)
        self.put(item, priority)

    def log_queue(self):
        log("Priority Queue state:")
        for priority, item in sorted(self.elements):
            log(f"  Vertex: {item}, Priority: {priority}")

# Scan and update walls
def scan_and_update_walls(x, y, dstar_lite):
    global explored_cells
    explored_cells.add((x, y))
    directions = [0, 1, 3]  # NORTH, EAST, WEST
    log(f"Scanning walls at ({x}, {y}) with orientation {current_orientation}")
    
    wall_detected = False
    
    for direction in directions:
        has_wall = check_wall(direction)
        log(f"Checking wall in direction {direction}: {has_wall}")
        
        nx, ny = None, None
        actual_direction = (current_orientation + direction) % 4
        
        if actual_direction == NORTH:  # NORTH
            nx, ny = x, y + 1
        elif actual_direction == EAST:  # EAST
            nx, ny = x + 1, y
        elif actual_direction == SOUTH:  # SOUTH
            nx, ny = x, y - 1
        elif actual_direction == WEST:  # WEST
            nx, ny = x - 1, y
            
        if valid_position(nx, ny, maze_width, maze_height):
            if has_wall:
                wall_detected = True
                dstar_lite.graph.remove_edge((x, y), (nx, ny))
                log(f"Removed edge between ({x}, {y}) and ({nx}, {ny}) due to wall.")
                API.setWall(x, y, direction_map[actual_direction])
            else:
                dstar_lite.graph.add_edge((x, y), (nx, ny))
                log(f"Added edge between ({x}, {y}) and ({nx}, {ny}) - no wall detected.")
    
    if wall_detected:
        log("Wall detected; recalculating shortest path.")
        dstar_lite.compute_shortest_path()

# # Check if there is a wall in the specified direction
# def check_wall(direction):
#     actual_direction = (current_orientation + direction) % 4
#     log(f"Checking wall. Current orientation: {current_orientation}, direction: {direction}, actual direction: {actual_direction}")
#     if actual_direction == NORTH:
#         return API.wallFront()
#     elif actual_direction == EAST:
#         return API.wallRight()
#     elif actual_direction == SOUTH:
#         return API.wallBack()  # Assuming this is how the API checks for a wall behind
#     elif actual_direction == WEST:
#         return API.wallLeft()
#     return False
def check_wall(direction):
    log(f"Checking wall. Current orientation: {current_orientation}, direction: {direction}")
    if direction == 0:  # Front
        return API.wallFront()
    elif direction == 1:  # Right
        return API.wallRight()
    elif direction == 3:  # Left
        return API.wallLeft()
# Validate if the given coordinates are within the maze bounds
def valid_position(x, y, width, height):
    return 0 <= x < width and 0 <= y < height

# Function to determine the next move based on the shortest path calculation
def move_to_next(x, y, graph, g):
    neighbors = graph.get_neighbors((x, y))
    log(f"Evaluating neighbors for move from ({x}, {y}): {neighbors}")

    lowest_g = float('inf')
    next_x, next_y = x, y

    for nx, ny in neighbors:
        log(f"Neighbor ({nx}, {ny}) has g value {g[(nx, ny)]}")
        if g[(nx, ny)] < lowest_g:
            lowest_g = g[(nx, ny)]
            next_x, next_y = nx, ny

    log(f"Next move determined: from ({x}, {y}) to ({next_x}, {next_y}) with g value {lowest_g}")
    
    # Determine the direction to move
    if next_x == x and next_y == y + 1:
        target_orientation = NORTH
    elif next_x == x + 1 and next_y == y:
        target_orientation = EAST
    elif next_x == x and next_y == y - 1:
        target_orientation = SOUTH
    elif next_x == x - 1 and next_y == y:
        target_orientation = WEST
    else:
        log(f"Unexpected move: from ({x}, {y}) to ({next_x}, {next_y}), which is not adjacent.")
        return x, y  # Abort move if something is wrong

    # Adjust orientation
    while current_orientation != target_orientation:
        if (target_orientation - current_orientation) % 4 == 1:
            turn_right()
        elif (target_orientation - current_orientation) % 4 == 3:
            turn_left()
        elif (target_orientation - current_orientation) % 4 == 2:
            turn_around()

    # Move forward after turning
    move_forward()

    return next_x, next_y

def show(g, rhs=None, highlight_cells=None):
    max_x = max([coord[0] for coord in g.keys()])
    max_y = max([coord[1] for coord in g.keys()])

    for y in range(max_y, -1, -1):  # Start from max_y down to 0 to display from top to bottom
        for x in range(max_x + 1):  # Start from 0 to max_x
            value = rhs[(x, y)] if rhs else g[(x, y)]
            if highlight_cells and (x, y) in highlight_cells:
                API.setColor(x, y, 'y')  # Highlight the cell in yellow
            if value == float('inf'):
                API.setText(x, y, 'inf')
            else:
                API.setText(x, y, str(int(value)))


def run_d_lite_6():
    try:
        global x, y, current_orientation

        width, height = 6, 6  # Fixed size for the maze
        start = (0, 0)
        goals = [(2, 2), (3, 2), (2, 3), (3, 3)]  # Multiple goal cells

        # Initialize the MazeGraph
        graph = MazeGraph(width, height)

        # Initialize the D* Lite algorithm
        dstar_lite = DStarLite(start, goals, graph)

        # Compute the initial shortest path
        log("Running D* Lite to calculate the initial shortest path.")
        dstar_lite.compute_shortest_path()

        # Main movement loop
        while (x, y) not in goals:
            log(f"Starting loop with mouse at ({x}, {y}), current orientation: {current_orientation}.")

            # Step 1: Scan for walls
            log(f"Mouse at ({x}, {y}), orientation: {current_orientation}. Scanning for walls.")
            scan_and_update_walls(x, y, dstar_lite)

            # Step 2: Recompute the shortest path if necessary
            if dstar_lite.priority_queue.top_key() < dstar_lite.calculate_key(start):
                log("Wall detected; recalculating shortest path.")
                dstar_lite.compute_shortest_path()

            # Step 3: Determine and execute the next move
            log(f"Determining and executing the next move from ({x}, {y}).")
            x, y = move_to_next(x, y, graph, dstar_lite.g)

            # Display the state of g and rhs values in the simulator
            show(dstar_lite.g, highlight_cells=[(x, y)])

        log("Mouse has reached the goal.")

    except Exception as e:
        log(f"Error during D* Lite execution: {e}")
        raise  # Re-raise the exception for further investigation


if __name__ == "__main__":
    try:
        log("Starting D* Lite algorithm...")
        run_d_lite_6()
        log("Finished running D* Lite algorithm.")
    except Exception as e:
        log(f"Error in main execution: {e}")
