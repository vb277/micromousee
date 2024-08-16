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

# Wall data structures to track where walls are in the maze
horizontal_walls = [[0] * maze_width for _ in range(maze_height + 1)]
vertical_walls = [[0] * (maze_width + 1) for _ in range(maze_height)]

# Set to track which cells have been explored
explored_cells = set()

# Function to log messages for debugging purposes
def log(string):
    sys.stderr.write("{}\n".format(string))
    sys.stderr.flush()

# Functions to control the mouse's movement and update its orientation

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

# Function to move the mouse forward and update its position in the maze
def move_forward():
    global x, y, current_orientation
    log(f"Attempting to move forward. Current position: ({x}, {y}), orientation: {current_orientation}")
    API.moveForward()

    # Update the mouse's position based on its current orientation
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

# Class to represent the D* Lite algorithm
class DStarLite:
    def __init__(self, start, goals, graph):
        # Initialize the D* Lite algorithm with a start point, goal points, and a graph of the maze
        self.start = start
        self.goals = goals
        self.graph = graph
        self.k_m = 0  # Km is a heuristic adjustment factor
        # Initialize g and rhs values for all nodes in the graph
        self.g = {v: float('inf') for v in graph.get_all_nodes()}
        self.rhs = {v: float('inf') for v in graph.get_all_nodes()}
        # Priority queue to manage nodes to be processed
        self.priority_queue = PriorityQueue()

        # Initialize the algorithm with the start and goal points
        self.initialize(start, goals)

    # Initialize the D* Lite algorithm
    def initialize(self, start, goals):
        log("Initializing D* Lite")
        for goal in goals:
            # Set the rhs value of the goal nodes to 0 (known cost to reach goal is zero)
            self.rhs[goal] = 0
            # Add the goal nodes to the priority queue with their calculated key
            self.priority_queue.put(goal, self.calculate_key(goal))

        log("D* Lite initialization complete.")
        for goal in goals:
            log(f"g[{goal}] = {self.g[goal]}")
            log(f"rhs[{goal}] = {self.rhs[goal]}")
            log(f"Added {goal} to the priority queue with key {self.calculate_key(goal)}")
            # Update the simulator with the rhs value at each goal location
            API.setText(goal[0], goal[1], str(int(self.rhs[goal])))
            log(f"Post-init check - rhs[{goal}] = {self.rhs[goal]}")

    # Calculate the priority key for a given vertex in the graph
    def calculate_key(self, vertex):
        g_rhs_min = min(self.g[vertex], self.rhs[vertex])
        return (g_rhs_min + self.heuristic(self.start, vertex) + self.k_m, g_rhs_min)

    # Heuristic function that uses Manhattan distance
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Update the cost estimate for a given vertex
    def update_vertex(self, vertex):
        if vertex not in self.goals:
            # Recalculate rhs for this vertex based on its neighbors
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
        
        # Log the current state of the priority queue
        self.priority_queue.log_queue()

        # Update the display with the new rhs value in the simulator
        if self.rhs[vertex] == float('inf'):
            API.setText(vertex[0], vertex[1], "inf")
        else:
            API.setText(vertex[0], vertex[1], str(int(self.rhs[vertex])))

    # Compute the shortest path using the D* Lite algorithm
    def compute_shortest_path(self):
        log("Starting to compute the shortest path...")
        # Process nodes in the priority queue
        while not self.priority_queue.empty() and (self.priority_queue.top_key() < self.calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]):
            u = self.priority_queue.get()
            log(f"Processing node {u} with g[{u}] = {self.g[u]} and rhs[{u}] = {self.rhs[u]}")
            self.priority_queue.log_queue()

            if self.g[u] > self.rhs[u]:  # Overconsistent case
                self.g[u] = self.rhs[u]  # Relax the g value
                for s in self.graph.get_neighbors(u):  # Update neighbors
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

# Class to represent the structure of the maze as a graph
class MazeGraph:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.graph = {}
        self.initialize_graph()
        self.initialize_full_connections()

    # Initialize the graph structure
    def initialize_graph(self):
        for x in range(self.width):
            for y in range(self.height):
                self.graph[(x, y)] = []

    # Create full connections between neighboring cells
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

    # Add an edge between two cells
    def add_edge(self, u, v):
        if v not in self.graph[u]:
            self.graph[u].append(v)
        if u not in self.graph[v]:
            self.graph[v].append(u)

    # Remove an edge between two cells
    def remove_edge(self, u, v):
        if v in self.graph[u]:
            self.graph[u].remove(v)
        if u in self.graph[v]:
            self.graph[v].remove(u)

    # Get neighbors of a cell
    def get_neighbors(self, node):
        return self.graph[node]

    # Check if two cells are connected
    def is_connected(self, u, v):
        return v in self.graph[u]

    # Get the cost of moving between two cells
    def cost(self, u, v):
        if not self.is_connected(u, v):
            return float('inf')
        return 1

    # Get all nodes in the graph
    def get_all_nodes(self):
        return self.graph.keys()

# Class to implement a priority queue for managing nodes in D* Lite
class PriorityQueue:
    def __init__(self):
        self.elements = []

    # Check if the queue is empty
    def empty(self):
        return len(self.elements) == 0

    # Insert an item with a given priority
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    # Get the item with the highest priority
    def get(self):
        return heapq.heappop(self.elements)[1]

    # Peek at the highest priority key
    def top_key(self):
        return self.elements[0][0] if self.elements else (float('inf'), float('inf'))

    # Remove an item from the queue
    def remove(self, item):
        for i in range(len(self.elements)):
            if self.elements[i][1] == item:
                del self.elements[i]
                heapq.heapify(self.elements)
                break

    # Update an item's priority
    def update(self, item, priority):
        self.remove(item)
        self.put(item, priority)

    # Log the current state of the priority queue
    def log_queue(self):
        log("Priority Queue state:")
        for priority, item in sorted(self.elements):
            log(f"  Vertex: {item}, Priority: {priority}")

# Function to scan for walls around the mouse and update the graph
def scan_and_update_walls(x, y, dstar_lite):
    global explored_cells
    explored_cells.add((x, y))
    directions = [0, 1, 3]  # NORTH, EAST, WEST
    log(f"Scanning walls at ({x}, {y}) with orientation {current_orientation}")
    
    wall_detected = False
    
    for direction in directions:
        # Check for walls in the specified direction
        has_wall = check_wall(direction)
        log(f"Checking wall in direction {direction}: {has_wall}")
        
        nx, ny = None, None
        actual_direction = (current_orientation + direction) % 4
        
        # Determine the position of the neighboring cell based on the actual direction
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
                log(f"Removed edge between ({x}, {y}) and ({nx, ny}) due to wall.")
                API.setWall(x, y, direction_map[actual_direction])
            else:
                dstar_lite.graph.add_edge((x, y), (nx, ny))
                log(f"Added edge between ({x}, {y}) and ({nx, ny}) - no wall detected.")
    
    # If a wall was detected, recompute the shortest path
    if wall_detected:
        log("Wall detected; recalculating shortest path.")
        dstar_lite.compute_shortest_path()

# Function to check if there is a wall in a specified direction
def check_wall(direction):
    log(f"Checking wall. Current orientation: {current_orientation}, direction: {direction}")
    if direction == 0:  # Front
        return API.wallFront()
    elif direction == 1:  # Right
        return API.wallRight()
    elif direction == 3:  # Left
        return API.wallLeft()

# Function to validate if a given coordinate is within the maze bounds
def valid_position(x, y, width, height):
    return 0 <= x < width and 0 <= y < height

# Function to determine the next move based on the shortest path calculation
def move_to_next(x, y, graph, g):
    neighbors = graph.get_neighbors((x, y))
    log(f"Evaluating neighbors for move from ({x}, {y}): {neighbors}")

    lowest_g = float('inf')
    next_x, next_y = x, y

    # Find the neighbor with the lowest g value
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

    # Adjust the mouse's orientation to face the target direction
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

# Function to display the current g and rhs values in the maze
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

# Main function to run the D* Lite algorithm and navigate the maze
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

# Entry point of the program
if __name__ == "__main__":
    try:
        log("Starting D* Lite algorithm...")
        run_d_lite_6()
        log("Finished running D* Lite algorithm.")
    except Exception as e:
        log(f"Error in main execution: {e}")
