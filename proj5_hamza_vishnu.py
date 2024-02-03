"""
This code defines the BiDirectionalRRTStarAPF class that implements the Improved Bi-Directional RRT* 
algorithm with Artificial Potential Field. It also contains a helper function to visualize the result.

The code first imports required libraries, sets some constants, and defines several functions for 
calculating forces, steering, and costs.

The main class, BiDirectionalRRTStarAPF, initializes the algorithm with dimensions, sampling 
strategy, start and end points, maximum samples, step size, obstacles, probability of random 
connection, and the number of vertices for rewiring.

The algorithm is run using the bidirectional() method, which iteratively adds vertices to the 
trees, tries to connect them, and swaps the trees when necessary. The method returns the 
shortest path found after the given number of samples is reached or when the probability of 
random connection is satisfied.

The plot() function is used to visualize the path, trees, and obstacles.

Finally, the main part of the code sets the dimensions, obstacles, and other parameters for the 
algorithm, runs it, and visualizes the result.

Note that this code requires the 'numpy', 'rtree', and 'matplotlib' libraries to be installed.
"""

import numpy as np
from rtree import index
import matplotlib.pyplot as plt
import time

# User input for variables
K = float(input("Enter the attractive force constant (K): "))
MU = float(input("Enter the repulsive force constant (MU): "))
RHO = float(input("Enter the obstacle radius of influence (RHO): "))
clearance = float(input("Enter the clearance value: "))
starting = input("Enter the starting point (x, y): ")
x, y = map(float, starting.split())
starting = (x, y)
goal = input("Enter the goal point (x, y): ")
x, y = map(float, goal.split())
goal = (x, y)
start_time = time.time()

def line_points(start, end, obstacles, r, k=None, mu=None, rho0=None):
    """
    Generates points along a line segment from the start point to the end point, considering the influence of obstacles.
    
    Args:
    start (tuple): The starting point coordinates (x, y) of the line segment.
    end (tuple): The end point coordinates (x, y) of the line segment.
    obstacles (list): A list of tuples, where each tuple represents the coordinates (xmin, ymin, xmax, ymax) of an obstacle.
    r (float): The distance between points on the line segment.
    k (float, optional): The gain for attractive force. Defaults to K.
    mu (float, optional): The gain for repulsive force. Defaults to MU.
    rho0 (float, optional): The threshold distance for obstacle repulsive force. Defaults to RHO.
    
    Yields:
    tuple: The next point on the line segment considering the influence of obstacles.
    """
    if k is None: k = K
    if mu is None: mu = MU
    if rho0 is None: rho0 = RHO
    d = costs(start, end)
    n_points = int(np.ceil(d / r))
    if n_points > 1:
        step = d / (n_points - 1)
        for i in range(n_points):
            yield steer(start, end, i * step, obstacles, k, mu, rho0)

def steer(start, goal, d, obstacles, k=None, mu=None, rho0=None):
    """
    Steers the path by creating a point along the line segment from the start point to the goal point, considering the influence of obstacles.
    
    Args:
    start (tuple): The starting point coordinates (x, y) of the line segment.
    goal (tuple): The end point coordinates (x, y) of the line segment.
    d (float): The distance along the line segment from the start point to the desired point.
    obstacles (list): A list of tuples, where each tuple represents the coordinates (xmin, ymin, xmax, ymax) of an obstacle.
    k (float, optional): The gain for attractive force. Defaults to K.
    mu (float, optional): The gain for repulsive force. Defaults to MU.
    rho0 (float, optional): The threshold distance for obstacle repulsive force. Defaults to RHO.
    
    Returns:
    tuple: The point on the line segment at the specified distance considering the influence of obstacles.
    """
    if k is None: k = K
    if mu is None: mu = MU
    if rho0 is None: rho0 = RHO
    v = np.array(goal) - np.array(start)
    u = v / np.sqrt(np.sum(v ** 2))
    potential_force = calculate_potential_force(start, goal, obstacles, k, mu, rho0)
    steer_point = np.array(start) + u * d + potential_force
    return tuple(steer_point)

def costs(a, b):
    """
    Calculates the Euclidean distance between two points.
    
    Args:
    a (tuple): The first point coordinates (x, y).
    b (tuple): The second point coordinates (x, y).
    
    Returns:
    float: The Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(b) - np.array(a))

def path_cost(E, a, b):
    """
    Calculates the cost of a path in a graph.
    
    Args:
    E (dict): A dictionary representing edges in the graph where keys are child nodes and values are parent nodes.
    a (tuple): The starting point coordinates (x, y) of the path.
    b (tuple): The end point coordinates (x, y) of the path.
    
    Returns:
    float: The total cost of the path from the start point to the end point.
    """
    cost = 0
    while not b == a:
        cost += costs(b, E[b])
        b = E[b]
    return cost

def calculate_repulsive_force(x, obstacles, mu=None, rho0=None):
    """
    Calculates the repulsive force experienced by a point due to the presence of obstacles.
    
    Args:
    x (tuple): The point coordinates (x, y) for which the repulsive force is to be calculated.
    obstacles (list): A list of tuples, where each tuple represents the coordinates (xmin, ymin, xmax, ymax) of an obstacle.
    mu (float, optional): The gain for repulsive force. Defaults to MU.
    rho0 (float, optional): The threshold distance for obstacle repulsive force. Defaults to RHO.
    
    Returns:
    numpy.ndarray: The repulsive force vector experienced by the point due to the presence of obstacles.
    """
    if mu is None: mu = MU
    if rho0 is None:    rho0 = RHO
    repulsive_force = np.zeros(2)
    for obs in obstacles:
        min_x = max(obs[0], min(x[0], obs[2]))
        min_y = max(obs[1], min(x[1], obs[3]))
        closest_point = np.array([min_x, min_y])
        distance = np.linalg.norm(x - closest_point)
        if distance <= rho0:
            direction = x - closest_point
            repulsive_force += mu / 2 * ((1 / distance) - (1 / rho0)) ** 2 * (direction / distance ** 3)

    return repulsive_force

def calculate_potential_force(start, end, obstacles, k=None, mu=None, rho0=None):
    """
    Calculate the potential force between a start and end point, taking into account attractive and repulsive forces.

    Args:
        start (array-like): The starting point as an array-like object.
        end (array-like): The end point as an array-like object.
        obstacles (array-like): A list of obstacles.
        k (float, optional): A scaling constant for the attractive force. Defaults to K.
        mu (float, optional): A scaling constant for the repulsive force. Defaults to MU.
        rho0 (float, optional): A parameter for the repulsive force calculation. Defaults to RHO.

    Returns:
        np.array: The total force acting on the object as a NumPy array.
    """
    if k is None: k = K
    if mu is None: mu = MU
    if rho0 is None: rho0 = RHO
    direction = np.array(end) - np.array(start)
    distance = np.sqrt(np.sum(direction ** 2))
    attractive_force = k * direction / distance
    repulsive_force = calculate_repulsive_force(start, obstacles, mu, rho0)
    total_force = attractive_force + repulsive_force
    return total_force

class Trees:
    def __init__(self, X):
        """
        Initialize a Trees object.

        Args:
            X (BiDirectionalRRTStarAPF): A BiDirectionalRRTStarAPF object.
        """
        p = index.Property()
        p.dimension = X.dimensions
        self.vertex = index.Index(interleaved=True, properties=p)
        self.count = 0
        self.edges = {}

class BiDirectionalRRTStarAPF:
    def __init__(self, dlen, Q, start, goal, max_samples, r, obstacles, prob=0.01, rcount=None):
        """
        Initialize a BiDirectionalRRTStarAPF object.

        Args:
            dlen (array-like): An array-like object representing the lengths of each dimension.
            Q (array-like): An array-like object containing steering parameters for the RRT* algorithm.
            start (array-like): The starting point as an array-like object.
            goal (array-like): The goal point as an array-like object.
            max_samples (int): The maximum number of samples to be taken.
            r (float): The radius of the circle around a point to be considered for rewiring.
            obstacles (array-like): A list of obstacles.
            prob (float, optional): The probability of early termination. Defaults to 0.01.
            rcount (int, optional): The number of nearest points to consider for rewiring. Defaults to None.
        """
        self.dimensions = len(dlen)
        self.dlen = dlen
        p = index.Property()    # R-tree properties
        p.dimension = self.dimensions   # Number of dimensions
        if obstacles is None:
            self.obs = index.Index(interleaved=True, properties=p)  # R-tree for obstacles
        else:
            self.obs = index.Index(self.create_obs(obstacles), interleaved=True, properties=p)  
        self.samples_taken = 0
        self.max_samples = max_samples
        self.Q = Q  # Steering parameters
        self.r = r  # Radius for rewiring
        self.prob = prob  # Probability of random connection
        self.start = start  # Start point
        self.goal = goal    # Goal point
        self.rcount = rcount if rcount is not None else 0   # Number of vertices to consider for rewiring
        self.bestc = float('inf')   # Best cost
        self.best = None    # Best path
        self.switched = False    # Flag to indicate if trees are swapped
        self.trees = []
        self.obstacles = obstacles
        self.create_t()
        
    def fr_obs(self, x):
        """
        Checks if a point is in collision with any obstacle.
            
        Args:
        x (tuple): The point coordinates (x, y) to be checked.
        """
        return self.obs.count(x) == 0
    
    def free_sam(self):
        """
        Samples a point until it is collision-free.
        """
        while True:
            s = tuple(np.random.uniform(self.dlen[:, 0], self.dlen[:, 1]))
            if self.fr_obs(s):
                return s

    def collision_check(self, start, end, obstacles, r):
        """
        Checks if a line segment is collision-free.
            
        Args:
        start (tuple): The starting point coordinates (x, y) of the line segment.
        end (tuple): The end point coordinates (x, y) of the line segment.
        obstacles (list): A list of tuples, where each tuple represents the coordinates (xmin, ymin, xmax, ymax) of an obstacle.
        r (float): The distance between points on the line segment.
        
        Returns:
        bool: True if the line segment is collision-free, False otherwise.
        """
        points = line_points(start, end, obstacles,r)
        return all(map(self.fr_obs, points))
    
    def create_obs(self,obstacles):
        """ 
        Generates obstacles for the R-tree.
        
        Args:
        obstacles (list): A list of tuples, where each tuple represents the coordinates (xmin, ymin, xmax, ymax) of an obstacle.
        
        Yields:
        tuple: A tuple containing the coordinates of the obstacle.
        """
        for obstacle in obstacles:
            yield (np.random.randint(0, 100000), obstacle, obstacle)

    def expansion_connect(self, a, b, point, list_nearby):
        """
        Connects two trees if the path between them is collision-free and the cost is less than the current best cost.

        Args:
        a (int): The index of the first tree.
        b (int): The index of the second tree.
        point (tuple): The point coordinates (x, y) to be connected.
        list_nearby (list): A list of tuples, where each tuple contains the cost and coordinates of a nearby point.
        
        Returns:
        bool: True if the trees are connected, False otherwise.
        """
        for nearest_cost, nearest_distance in list_nearby:
            new_cost = nearest_cost + path_cost(self.trees[a].edges, self.start, point)     # Calculate cost of path from start to point in tree a
            if new_cost < self.bestc and self.collision_check(nearest_distance, point, self.obstacles, self.r):
                self.trees[b].count += 1
                self.trees[b].edges[point] = nearest_distance
                self.bestc = new_cost
                reconstructed_a = self.backtracking(a, self.start, point) # Reconstruct path from start to point in tree a
                reconstructed_b = self.backtracking(b, self.goal, point)    # Reconstruct path from goal to point in tree b
                del reconstructed_b[-1] # Remove point from path in tree b
                reconstructed_b.reverse()   # Reverse path in tree b
                self.best = reconstructed_a + reconstructed_b   # Set best path
                break

    def switch(self):
        """
        Swaps the trees and the start and goal points.
        
        Returns:
        bool: True if the trees are swapped, False otherwise.
        """
        self.trees[0], self.trees[1] = self.trees[1], self.trees[0]
        self.start, self.goal = self.goal, self.start
        self.switched = not self.switched

    def unswap(self):
        """
        Unswaps the trees and the start and goal points.
        
        Returns:
        bool: True if the trees are unswapped, False otherwise.
        """
        if self.switched:
            self.switch()
        if self.best is not None and self.best[0] is not self.start:
            self.best.reverse()

    def near_points(self, tree, start, point):
        """
        Gets the nearby vertices of a point in a tree.
        
        Args:
        tree (int): The index of the tree.
        start (tuple): The starting point coordinates (x, y) of the path.
        point (tuple): The end point coordinates (x, y) of the path.
        
        Returns:
        list: A list of tuples, where each tuple contains the cost and coordinates of a nearby point.
        """
        nearest_distance = self.trees[tree].vertex.nearest(point, num_results=self.rcount_vertices(tree), objects="raw")
        list_nearby = [(path_cost(self.trees[tree].edges, start, nearest_distance) + costs(nearest_distance, point), nearest_distance) for nearest_distance in nearest_distance]
        list_nearby.sort(key=lambda x: x[0])
        return list_nearby

    def rewire(self, tree, point, list_nearby):
        """
        Rewires the tree if the path between the point and its nearby points is collision-free and the cost is less than the current cost.
        
        Args:
        tree (int): The index of the tree.
        point (tuple): The point coordinates (x, y) to be rewired.
        list_nearby (list): A list of tuples, where each tuple contains the cost and coordinates of a nearby point.
    
        """
        for _, nearest_distance in list_nearby:
            curr_cost = path_cost(self.trees[tree].edges, self.start, nearest_distance)
            tent_cost = path_cost(self.trees[tree].edges, self.start, point) + costs(point, nearest_distance)
            if tent_cost < curr_cost and self.collision_check(nearest_distance, point, self.obstacles, self.r):
                self.trees[tree].edges[nearest_distance] = point

    def valid_connect(self, tree, point, list_nearby):
        """
        Connects the point to the nearest point if the path between them is collision-free and the cost is less than the current best cost.
        
        Args:
        tree (int): The index of the tree.
        point (tuple): The point coordinates (x, y) to be connected.
        list_nearby (list): A list of tuples, where each tuple contains the cost and coordinates of a nearby point.edges.
        """
        for nearest_cost, nearest_distance in list_nearby:
            if nearest_cost + costs(nearest_distance, self.goal) < self.bestc and self.pointconnect(tree, nearest_distance, point):
                break

    def rcount_vertices(self, tree):
        """
        Gets the number of vertices to consider for rewiring.
        
        Args:
        tree (int): The index of the tree.
        
        Returns:
        int: The number of vertices to consider for rewiring.
        """
        if self.rcount is None:
            return self.trees[tree].count
        return min(self.trees[tree].count, self.rcount)

    def create_t(self):
        """
        Adds a new tree to the list of trees.
        """
        self.trees.append(Trees(self))

    def create_v(self, tree, v):
        """
        Adds a vertex to a tree.
        
        Args:
        tree (int): The index of the tree.
        v (tuple): The point coordinates (x, y) to be added.
        """
        self.trees[tree].vertex.insert(0, v + v, v)
        self.trees[tree].count += 1 
        self.samples_taken += 1 
    
    def new_connect(self, tree, q):
        """
        Generates a new point and connects it to the nearest point in a tree.
        
        Args:
        tree (int): The index of the tree.
        q (array-like): An array-like object containing steering parameters for the RRT* algorithm.
        
        Returns:
        tuple: The new point coordinates (x, y).
        """
        x_rand = self.free_sam()
        
        nearest_distanceest = next(self.trees[tree].vertex.nearest(x_rand, 1, objects="raw"))
        point = np.maximum(steer(nearest_distanceest, x_rand, q[0], self.obstacles), self.dlen[:, 0])
        point = tuple(np.minimum(steer(nearest_distanceest, x_rand, q[0], self.obstacles), self.dlen[:, 1]))
        if not self.trees[tree].vertex.count(point) == 0 or not self.fr_obs(point):
            return None
        self.samples_taken += 1
        return point
    
    def pointconnect(self, tree, x_a, x_b):
        """
        Connects a point to another point in a tree if the path between them is collision-free.
        
        Args:
        tree (int): The index of the tree.
        x_a (tuple): The first point coordinates (x, y) to be connected.
        x_b (tuple): The second point coordinates (x, y) to be connected.
        
        Returns:
        bool: True if the points are connected, False otherwise.
        """
        if self.trees[tree].vertex.count(x_b) == 0 and self.collision_check(x_a, x_b, self.obstacles, self.r):
            self.create_v(tree, x_b)
            self.trees[tree].edges[x_b] = x_a
            return True
        return False
        
    def backtracking(self, tree, start, goal):
        """
        Reconstructs the path from the start point to the goal point in a tree.
        
        Args:
        tree (int): The index of the tree.
        start (tuple): The starting point coordinates (x, y) of the path.
        goal (tuple): The end point coordinates (x, y) of the path.
        
        Returns:
        list: A list of tuples, where each tuple contains the coordinates of a point in the path.
        """
        path = [goal]
        current = goal
        if start == goal: return path
        while not self.trees[tree].edges[current] == start:
            path.append(self.trees[tree].edges[current])
            current = self.trees[tree].edges[current]
        path.append(start)
        path.reverse()
        return path
    
    def bidirectional(self):
        """
        Runs the Improved Bi-Directional RRT* algorithm.
        """
        self.create_v(0, self.start)  # Add start to tree 0
        self.trees[0].edges[self.start] = None  # Add edge from start to None
        self.create_t() # Add tree 1
        self.create_v(1, self.goal)   # Add goal to tree 1
        self.trees[1].edges[self.goal] = None   # Add edge from goal to None
        while True: 
            for q in self.Q:    # For each steering parameter
                for i in range(q[1]):
                    point = self.new_connect(0, q) # Generate new point and connect it to the nearest point in tree 0
                    if point is None:
                        continue    # If no new point is generated, continue
                    list_nearby = self.near_points(0, self.start, point) # Get nearby vertices of new point in tree 0
                    self.valid_connect(0, point, list_nearby)   # Connect new point to the nearest point in tree 0
                    if point in self.trees[0].edges:    # If new point is connected to a point in tree 0
                        self.rewire(0, point, list_nearby)   # Rewire tree 0
                        list_nearby = self.near_points(1, self.goal, point)  # Get nearby vertices of new point in tree 1
                        self.expansion_connect(0, 1, point, list_nearby) # Connect trees 0 and 1
                    if self.prob and np.random.random() < self.prob:  # If probability of random connection is satisfied
                        if self.best is not None: # If a path is found
                            print(f"Path Generated\nSamples Checked: {self.samples_taken}")
                            self.unswap()   # Unswap trees
                            return self.best  # Return path
                    if self.samples_taken >= self.max_samples:  # If maximum samples are taken
                        self.unswap()   # Unswap trees
                        if self.best is not None:
                            print(f"Path Generated\nSamples Checked: {self.samples_taken}")
                            return self.best
                        return self.best
            self.switch()

def plot(obstacles, path, trees, snode, goal):
    """
    Plot the RRT* tree, obstacles, and the final path.

    Args:
        obstacles (array-like): A list of obstacles.
        path (array-like): The final path as a list of points.
        trees (list): A list of Tree objects.
        snode (array-like): The starting point as an array-like object.
        goal (array-like): The goal point as an array-like object.
    """
    tree_color = ['red', 'cyan']
    fig, ax = plt.subplots()
    ax.set_title('Improved Bi-Directional RRT* with Artificial Potential Field')
    for i, tree in enumerate(trees):
            for snode, end in tree.edges.items():
                if end is not None:
                    ax.plot([snode[0], end[0]], [snode[1], end[1]], color=tree_color[i])
    for obs in obstacles:
            ax.add_patch(plt.Rectangle((obs[0], obs[1]), obs[2]-obs[0], obs[3]-obs[1], facecolor='black'))
    x, y = zip(*path)
    ax.plot(x, y, color='blue', linewidth=2)
    ax.scatter(goal[0], goal[1], color='green', s=20)
    plt.show()

dimensions = np.array([(0, 600), (0, 400)])
pixels = np.ones((400, 600), np.uint8) * 255
clearance = int(clearance / 10)

obstacles = [(140-clearance, 75-clearance, 170+clearance, 105+clearance),
                (140-clearance, 185-clearance, 170+clearance, 215+clearance),
                (140-clearance, 295-clearance, 170+clearance, 325+clearance),
                (290-clearance, 0, 320+clearance, 30+clearance),
                (290-clearance, 120-clearance, 320+clearance, 150+clearance),
                (290-clearance, 250-clearance, 320+clearance, 280+clearance),
                (290-clearance, 370-clearance, 320+clearance, 400),
                (440-clearance, 75-clearance, 470+clearance, 105+clearance),
                (440-clearance, 185-clearance, 470+clearance, 215+clearance),
                (440-clearance, 295-clearance, 470+clearance, 325+clearance)]

Q = np.array([(8, 4)])  
r, max_samples, rcount, prob = 1, 2000, 10, 0.1    
rrt = BiDirectionalRRTStarAPF(dimensions, Q, starting, goal, max_samples, r, obstacles, prob, rcount)
path = rrt.bidirectional()
if path is not None:    
    path = np.array(path).astype(int)
    plot(obstacles, path, rrt.trees, starting, goal)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
else: 
    print("No path found")