# Implementation of the Bi-Directional RRT* algorithm integrating Artificial Potential Field and Path Optimization



## ENPM661 - Planning For Autonomous Robots 

**PROJECT 5**

Hamza Shah Khan: 119483152 | hamzask@umd.edu

Vishnu Mandala: 119452608 | vishnum@umd.edu

## Description

This script contains an implementation of the Bidirectional Rapidly-exploring Random Trees Star (RRT*) algorithm integrated with artificial potential field and path optimization for robotic path planning. The goal of the algorithm is to find an optimal path between a start and a goal position, while avoiding obstacles and optimizing the path length.

## Features
* Bidirectional RRT* algorithm implementation for efficient path planning.
* Artificial Potential Field integration for smoother and more natural paths.
* Path optimization to remove redundant points and shorten the path.

### **Dependencies**

* python 3.11 (any version above 3 should work)
* Python running IDE (We used VS Code)

### **Libraries**
* NumPy
* Time
* Rtree
* Matplotlib

### **Contents**

* proj5_hamza_vishnu.py	
* proj5_hamza_vishnu.pdf
* README.md

### **Instructions**
1. Download the zip file and extract it
	
2. Install python and the required dependencies: 

	`pip install numpy rtree matplotlib`
	
3. Run the code:

	`$python3 proj5_hamza_vishnu.py`
	
4. Type in the Attractive Force Constant (K), Repulsive Force Constant (MU), Obstacle influence Radius (Rho), Clearance Value, Start Node(x y) and Goal Node(x y).
5. The optimal path will be displayed on the screen and the number of samples taken along with the runtime will be printed in the terminal.

### **Example Output**
Enter the attractive force constant (K): 60

Enter the repulsive force constant (MU): 20

Enter the obstacle radius of influence (RHO): 300

Enter the clearance value: 200

Enter the starting point (x, y): 20 10

Enter the goal point (x, y): 590 398

Path Generated

Samples Checked: 26

Time taken: 9.2

