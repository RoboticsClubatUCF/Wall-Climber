# Computer Vision Files
### Author: Quinn Barber (Software Team Lead)

This directory contains all computer vision related files with the
Wall-Climber robot. The primary of which is contained within `imaging.py`.
This file is responsible for processing the image of the maze taken from
the camera, of which we will find the path the robot is required to take
from the start to the end of the maze. We use this to send directional
commands to the robot.

Processes/Algorithms used:
* Morphological Noise Reduction & Polygon Approximation
* Quad-tree approximation for pixel-to-inch conversion rates
* Breadth First Search for shortest path (Hand-made)
* Depth First Search for finding connected components (OpenCV)