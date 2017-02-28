import skimage
from skimage import io 
from scipy.ndimage.filters import gaussian_filter
import numpy as np 
import math
from itertools import product, starmap
from scipy import signal
import os
import matplotlib.pyplot as plt


# Read in image and convert it from uint8 to float64
 
file = input( "Enter the name of the file ")

building = io.imread(file)
building = skimage.img_as_float(building)


# Remove the 3 channels. Convert channel to only have one channel: Luminance
building = skimage.color.rgb2grey(building)

# Smooth image by convolving it with 7x7 gaussian kernel
gaussian_kernel = np.array([[0.003765,	0.015019,	0.023792,	0.015019,	0.003765],
[0.015019,	0.059912,	0.094907,	0.059912,	0.015019],
[0.023792,	0.094907,	0.150342,	0.094907,	0.023792],
[0.015019,	0.059912,	0.094907,	0.059912,	0.015019],
[0.003765,	0.015019,	0.023792,	0.015019,	0.003765]], dtype=np.float)
filtered_image = signal.convolve2d(building, gaussian_kernel)

# Find the gradient of the smoothed image
x_gradient, y_gradient = np.gradient(filtered_image)

# At each pixel, compute the edge strength and edge orientation
edge_strengths = np.zeros(building.shape, dtype=float)
edge_orientations = np.zeros(building.shape, dtype=float)

pi = 3.1415926
for point, val in np.ndenumerate(building):
	# Formula for magnitue -> sqrt(a^2 + b^2)
	magnitude = np.sqrt(x_gradient[point]**2 + y_gradient[point]**2)
	edge_strengths[point] = magnitude

	# Formula for orientation -> arctan(y_gradient / x_gradient) ""CHECK IF I NEED TO WORRY ABOUT DIVIDE BY 0
	# plots points between -pi/2 and pi/2
	orientation = np.arctan(y_gradient[point] / x_gradient[point])
	edge_orientations[point] = orientation

print("magnitude and orientation calculated")


# Determine the D* matrix, check each value in edge_orientations and store the angle it's closest to (0, pi/4, pi/2, 3pi/4)
angles = [0, np.divide(pi, 4), np.divide(pi, 2), -1 * np.divide(pi,4), -1 *np.divide(pi,2)]
minIndex = 0
minDiff = 10
for point, val in np.ndenumerate(edge_orientations):
	# Iterate through the 4 options, choose the one that has the least angle difference. Assign the index to the edge_orientations array
	for angle in angles:
		if np.absolute(val - angle) < minDiff:
			minIndex = angles.index(angle)
			minDiff = np.absolute(val - angle)
	edge_orientations[point] = minIndex
	#print(edge_orientations[point])
	minDiff = 10
	minIndex=0

print("angle assignment done")

# Don't modify edge_strengths directly. Make a copy. 
edge_strengths_copy = np.copy(edge_strengths)


# Thin the edges by doing non-maximum supression
# If the strength of neighboring points along the current pixels
for row in range(1, edge_orientations.shape[0]-1):  # -----> Vertical edge
	for col in range(1, edge_orientations.shape[1]-1):
		if edge_orientations[(row,col)] == 2 or edge_orientations[(row,col)] == 4: # 0
			if (edge_strengths_copy[(row, col+1)] < edge_strengths_copy[(row,col)]):
				edge_strengths[(row, col+1)] = 0
			if (edge_strengths_copy[(row, col-1)] < edge_strengths_copy[(row,col)]):
				edge_strengths[(row, col-1)] = 0

		elif edge_orientations[(row,col)] == 3: # pi / 4
			if (edge_strengths_copy[(row-1, col+1)] < edge_strengths_copy[(row,col)]):
				edge_strengths[(row-1, col+1)] = 0
			if (edge_strengths_copy[(row+1, col-1)] < edge_strengths_copy[(row,col)]):
				edge_strengths[(row+1, col-1)] = 0

		elif edge_orientations[(row,col)] == 0: # or edge_orientations[(row, col)] == 4:  pi /2
			if (edge_strengths_copy[(row+1, col)] < edge_strengths_copy[(row,col)]):
				edge_strengths[(row+1, col)] = 0
			if (edge_strengths_copy[(row-1, col)] < edge_strengths_copy[(row,col)]):
				edge_strengths[(row-1, col)] = 0

		elif edge_orientations[(row,col)] == 1: # -pi/4
			if (edge_strengths_copy[(row-1, col-1)] < edge_strengths_copy[(row,col)]):
				edge_strengths[(row-1, col-1)] = 0
			if (edge_strengths_copy[(row+1, col+1)] < edge_strengths_copy[(row,col)]):
				edge_strengths[(row+1, col+1)] = 0

# check that the current pixel I am looking at is within the image array
def in_bounds(x, y):
	lower_bound = 0
	upper_x_bound = building.shape[0]
	upper_y_bound = building.shape[1]
	if (x < 0 or y < 0 or x >= upper_x_bound or y >= upper_y_bound):
		return False
	else:
		return True
	

# Thresholds determine how many edges will be detected. Weak edges are chained to strong edges
marked_points = np.zeros(building.shape) #Flower -> .015, .008
strong_edge_thresh = .02
weak_edge_thresh = .012

# Iterative dfs to chain weak edges pixels to strong edge pixels
stack = []
for x in range(building.shape[0]):
	for y in range(building.shape[1]):
		if (edge_strengths[(x,y)] >= strong_edge_thresh):
			stack.append((x,y))
		elif (edge_strengths[(x,y)] < weak_edge_thresh):
			marked_points[(x,y)] = 1
			building[(x,y)] = 0

		while len(stack) != 0:
			current_point = stack.pop()
			marked_points[current_point] = 1  # mark this point so that we don't come back to it
			# Some code I found to quickly get the neighbors of any point in a matrix
			cells = starmap(lambda a,b: (current_point[0]+a, current_point[1]+b), product((0,-1,+1), (0,-1,+1)))
			for point in cells:
				if in_bounds(point[0], point[1]) and edge_strengths[point] >= weak_edge_thresh and marked_points[point] == 0:
					building[point] = 1
					stack.append(point)

# If a point has not yet been marked, then it must be a weak edge that does not chain to a strong edge. Remove it. 
for x in range(building.shape[0]):
	for y in range(building.shape[1]):
		point = (x,y)
		if marked_points[point] == 0:
			building[point] = 0


plt.imshow(building,  cmap='gray')
plt.show()
















