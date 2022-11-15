import matplotlib.pylab as plt
from skimage.morphology import skeletonize, thin
import numpy as np
import cv2 as cv
from skimage.util import invert

# Change file name accordingly to your image
img_name = 'maze6.jpg'

rgb_img = cv.imread(img_name)

# Copy the image over so that we can mask for the start and end points
find_points_img = rgb_img
hsv_points = cv.cvtColor(find_points_img, cv.COLOR_BGR2HSV)

lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

points_mask = cv.inRange(hsv_points, lower_green, upper_green)

# Here is the original mask of the points, uncomment line below to save this image
# cv.imwrite("maskPoints.jpg", points_mask)

# This mask isn't perfect, so we must find the contours and remove the smaller bits and pieces that got through the mask
contours, hierarchy = cv.findContours(points_mask, mode = cv.RETR_LIST, method = cv.CHAIN_APPROX_SIMPLE)

# Create this empty image of the same size
newMask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)

# Now we go through all contours, and only add the start and end nodes (this is if the contour has a larger area)
# The bits and pieces will not be drawn onto this new mask
for i, cnt in enumerate(contours):
    if hierarchy[0][i][2] == -1:
        if cv.contourArea(cnt) > 1000:
            cv.drawContours(newMask, [cnt], 0, (255), -1)


# Here is the finished mask of the points, uncomment line below to save this image
# cv.imwrite("newMask.jpg", newMask)

# Now we must find the center of the start and end points based off of this mask
# We can do this with connected components in openCV using the line below
output = cv.connectedComponentsWithStats(newMask, 4, cv.CV_32S)
centroids = output[3]
stats = output[2]

# The centroids are the centers of every connected component
# However, this includes all black pixels being connected (which aren't the points)
# We can find if the area is too large using the stats of the components
# We can then find the component of the black pixels and delete them
for i in range(len(stats)):
    if stats[i][4] > 50000:
        centroids = np.delete(centroids, i, axis=0)

# This leaves us with only the centroid of the start and end nodes
# Ta-da!
x0, y0 = int(centroids[0][0]), int(centroids[0][1])
x1, y1 = int(centroids[1][0]), int(centroids[1][1])

print("Start and End Nodes found at the coordinates below\n")
print("Start: x = ", x0, " y = ", y0, "\n")
print("End: x = ", x1, " y = ", y1, "\n")

# Let us now go back to our original image and mask for the walls of the maze
# We do this given the lowest and highest blue values of the walls
low_blue = np.array([50, 50, 50])
high_blue = np.array([255, 255, 255])

hsv = cv.cvtColor(rgb_img, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, low_blue, high_blue)

# This is the original mask of the walls, uncomment the line below to save this image
# cv.imwrite("mask.jpg", mask)

# We now do the same process of getting rid of the small bits and pieces using contours
contours, hierarchy = cv.findContours(mask, mode = cv.RETR_LIST, method = cv.CHAIN_APPROX_SIMPLE)

# This is the empty image
newMask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)

# We find the largest and second-largest contours
# In this case, the second-largest contours is the total area inside the walls of the maze
# While the largest contour is the area of the entire maze itself
largest = contours[0]
secondLargest = contours[0]

# Go through all the contours and find this largest and second-largest value
for i, cnt in enumerate(contours):
    if hierarchy[0][i][2] == -1:
        if cv.contourArea(cnt) > cv.contourArea(largest):
            secondLargest = largest
            largest = cnt

# Draw the second largest to have the total area inside the maze
cv.drawContours(newMask, [secondLargest], 0, (255), -1)

# This is the end mask of the area inside the maze, uncomment the line below to save this image
# cv.imwrite("afterMask.jpg", newMask)

# Save this new mask
mask = newMask

# Go through and threshold for the black and white of this mask
if mask.shape.__len__() > 2:
    thr_img = (mask[:, :, 0] > np.max(mask[:, :, 0])/2)
else:
    thr_img = mask > np.max(mask)/2

# Skeletonize the inside of this
skeleton = skeletonize(thr_img)

# map of routes.
mapT = ~skeleton

_mapt = np.copy(mapT)

# searching for our end point and connect to the path.
# THIS VALUE MAY HAVE TO BE CHANGED
# IT IS THE RADIUS THAT YOU ARE ALLOWING FROM THE PATH TO THE START/END
# THIS COULD GLITCH OUT IF THE AREA INSIDE THE WALLS BY THE START/END IS TOO LARGE
# I.E. NO WALLS WITHIN RANGE OF START AND END
boxr = 30

# Just a little safety check, if the points are too near the edge, it will error.
if y1 < boxr: y1 = boxr
if x1 < boxr: x1 = boxr

cpys, cpxs = np.where(_mapt[y1 - boxr:y1 + boxr, x1 - boxr:x1 + boxr] == 0)
# calibrate points to main scale.
cpys += y1 - boxr
cpxs += x1 - boxr

# find the closest point of possible path end points
idx = np.argmin(np.sqrt((cpys - y1) ** 2 + (cpxs - x1) ** 2))
y, x = cpys[idx], cpxs[idx]

pts_x = [x]
pts_y = [y]
pts_c = [0]

# mesh of displacements.
xmesh, ymesh = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
ymesh = ymesh.reshape(-1)
xmesh = xmesh.reshape(-1)

dst = np.zeros(thr_img.shape)

# Breath first algorithm exploring a tree
while True:
    # update distance.
    idc = np.argmin(pts_c)
    ct = pts_c.pop(idc)
    x = pts_x.pop(idc)
    y = pts_y.pop(idc)
    # Search 3x3 neighbourhood for possible
    ys, xs = np.where(_mapt[y - 1:y + 2, x - 1:x + 2] == 0)
    # Invalidate these point from future searchers.
    _mapt[ys + y - 1, xs + x - 1] = ct
    _mapt[y, x] = 9999999
    # set the distance in the distance image.
    dst[ys + y - 1, xs + x - 1] = ct + 1
    # extend our list.s
    pts_x.extend(xs + x - 1)
    pts_y.extend(ys + y - 1)
    pts_c.extend([ct + 1] * xs.shape[0])
    # If we run of points.
    if not pts_x:
        break
    if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) < boxr:
        edx = x
        edy = y
        break

path_x = []
path_y = []

y = edy
x = edx
# Traces best path
while True:
    # x, y starting at the end goes all the way back
    nbh = dst[y - 1:y + 2, x - 1:x + 2]
    nbh[1, 1] = 9999999
    nbh[nbh == 0] = 9999999
    # If we reach a dead end
    if np.min(nbh) == 9999999:
        break
    idx = np.argmin(nbh)
    # find direction
    y += ymesh[idx]
    x += xmesh[idx]

    if np.sqrt((x - x1) ** 2 + (y - y1) ** 2) < boxr:
        print('Optimum route found.')
        break
    path_y.append(y)
    path_x.append(x)

plt.figure(figsize=(14,14))
plt.imshow(rgb_img)
plt.plot(path_x, path_y, 'r-', linewidth=5)
plt.show()