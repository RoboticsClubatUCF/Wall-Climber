import matplotlib.pylab as plt
from skimage.morphology import skeletonize, thin
import numpy as np
import cv2 as cv
from skimage.util import invert
from google.colab.patches import cv2_imshow


def spencer_transform(mask, img, pix_bound):
    dst = cv.cornerHarris(mask,2,3,0.04)
    dst = cv.dilate(dst,None)
    img[dst>0.01*dst.max()]=[0,0,255]
    xpix = []
    ypix = []
    
    # Gets pixel coordinates
    for row in range(pix_bound):
        for col in range(pix_bound):
            if list(img[row, col]) == [0, 0, 255]:
                xpix.append(row)
                ypix.append(col)
    x_min_id = xpix.index(min(xpix))
    x_max_id = xpix.index(max(xpix))
    y_min_id = ypix.index(min(ypix))
    y_max_id = ypix.index(max(ypix))
    
    corner1 = [xpix[x_min_id], ypix[x_min_id]] #small x
    corner2 = [xpix[x_max_id], ypix[x_max_id]] #big x
    corner3 = [xpix[y_min_id], ypix[y_min_id]] #small y
    corner4 = [xpix[y_max_id], ypix[y_max_id]] #big y

    # locates the center of the maze
    x_center = (corner1[0] + corner2[0] + corner3[0] + corner4[0]) / 4
    y_center = (corner1[1] + corner2[1] + corner3[1] + corner4[1]) / 4
    center = [x_center, y_center]
    corner_arr = [corner1, corner2, corner3, corner4]

    # reorders the corner_arr to the correct order for the transformation
    temp_arr = [[0,0], [0,0], [0,0], [0,0]]

    for i in range(4):
        if (corner_arr[i][0] < center[0] and corner_arr[i][1] < center[1]):
            temp_arr[0] = corner_arr[i]
        elif (corner_arr[i][0] < center[0] and corner_arr[i][1] > center[1]):
            temp_arr[1] = corner_arr[i]
        elif (corner_arr[i][0] > center[0] and corner_arr[i][1] > center[1]):
            temp_arr[2] = corner_arr[i]
        elif (corner_arr[i][0] > center[0] and corner_arr[i][1] < center[1]):
            temp_arr[3] = corner_arr[i]
    
    corner_arr = temp_arr
    
    '''
    # Flips the x and y of the corners, because for some crazy reason cv2 wants
    # them in reverse order.
    for i in range(4):
        temp = corner_arr[i][0]
        corner_arr[i][0] = corner_arr[i][1]
        corner_arr[i][1] = temp
    '''


    final_corner_arr = np.float32(corner_arr)

    out_corner1 = [0, 0] 
    out_corner2 = [0, pix_bound - 1]
    out_corner3 = [pix_bound - 1, pix_bound - 1]
    out_corner4 = [pix_bound - 1, 0] 
    new_corner_arr = [out_corner1, out_corner2, out_corner3, out_corner4]
    new_corner_arr = np.float32(new_corner_arr)


    # Verifies corner position (commented out for now because it's not needed)
    
    # top left corner (1)
    img = cv.circle(img, (corner_arr[0][0], corner_arr[0][1]), 4*8, (0, 0, 255), 1)
    # bottom left corner (2)
    img = cv.circle(img, (corner_arr[1][0], corner_arr[1][1]), 4*8, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[1][0], corner_arr[1][1]), 8*8, (0, 0, 255), 1)
    # bottom right corner (3)
    img = cv.circle(img, (corner_arr[2][0], corner_arr[2][1]), 4*8, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[2][0], corner_arr[2][1]), 8*8, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[2][0], corner_arr[2][1]), 12*8, (0, 0, 255), 1)
    # top right corner (4)
    img = cv.circle(img, (corner_arr[3][0], corner_arr[3][1]), 4*8, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[3][0], corner_arr[3][1]), 8*8, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[3][0], corner_arr[3][1]), 12*8, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[3][0], corner_arr[3][1]), 16*8, (0, 0, 255), 1)
    cv2_imshow(img)

    # Warps the image based on the located corners
    matrix = cv.getPerspectiveTransform(final_corner_arr, new_corner_arr)
    warped_img = cv.warpPerspective(img, matrix, (pix_bound, pix_bound))

    # Returns the warped image
    return warped_img  



# Change file name accordingly to your image
img_name = 'maze5.jpg'

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
totalMask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)


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
cv.drawContours(totalMask, [largest], 0, (255), -1)

# This is the end mask of the area inside the maze, uncomment the line below to save this image
# cv.imwrite("afterMask.jpg", newMask)

# Save this new mask
mask = newMask

new_rgb_img = spencer_transform(totalMask, rgb_img, rgb_img.shape[0])
cv2_imshow(new_rgb_img)

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
