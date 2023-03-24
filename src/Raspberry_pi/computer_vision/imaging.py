
# Imaging python file, the purpose of this file is to take and process
# the image that we are using for the maze with the robot on it.

# We then will find the shortest path from where the robots start point is
# to where the robots end point is, using this we send directional commands
# to the robot in order to move it to this destination!

# Author: Quinn Barber

from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import math
import sys


def importImage(image_name):

    # Read in the image as an image file using CV
    originalImage = cv.imread(image_name)

    # Get hsv points of the image
    hsv_points = cv.cvtColor(originalImage, cv.COLOR_BGR2HSV)

    # Add these values to an array to return
    imageArray = []
    imageArray.append(originalImage)
    imageArray.append(hsv_points)

    return imageArray

def createMasks(hsv_points, originalImage):

    # Masks are created using the inRange function in CV with lower and upper values

    # Mask for green
    lower_green = np.array([0, 100, 50])
    upper_green = np.array([80, 255, 140])

    # Mask for red
    lower_red = np.array([0, 180,20])
    upper_red = np.array([10, 255, 255])

    # Mask for blue
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 255, 95])

    # Mask for orange
    lower_orange = np.array([0, 100, 150])
    upper_orange = np.array([80, 255, 255])

    # Creating the actual masks
    green_mask = cv.inRange(originalImage, lower_green, upper_green)
    red_mask = cv.inRange(hsv_points, lower_red, upper_red)
    blue_mask = cv.inRange(originalImage, lower_blue, upper_blue)
    orange_mask = cv.inRange(originalImage, lower_orange, upper_orange)

    # Set them into a dictionary and return them
    mask_dict = {'green': green_mask, 'red': red_mask, 'blue': blue_mask, 'orange': orange_mask}

    return mask_dict

def contourMasks(mask_dict, originalImage):

    # Removes noise from the masks (green, blue, and orange)

    # Find all contours of the green mask and initialize empty image file to draw updated mask into (updatedGreenMask)
    contours, hierarchy = cv.findContours(mask_dict['green'], mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    updatedGreenMask = np.zeros(originalImage.shape[:2], dtype=np.uint8)

    # Add only the contours we want into updatedGreenMask
    for i, contour in enumerate(contours):
        if hierarchy[0][i][2] == -1:
            if cv.contourArea(contour) > 1000:
                cv.drawContours(updatedGreenMask, [contour], 0, 255, -1)

    # Update the mask stored in the dictionary
    mask_dict['green'] = updatedGreenMask

    # Find contours of the blue mask and initialize empty image file to draw updated mask into (updatedBlueMask)
    contours, hierarchy = cv.findContours(mask_dict['blue'], mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    updatedBlueMask = np.zeros(originalImage.shape[:2], dtype=np.uint8)

    # For the blue mask, we are looking for the second-largest contour to add to our
    # new mask, this in every case is the area that is contained within the walls of
    # the maze, this is what we are running skeletonize on to find possible paths
    # throughout the maze

    # To do this we initialize the largest and secondLargest to the first in the array
    largest = contours[0]
    secondLargest = contours[0]

    # Update the values as we loop through the contours
    for i, contour in enumerate(contours):
        if hierarchy[0][i][2] == -1:
            if cv.contourArea(contour) > cv.contourArea(largest):
                secondLargest = largest
                largest = contour

    cv.drawContours(updatedBlueMask, [secondLargest], 0, 255, -1)

    # Update the mask stored in the dictionary
    mask_dict['blue'] = updatedBlueMask

    # We will now remove noise from the orange mask using the same method we used for
    # the green mask
    contours, hierarchy = cv.findContours(mask_dict['orange'], mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    updatedOrangeMask = np.zeros(originalImage.shape[:2], dtype=np.uint8)

    for i, contour in enumerate(contours):
        if hierarchy[0][1][2] == -1:
            if cv.contourArea(contour) > 1000:
                cv.drawContours(updatedOrangeMask, [contour], 0, 255, -1)

    # Update the mask stored in the dictionary
    mask_dict['orange'] = updatedOrangeMask

    # For the red mask, we are going to remove noise using a different method
    # in this case, we are applying a morphological operation that removes noise for us
    kernel = np.ones((5,5), np.uint8)
    thresh = cv.morphologyEx(mask_dict['red'], cv.MORPH_OPEN, kernel)

    # Let us get the contours of this updated threshold, as well as initialize our empty image file
    contours, hierarchy = cv.findContours(thresh, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    updatedRedMask = np.zeros(originalImage.shape[:2], dtype=np.uint8)

    # We are about to find all the polygons within this image, let us initialize an
    # array that will store these polygons points
    polygonArr = []


    # We will now loop through these contours and draw approximated polygon shapes
    # for each inch by inch square
    # We will also add them to our polygon array
    for i, contour in enumerate(contours):
        if hierarchy[0][i][2] == -1:
            approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
            cv.drawContours(updatedRedMask, [approx], 0, (255, 255, 255), 2)
            polygonArr.append(approx)

    # Update the mask stored in the dictionary
    mask_dict['red'] = updatedRedMask

    # Return our new masks as well as our polygon array!
    return mask_dict, polygonArr


def findCentroids(mask_dict):

    # This functions purpose is to find the centroids of our green and orange masks
    # effectively, we are finding the coordinates of where our robot is, and where
    # it wants to go to!

    # Our output is information on every connected component within this mask
    # this is done automatically through OpenCV! Even though I would love to
    # write the DFS for it myself!
    output = cv.connectedComponentsWithStats(mask_dict['green'], 4, cv.CV_32S)
    centroids = output[3]
    stats = output[2]

    # Every connected component also includes the black pixels, which are all connected
    # Let us find this (by finding the component that is very large) and delete it
    # as we do not need this.
    for i in range(len(stats)):
        if stats[i][4] > 50000:
            centroids = np.delete(centroids, i, axis=0)

    # This should leave us with only one component left, which is our robots starting point!
    # Green -> Start || Orange -> End
    # Ta-da!  <-- Arup Guha Tribute
    startX, startY = int(centroids[0][0]), int(centroids[0][1])

    # Let us repeat this process for our orange mask to find the end coordinates.
    output = cv.connectedComponentsWithStats(mask_dict['orange'], 4, cv.CV_32S)
    centroids = output[3]
    stats = output[2]

    for i in range(len(stats)):
        if stats[i][4] > 50000:
            centroids = np.delete(centroids, i, axis=0)

    # And here are our end coordinates!
    endX, endY = int(centroids[0][0]), int(centroids[0][1])

    # Let us return these values!
    return startX, startY, endX, endY


def findPaths(mask_dict):

    # Threshold this image to prep to send it to skeletonize
    if mask_dict['blue'].shape.__len__() > 2:
        thr_img = (mask_dict['blue'][:, :, 0] > np.max(mask_dict['blue'][:, :, 0])/2)
    else:
        thr_img = mask_dict['blue'] > np.max(mask_dict['blue']) / 2

    # Skeletonize the inside of this, which will be our paths
    skeleton = skeletonize(thr_img)

    # Map of routes
    mapRoutes = ~skeleton

    # We can show these routes here
    plt.imshow(mapRoutes)
    plt.show()

    # Return these routes
    return mapRoutes, thr_img

def findShortestPath(startX, startY, endX, endY, mapRoutes, thresh):

    # We have our start and end points, but they may not necessarily fall onto
    # a path of routes, we must use a radius to find the closest equivalence to this
    # path. Usually it is not that far off, but this value may cause errors, so
    # be weary of this!
    boxr = 100

    # Safety check to make sure the points are not out of the image given this radius
    if endY < boxr: endY = boxr
    if endX < boxr: endX = boxr

    # Here are our points
    cpys, cpxs = np.where(mapRoutes[endY - boxr:endY + boxr, endX - boxr:endX + boxr] == 0)

    # Calibrate points to main scale.
    cpys += endY - boxr
    cpxs += endX - boxr

    # Find the closest point of possible path end points
    idx = np.argmin(np.sqrt((cpys - endY) ** 2 + (cpxs - endX) ** 2))
    x, y = cpxs[idx], cpys[idx]

    pts_x = [x]
    pts_y = [y]
    pts_c = [0]

    # Mesh of displacements
    xmesh, ymesh = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    ymesh = ymesh.reshape(-1)
    xmesh = xmesh.reshape(-1)

    dst = np.zeros(thresh.shape)

    # Breath first algorithm exploring a tree
    while True:
        # update distance.
        idc = np.argmin(pts_c)
        ct = pts_c.pop(idc)
        x = pts_x.pop(idc)
        y = pts_y.pop(idc)
        # Search 3x3 neighbourhood for possible
        ys, xs = np.where(mapRoutes[y - 1:y + 2, x - 1:x + 2] == 0)
        # Invalidate these point from future searchers.
        mapRoutes[ys + y - 1, xs + x - 1] = ct
        mapRoutes[y, x] = 9999999
        # set the distance in the distance image.
        dst[ys + y - 1, xs + x - 1] = ct + 1
        # extend our list.s
        pts_x.extend(xs + x - 1)
        pts_y.extend(ys + y - 1)
        pts_c.extend([ct + 1] * xs.shape[0])
        # If we run of points.
        if not pts_x:
            break
        if np.sqrt((x - startX) ** 2 + (y - startY) ** 2) < boxr:
            edx = x
            edy = y
            break

    path_x = []
    path_y = []

    x = edx
    y = edy

    # Trace the best path back
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
        x += xmesh[idx]
        y += ymesh[idx]

        # Here is where radius errors can occur
        if np.sqrt((x - endX) ** 2 + (y - endY) ** 2) < boxr:
            print('Optimum route found.')
            break
        path_x.append(x)
        path_y.append(y)

    # Return our path!
    return path_x, path_y

def main():
    imageArray = importImage('maze8.jpg')

    originalImage = imageArray[0]
    hsv_points = imageArray[1]

    mask_dict = createMasks(hsv_points, originalImage)
    mask_dict, polygonArr = contourMasks(mask_dict, originalImage)

    startX, startY, endX, endY = findCentroids(mask_dict)

    mapRoutes, thresh = findPaths(mask_dict)
    path_x, path_y = findShortestPath(startX, startY, endX, endY, mapRoutes, thresh)

    plt.figure(figsize=(14,14))
    plt.imshow(originalImage)
    plt.plot(path_x, path_y, 'r-', linewidth=5)
    plt.show()

main()