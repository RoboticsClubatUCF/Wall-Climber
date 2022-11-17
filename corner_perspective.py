import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow


def perspective_transform(img_name, pix_bound):
    img = cv.imread(img_name)
    img = cv.resize(img, (pix_bound, pix_bound))
    xpix = []
    ypix = []
    
    for row in range(pix_bound):
        for col in range(pix_bound):
            color = list(img[row, col])
            if (color[2] < color[0] and color[1] < color[0]):
                # Paints each selected pixel even darker blue. For testing.
                # Could also be used as a mask of some sort?
                #img[row, col] = [255, 0, 0]
                xpix.append(row)
                ypix.append(col)

    x_min_id = xpix.index(min(xpix))
    x_max_id = xpix.index(max(xpix))
    y_min_id = ypix.index(min(ypix))
    y_max_id = ypix.index(max(ypix))
    
    corner1 = [xpix[x_min_id], ypix[x_min_id]] #small x, small y
    corner2 = [xpix[x_max_id], ypix[x_max_id]] #big x, big y
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

    # Flips the x and y of the corners, because for some crazy reason cv2 wants
    # them in reverse order.
    for i in range(4):
        temp = corner_arr[i][0]
        corner_arr[i][0] = corner_arr[i][1]
        corner_arr[i][1] = temp

    final_corner_arr = np.float32(corner_arr)

    out_corner1 = [0, 0] 
    out_corner2 = [0, pix_bound - 1]
    out_corner3 = [pix_bound - 1, pix_bound - 1]
    out_corner4 = [pix_bound - 1, 0] 
    new_corner_arr = [out_corner1, out_corner2, out_corner3, out_corner4]
    new_corner_arr = np.float32(new_corner_arr)

    # Verifies corner position (commented out for now because it's not needed)
    '''
    # top left corner (1)
    img = cv.circle(img, (corner_arr[0][0], corner_arr[0][1]), 4, (0, 0, 255), 1)
    # bottom left corner (2)
    img = cv.circle(img, (corner_arr[1][0], corner_arr[1][1]), 4, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[1][0], corner_arr[1][1]), 8, (0, 0, 255), 1)
    # bottom right corner (3)
    img = cv.circle(img, (corner_arr[2][0], corner_arr[2][1]), 4, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[2][0], corner_arr[2][1]), 8, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[2][0], corner_arr[2][1]), 12, (0, 0, 255), 1)
    # top right corner (4)
    img = cv.circle(img, (corner_arr[3][0], corner_arr[3][1]), 4, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[3][0], corner_arr[3][1]), 8, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[3][0], corner_arr[3][1]), 12, (0, 0, 255), 1)
    img = cv.circle(img, (corner_arr[3][0], corner_arr[3][1]), 16, (0, 0, 255), 1)
    '''

    # Warps the image based on the located corners
    matrix = cv.getPerspectiveTransform(final_corner_arr, new_corner_arr)
    warped_img = cv.warpPerspective(img, matrix, (pix_bound, pix_bound))

    # Returns the warped image
    return warped_img  



def main():
    image = perspective_transform("Maze.jpg", 400)
    cv2_imshow(image)
    return 0

main()
