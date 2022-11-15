import cv2 as cv
import numpy as np


def main():
    pix_bound = 400 # size of the image
    img_name = "C:\\Users\\spenc\\Downloads\\UCF\\Robotics\\Wall-Climber\\Wall-Climber\\src\\Raspberry_pi\\computer_vision\\mask.jpg" # file name or file path
    img = cv.imread(img_name)
    img = cv.resize(img, (pix_bound, pix_bound))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
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
    
    corner1 = (xpix[x_min_id], ypix[x_min_id]) #small x
    corner2 = (xpix[x_max_id], ypix[x_max_id]) #big x
    corner3 = (xpix[y_min_id], ypix[y_min_id]) #small y
    corner4 = (xpix[y_max_id], ypix[y_max_id]) #big y
    corner_arr = [corner1, corner2, corner3, corner4]
    print(corner_arr)

    # Verifies corner position
    img = cv.circle(img, (corner1[1], corner1[0]), 8, (255, 0, 0), 1)
    img = cv.circle(img, (corner2[1], corner2[0]), 8, (255, 0, 0), 1)
    img = cv.circle(img, (corner3[1], corner3[0]), 8, (255, 0, 0), 1)
    img = cv.circle(img, (corner4[1], corner4[0]), 8, (255, 0, 0), 1)

    cv.imshow('dst',img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()    


main()