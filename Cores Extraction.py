import numpy as np
import cv2 as cv
import time
from imutils import perspective as im
import ContourInfo as cont

# This code is used for creating an image pipeline for the core images
# It takes in the masked image, finds the contours, sorts them, then extracts
# them from the original image, and finally channels them into the pipeline.

# create the image files to read in
imageNum = 7
maskFile = r'C:\Users\GoldSpot_Cloudberry\OneDrive - Goldspot Discoveries Inc\Documents\Goldspot\Images\mask' + str(
    imageNum) + '.png'
maskImg = cv.imread(maskFile)

origFile = r'C:\Users\GoldSpot_Cloudberry\OneDrive - Goldspot Discoveries Inc\Documents\Goldspot\Images\image' + str(
    imageNum) + '.png'
origImg = cv.imread(origFile)


# this function identifies all contours in the image whose area is >= areaThres
def drawContours(image):
    # create a threshold for detecting contours on the masked image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    flag, thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)
    tempCont, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # add all the contours whose area is greater than the given threshold
    for i in range(len(tempCont)):
        if cv.contourArea(tempCont[i]) >= 0:#areaThres:
            contours.append(tempCont[i])

    print("There are %d contours in this image" % (len(contours)))


# this function sorts the contours from top to bottom and left to right (currently the most time expensive function)
def sortContours(image, contours):
    NO_CONT = -1
    rows, cols = image.shape[:2]
    arr = np.arange(rows * cols, dtype=np.int32)
    arr = arr.reshape((rows, cols))
    labels = np.full_like(arr, NO_CONT, dtype=np.int32)

    # initializes the dependency tree with the contour information of all the contours
    dependency_tree = {}
    for ind, contour in enumerate(contours):
        cv.drawContours(labels, [contour], -1, ind, -1)
        dependency_tree[ind] = cont.ContourInfo(ind, contour)

    # construct the dependencies tree, processing cols from bottom up
    for c in range(cols):
        lastCont = NO_CONT
        # we scan from bottom up because we want the top contours to depend on the bottom ones
        # so that they can get pruned off the tree first
        for r in range(rows - 1, -1, -1):
            currCont = labels[r][c]
            if currCont != NO_CONT:
                if (lastCont != currCont) and (lastCont != NO_CONT):
                    dependency_tree[lastCont].add_dependency(currCont)
                lastCont = currCont

    # sort the dependency tree by removing one leaf at a time
    sorted_contours = []
    while bool(dependency_tree):
        # we construct a list of candidates containing all nodes that are leaves in the dependency tree
        candidates = []
        for node in dependency_tree.values():
            if node.is_leaf():
                candidates.append(node.index)

        # sort the candidates by their depth, which gives precedence to the leftmost contours
        candidates.sort(key=lambda n: dependency_tree[n].depth())

        # add the best contour to our sorted list and completely remove that contour from the dependency tree
        bestContour = dependency_tree.pop(candidates[0])
        sorted_contours.append(contours[bestContour.index])
        for node in dependency_tree.values():
            node.remove_dependency(candidates[0])

    return sorted_contours


# this function now extracts the contours from the original image
def extractContours(image):
    for i in range(len(contours)):
        rect = cv.minAreaRect(contours[i])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        # orders the points in clockwise to perform the proper homography later
        src_pts = im.order_points(box).astype("float32")

        # compute the dimensions of the rectangle
        width = np.linalg.norm(src_pts[1] - src_pts[0])
        height = np.linalg.norm(src_pts[2] - src_pts[1])

        # coordinates of the points in box points after rectangle is horizontal
        dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

        # the homography transformation matrix
        M, status = cv.findHomography(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv.warpPerspective(image, M, (width, height))

        # writes the image to file and adds it to the images array
        cv.imwrite("crop_img" + str(i) + ".png", warped)
        images.append(warped)
        cv.waitKey(0)


# this function creates the image pipeline from all the contour images in the array
def createPipeline():
    # compute the height and width of the image pipeline
    height = max([image.shape[0] for image in images])
    width = sum([image.shape[1] for image in images])

    result = np.zeros((height, width, 3), dtype=np.uint8)
    currX = 0
    # crop out all the contour images into the pipeline
    for image in images:
        result[:image.shape[0], currX:image.shape[1] + currX, :] = image
        currX += image.shape[1]

    cv.imwrite("Image " + str(imageNum) + ".jpg", result)


areaThres = 100 * 100
images = []
contours = []
totalTime = 0

start = time.time()
drawContours(maskImg)
end = time.time()
totalTime += (end - start)
print("Detecting the contours took %.3f seconds" % (end - start))

start = time.time()
contours = sortContours(maskImg, contours)
end = time.time()
totalTime += (end - start)
print("Sorting the contours took %.3f seconds" % (end - start))

start = time.time()
extractContours(origImg)
end = time.time()
totalTime += (end - start)
print("Extracting the contours took %.3f seconds" % (end - start))

start = time.time()
createPipeline()
end = time.time()
totalTime += (end - start)
print("Creating the pipeline took %.3f seconds" % (end - start))
print("The entire program ran in %.3f seconds" % (totalTime))
