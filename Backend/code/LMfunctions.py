import csv
import cv2
import numpy as np
import json
import os
import pickle
import base64
from scipy.interpolate import splprep, splev

import preferences as pref

####################################################################################
def classData(outputPath):
    dictFilePath = os.path.join(outputPath, "categories.dict")
    fileTest = os.path.isfile(dictFilePath)
    if fileTest == False:
        print("Can't find the categories.dict file!")
        quit()
    with open(dictFilePath, 'rb') as handle:
        dictData = pickle.load(handle)
        return dictData

####################################################################################
def getClassName(index, outputFolder):
    classDict = classData(outputFolder)
    #the labelmetococo script and the labelmetoAug start at different indexes, one at zero and the other at 1
    #test if there's a zero start
    adjust = 0
    try: 
        classDict[index]
    except KeyError: 
        adjust = 1
    index = int(index)+ int(adjust)
    return classDict[index]

####################################################################################
def imageSize(image):
    imageHeight, imageWidth, channels = image.shape
    return imageHeight, imageWidth

####################################################################################
#reduce the amount of points on the path for labelme json
#https://agniva.me/scipy/2016/10/25/contour-smoothing.html
def smoothContour(contour):
    smoothened = []
    x,y = contour.T
    # Convert from numpy arrays to normal arrays
    x = x.tolist()[0]
    y = y.tolist()[0]
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
    try:
        tck, u = splprep([x,y], u=None, s=1.0, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        pointCount = pref.getPointCountForPath() #increase last variable to increase the points on the path
        u_new = np.linspace(u.min(), u.max(), pointCount) #75) 
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))
        contour = np.array(smoothened)
    except:
        contour = contour
    return contour

####################################################################################
#labelmeFormat function
def labelMePoints(contour):
    try:
        points = contour.reshape((-1,2))
        points = points.tolist()
    except:
        points = []
    return points

####################################################################################
#labelmeFormat function
def labelMeShape(className, points):
    null = nullForJson()
    shape = {"label":className, "points": points, "group_id": null, "shape_type": "polygon",  "flags": {}}
    return shape

####################################################################################
def goodClass(className):
    #checks if the classname is on the ignore list
    result = True
    for i in pref.ignoreClassNames():
        if i == className:
            result = False
    return result

####################################################################################
#labelmeFormat function
def labelMeShapes(contours, classNumbers, outputFolder):
    #TODO ONLY KEEP GOODCLASSES?
    print("Ignore these nulls and warnings...")
    shapes = []
    for counter in range(len(contours)):
        contour = contours[counter]
        smoothedContour = smoothContour(contour) #BUG HERE? raise TypeError('m > k must hold')
        classNumber = classNumbers[counter]
        className = getClassName(classNumber, outputFolder)
        classNamePrint = str(className) + "_predicted_" + str(counter)
        points = labelMePoints(smoothedContour)
        shape = labelMeShape(classNamePrint, points)
        if goodClass(className):
            shapes.append(shape)
    return shapes

####################################################################################
#labelmeFormat function
def labelMeData(version, shapes, fileName, imageData, imageHeight, imageWidth):
    data = {"version": version, "flags": {}, "shapes": shapes, "imagePath": fileName, "imageData": imageData, "imageHeight": imageHeight, "imageWidth": imageWidth}
    return data 

####################################################################################
#labelmeFormat function
def saveLabelMeJson(jsonData, imageNameNoExtension, pathClassifyDirectory):
    saveFilePath = os.path.join(pathClassifyDirectory, imageNameNoExtension + ".json")
    with open(saveFilePath, "w") as f:
        json.dump(jsonData, f, ensure_ascii=False, indent=2)

####################################################################################
#labelmeFormat function
def saveLabelMePNG(image, imageNameNoExtension, imagesOutDirectory):
    saveFilePath = os.path.join(imagesOutDirectory, imageNameNoExtension + ".png")
    #print("Here: " + image.size)
    if image.size !=  0:
        cv2.imwrite(saveFilePath, image)

####################################################################################
def nullForJson():
    return print('"null"'.strip('"'))

####################################################################################
        #labelmeFormat function
def saveAsLabelMe(imageNameNoExtension, image, contours, classNumbers, outputFolder, pathClassifyDirectory):
    _, encoded_img = cv2.imencode('.png', image)
    imageHeight, imageWidth = imageSize(image)
    shapes = labelMeShapes(contours, classNumbers, outputFolder)
    version = pref.labelMeVersion()
    fileName = imageNameNoExtension + ".png"
    if pref.labelMe_SaveImageDataInJson() == True:
        imageData = base64.b64encode(encoded_img).decode("utf-8")
    else:
        imageData = nullForJson()
    if pref.labelMe_SaveImageFilePNG() == True:
        saveLabelMePNG(image, imageNameNoExtension, pathClassifyDirectory)
    jsonData = labelMeData(version, shapes, fileName, imageData, imageHeight, imageWidth)
    if pref.labelMe_SaveFileJson() == True:
        saveLabelMeJson(jsonData, imageNameNoExtension, pathClassifyDirectory)