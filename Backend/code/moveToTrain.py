import os
import json
import glob
import cv2
import numpy as np

import preferences as pref
import LMfunctions as LM



##########################################################################################
def process(jsondata, fileName, trainDirectoryPath, classifyDirectoryPath):
    moveCompleteImageToTrain = pref.moveCompleteImageToTrain()
    #print("here1")
    pngName = fileName + ".png"
    pngPath = os.path.join(classifyDirectoryPath, pngName)
    image = cv2.imread(pngPath)
    imageHeight, imageWidth = LM.imageSize(image)
    version = pref.labelMeVersion()
    imageData = LM.nullForJson()
    shapes = jsondata["shapes"]
    #print(boxes)

    print("File with new labels: " + fileName)

    if pref.moveCompleteImageToTrain() == True:
        #this should work...
        print("Copying PNG and JSON to train directory for file: " + fileName)
        shapes = jsondata["shapes"]
        newName = nameAndNumber(fileName, trainDirectoryPath)
        #print("Newname: " + newName)
        jsonData = LM.labelMeData(version, shapes, (newName + ".png"), imageData, imageHeight, imageWidth)
        LM.saveLabelMeJson(jsonData, newName, trainDirectoryPath)
        LM.saveLabelMePNG(image, newName, trainDirectoryPath)
   
    ##########################################################################################
    #NOT WOKRING YET
    elif pref.moveCompleteImageToTrain() == "Crop": #added this as an after thought
        #crop the image within bounding box... bounding box needs to be called 'crop'
        #any masks inside that area will be kept and the image and json will be put into the train dir
        for shape in shapes:
            
            label = shape["label"]
            if  "crop" in label:
                print("found a crop box 2")
                points = shape["points"]
                shapes = intersectingShapes(points, shapes)
                print(shapes)
                #exit()
                newName = nameAndNumber(fileName, trainDirectoryPath)

                
                
                jsonData = LM.labelMeData(version, shapes, (newName + ".png"), imageData, imageHeight, imageWidth)
                LM.saveLabelMeJson(jsonData, newName, trainDirectoryPath)
                print("New Name: " + newName)
                #c2.waitKey(0) # 0==wait forever
                start_point, end_point = startEndPoints(points)
                mask = np.zeros(image.shape[:2], dtype="uint8")
                noprint = cv2.rectangle(mask, start_point, end_point, 255, -1)
                masked = cv2.bitwise_and(image, image, mask=mask)
                LM.saveLabelMePNG(masked, newName, trainDirectoryPath)
                print("found a crop box 2")
                #make the json data
                #print(boxes)




    #not sure if the code below works anymore!
    else:
        print("keep single images")
        ##########################################################################################
        #this was to crop each mask into their own file with a 0'd background so the image stayed the same size
        count = 1
        for shape in jsondata["shapes"]:
            label = shape["label"]
            #print(label)
            if  "_predicted" not in label:
                #print("New shape to save out...")
                
                newName = nameAndNumber(fileName, trainDirectoryPath)
                shapes = []
                shapes.append(shape)
                jsonData = LM.labelMeData(version, shapes, (newName + ".png"), imageData, imageHeight, imageWidth)
                LM.saveLabelMeJson(jsonData, newName, trainDirectoryPath)
                #https://stackoverflow.com/questions/61168140/opencv-removing-the-background-with-a-mask-image
                #make a bounding box for the shape/shapes?
                points = shape["points"]
                #padding= 20
                #this could error if the shape is too close to an edge
                min_x, min_y, max_x, max_y = boundingBox(points)
                start_point, end_point = startEndPoints(points)
                mask = np.zeros(image.shape[:2], dtype="uint8")
                noprint = cv2.rectangle(mask, start_point, end_point, 255, -1)
                masked = cv2.bitwise_and(image, image, mask=mask)
                LM.saveLabelMePNG(masked, newName, trainDirectoryPath)
                #k = cv2.waitKey(0) # 0==wait forever

##########################################################################################
def nameAndNumber(fileName, trainDirectoryPath):
    newName = fileName
    count = 0
    pngFiles = glob.glob(os.path.join(trainDirectoryPath, "*.png"))
    #print(pngFiles)
    for file in pngFiles:    
        if fileName in file:
            #print('y')
            count += 1
            newName = fileName + "_" + str(count)
    return newName

##########################################################################################
def boundingBox(points):
    min_x = int(min(point[0] for point in points)) 
    min_y = int(min(point[1] for point in points)) 
    max_x = int(max(point[0] for point in points)) 
    max_y = int(max(point[1] for point in points)) 
    return min_x, min_y, max_x, max_y


####################################################################################
def intersectingShapes(box, shapes):
    #print("____________________11")
    newshapes = []
    print(len(shapes))
    if(len(shapes) > 0):
       # print("____________________12")
        #for counter in range(len(boxes)):
        for shape in shapes:
            label = shape["label"]
            print(shape["label"])
            if  "crop" not in shape["label"]:
                #print(222)
                points = shape["points"]
                if boundingBoxIntersecting(box, points) == True:
                    newshapes.append()
                    #print("Hi_____________________________")
    return newshapes

#########################################################################################
def startEndPoints(points):
    min_x, min_y, max_x, max_y = boundingBox(points)
    start_point = (min_x, min_y) #top left corner?
    end_point = (max_x, max_y)
    return start_point, end_point

####################################################################################
def boundingBoxIntersecting(boxA, boxB):
    #print(33333333333333333333333333333333333333333)
    boxA = bounding_box(boxA)
    boxB = bounding_box(boxB)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    #print("Area: " + interArea)
    if interArea == 0:
        return False
    return True

####################################################################################
def bounding_box(points):
    #TODO THE BOXES NEED TO BE CONVERTED SO THE COMPARISION WORKS
  x_coordinates, y_coordinates = zip(*points)
  return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)


##########################################################################################
def main():
    baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nameClassifyDir = pref.nameClassifyDirectory()
    nameTrainDir = pref.nameTrainDir()
    trainDirectoryPath = os.path.join(baseDir, nameTrainDir)
    classifyDirectoryPath = os.path.join(baseDir, nameClassifyDir)
    print("Scanning classification files for new Masks... ")
    if os.path.exists(classifyDirectoryPath):
        labelme_json = glob.glob(os.path.join(classifyDirectoryPath, "*.json"))
        for num, json_file in enumerate(labelme_json):
            #print("File: " + json_file)
            with open(json_file, "r") as fp:
                _, jsonfileName = os.path.split(json_file)
                (fileName, ext) = os.path.splitext(jsonfileName)
                jsondata = json.load(fp)
                #print("json: " + str(jsondata))
                try:
                    for shapes in jsondata["shapes"]:
                        label = shapes["label"]
                        #print(label)
                        if  "_predicted" not in label:
                            print("Process fileName: " + fileName)
                            process(jsondata, fileName, trainDirectoryPath, classifyDirectoryPath)
                            break
                except:
                    one = 1

if __name__ == "__main__":
    main()
    print("______________moveToTrain.py Done.")

