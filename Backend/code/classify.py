#sorry I wasn't aware of PEP8 before writing this code!

import PIL.Image
import csv
import cv2
import imutils
import numpy as np
import os
import shutil

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import preferences as pref
import LMfunctions as LM







class classifyDetectron2():
    #tally's
    def __init__(self):
        self.counterPrediction = 0
        self.counterValidation = 0
        self.counterPredictionAndValidation = 0
        self.counterPredictionNoValidation = 0
        self.counterValidationNoPrediction = 0

      

        ####################################################################################
        def convertWidthHeight(pixels):
            newMeasurement = pref.conversionHeightWidth(pixels)
            newMeasurement = round(newMeasurement, 2)
            return newMeasurement

        ####################################################################################
        def convertArea(pixels):
            newMeasurement = pref.conversionArea(pixels)
            newMeasurement = round(newMeasurement, 2)
            return newMeasurement

        ####################################################################################
        def removeOldJson():
            if os.path.isdir(pref.pathClassifyDirectory()):
                files = os.listdir(pref.pathClassifyDirectory())
                for aFile in files:
                    if (".json" in str(aFile)): 
                        filePath = os.path.join(pref.pathClassifyDirectory(), aFile)
                        os.remove(filePath)
            print("Removed old json files from the Classify Dir...")
        
        if pref.labelMe_SaveFileJson() == True: #saves the labelme json file for each image 
            removeOldJson()

        ####################################################################################
        def countClasses(outputPath):
            dictData = LM.classData(outputPath)
            return len(dictData)

        ####################################################################################
        #get the groundtruth data
        def setUpconfigs():
            existsFile = os.path.isfile(pref.pathValidationJson())
            if (existsFile  == True):
                groundtruth_metadata, validationDictsArray = configsTest()
                return validationDictsArray
            else:
                print("No COCO json file with validation segmetations")
            return []

        ####################################################################################
        #get the groundtruth data
        def saveValidationResults(outputDirectoryName):
            logFile = os.path.join(pref.nameValidationResultsFile() + '.csv')
            header = ["Output Directory", 'PredictionsForValidationImages_Pd', 'Validations_Ad', 'Prediction With Validations_Cd', 'Prediction Without Validations', "Validations without Predictions", "Precision Cd / Pd", "Recall Cd / Ad", "F1 2pr/p+r", "Pref Iterations", "Pref cfgNUM_WORKERS", "Pref cfgIMS_PER_BATCH", "Pref cfgBASE_LR", "Pref accuracy", "Pref accuracyValidationMaskIOU"]
            predictionsWithValidationImage = self.counterPredictionAndValidation + self.counterPredictionNoValidation
            precision = ((self.counterPredictionAndValidation / predictionsWithValidationImage) * 100)
            recall = ((self.counterPredictionAndValidation / self.counterValidation) * 100)
            F1 = ((2 * (precision * recall)) / (precision + recall))
            footer = [outputDirectoryName] + [predictionsWithValidationImage] + [self.counterValidation] + [self.counterPredictionAndValidation] + [self.counterPredictionNoValidation] + [self.counterValidationNoPrediction] + [round(precision, 2)] + [round(recall, 2)] + [round(F1, 2)] + [pref.iterations()] +  [pref.cfgNUM_WORKERS()]  + [pref.cfgIMS_PER_BATCH()] + [pref.cfgBASE_LR()] + [pref.accuracy()] + [pref.accuracyValidationMaskIOU()]
            print("Predications : " + str(predictionsWithValidationImage))
            print("Validations : " + str(self.counterValidation))
            print("Precision : " + str(precision))
            print("Recall : " + str(recall))
            print("F1 : " + str(F1))
            existsFile = os.path.isfile(logFile)
            if (existsFile  == True): #read it in, add the line and save it
               with open(logFile, 'a', encoding='UTF8', newline='') as logFile:
                   writer = csv.writer(logFile)
                   writer.writerow(footer)
                   logFile.close
            else:
                with open(logFile, 'w', encoding='UTF8', newline='') as logFile:
                    writer = csv.writer(logFile)
                    writer.writerow(header)
                    writer.writerow(footer)
                    logFile.close
        
                    ####################################################################################
        def renameOutputDir(): 
            if os.path.isdir(cfg.OUTPUT_DIR):
                count = 1
                files = os.listdir(".")
                for aFile in files:
                    if os.path.isdir(aFile):
                        if ("output_" == aFile[ 0 : 7 ]): 
                            count += 1
                name = "output_" + str(count)
                print("Renaming output dir as : " + name)
                outputPath = os.path.join(".", name)
                try:
                    shutil.copytree(cfg.OUTPUT_DIR,outputPath)
                except:
                    print("Couldn't move the output folder?")
                return name



        ####################################################################################
        #detectron configs for classification 
        register_coco_instances(pref.nameClassifyDirectory(), {}, "", pref.pathClassifyDirectory())
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(pref.pathYaml()))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(pref.pathYaml())
        classCount = countClasses(cfg.OUTPUT_DIR)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(classCount) #+ 1
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, pref.nameModel())
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = pref.accuracy()
        predictor = DefaultPredictor(cfg)
        imagesOutDirectory = os.path.join(cfg.OUTPUT_DIR, pref.nameOutDirectory())


        ####################################################################################
        def removeOldPredicationImages():
            if os.path.isdir(imagesOutDirectory):
                shutil.rmtree(imagesOutDirectory)
                #files = os.listdir(imagesOutDirectory)
                #for aFile in files:
                    #filePath = os.path.join(imagesOutDirectory, aFile)
                   # os.remove(filePath)
            print("Removed old Output/Images Dir...")


        if pref.savePredicationImage()  == True:
            removeOldPredicationImages()

        ####################################################################################
        #set up detectron to read in the validation json data
        def configsTest():
            register_coco_instances(pref.nameValidationDirectory(), {}, pref.pathValidationJson(), pref.pathValidationDirectory())
            groundtruth_metadata = MetadataCatalog.get(pref.nameValidationDirectory())
            groundtruth_dicts = DatasetCatalog.get(pref.nameValidationDirectory())
            return groundtruth_metadata, groundtruth_dicts

        ####################################################################################
        def contourFrom(bitMask):
            imageIndices = bitMask.astype(np.uint8)
            imageIndices *= 255
            contours = cv2.findContours(imageIndices.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            try:
                contour = contours[np.argmax([cv2.contourArea(x) for x in contours])]
            except:
                contour = []
            return contour

        ####################################################################################
        def contoursArrayFrom(bitMaskArray):
            contoursArray = []
            for bitMask in bitMaskArray:
                contours = contourFrom(bitMask)
                if len(contours) != 0:
                    contoursArray.append(contours)
            return contoursArray

        ####################################################################################
        def getColour(maskClass, P_V):
            colour = (0,255,0) #green
            if P_V == "V":
                colour = (255,0,255) #magenta
            return colour

        ####################################################################################
        def getContourArea(contour): 
            return cv2.contourArea(contour)

        ####################################################################################
        def cart2pol(x, y):
            theta = np.arctan2(y, x)
            rho = np.hypot(x, y)
            return theta, rho

        ####################################################################################
        def pol2cart(theta, rho):
            x = rho * np.cos(theta)
            y = rho * np.sin(theta)
            return x, y

        ####################################################################################
        def rotate_contour(contour, angle):
            try:
                M = cv2.moments(contour)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                contour_norm = contour - [cx, cy]
                coordinates = contour_norm[:, 0, :]
                xs, ys = coordinates[:, 0], coordinates[:, 1]
                thetas, rhos = cart2pol(xs, ys)
                thetas = np.rad2deg(thetas)
                thetas = (thetas + angle) % 360
                thetas = np.deg2rad(thetas)
                xs, ys = pol2cart(thetas, rhos)
                contour_norm[:, 0, 0] = xs
                contour_norm[:, 0, 1] = ys
                contour_rotated = contour_norm + [cx, cy]
                contour_rotated = contour_rotated.astype(np.int32)
                return contour_rotated
            except:
                return 0


        ####################################################################################        
        def getboundingCircleMeasurements(contour):
            (xcircleCenter, ycircleCenter), radius = cv2.minEnclosingCircle(contour)
            center = (int(xcircleCenter), int(ycircleCenter))
            radius = int(radius) #x2 for height
            heightcircle = radius * 2
            #make the circle height an even number for easier comparision with box height
            if (heightcircle % 2) != 0:
                heightcircle += 1
            return heightcircle, xcircleCenter, ycircleCenter

        ####################################################################################
        def getBoundingBoxMeasurements(contour):
            xbox, ybox, widthbox, heightbox = cv2.boundingRect(contour)
            #this is so 2 even numbers can be compared... May not be the best way to do it... i.e. the height of the bounding circle and the height of the bounding box
            if (heightbox % 2) != 0:
                heightbox += 1
            return xbox, ybox, widthbox, heightbox

        ####################################################################################
        def getContourMeasurements(image, contour):
            x, y, widthbox, heightbox = getBoundingBoxMeasurements(contour)
            heightcircle, xcircleCenter, ycircleCenter = getboundingCircleMeasurements(contour)
            area = getContourArea(contour) #not using this anymore
            if area != 0: 
                for angle in range(360):
                    rotated = rotate_contour(contour, angle) #normal bounding box
                    x, y, widthbox, heightbox = getBoundingBoxMeasurements(rotated)
                    if heightcircle == heightbox:
                        break
            if pref.drawBoundingBoxPredictionMasksFile() == True:
                drawRotatedBoundingBox(image, x, y, widthbox, heightbox, rotated, angle, heightcircle, xcircleCenter)
            return widthbox, heightbox, angle, area


        ####################################################################################
        def getContourMeasurementsArray(image, contours, bitmasks): 
            #TODO remove results once we are happy with using the contour area vs the mask area
            measurementsArray = []
            for counter in range(len(contours)):
                mask = bitmasks[counter]
                areaMask = np.sum(mask)
                contour = contours[counter]
                widthbox, heightbox, angle, area = getContourMeasurements(image, contour)
                measurementsArray.append([widthbox, heightbox, angle, area, contour, areaMask])
            return measurementsArray

        ####################################################################################
        def measurementsArrayAccuracy(measurementsArrayItem, measurementsArrayValidationItem):
            widthbox, heightbox, angle, _, _, areaMask = measurementsArrayItem
            widthboxValidation, heightboxValidation, angleValidation, _, _, areaMaskValidation = measurementsArrayValidationItem
            widthboxAccuracy = accuracy(widthbox, widthboxValidation)
            heightboxAccuracy = accuracy(heightbox, heightboxValidation)
            angleAccuracy = accuracy(angle, angleValidation)
            areaAccuracy = accuracy(areaMask, areaMaskValidation)
            return widthboxAccuracy, heightboxAccuracy, angleAccuracy, areaAccuracy

        ####################################################################################
        def fileNameWithoutExt(filePath):
            imageName = os.path.basename(filePath)
            imageNamesplit = os.path.splitext(imageName)
            imageNameNoExtension = imageNamesplit[0]
            return imageNameNoExtension

        ####################################################################################
        def classifyImage(image):
            outputs = predictor(image)
            classNumbers = outputs["instances"].pred_classes.to(device="cpu").numpy()
            bitMasks = outputs["instances"].pred_masks.to(device="cpu").numpy()
            scores = outputs["instances"].scores.to(device="cpu").numpy()
            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            return classNumbers, bitMasks, boxes, scores

        ####################################################################################
        def classifyValidationImage(pathImageValidation, listGroundtruthDicts):
            segments = []
            classNumbers = []
            boxes = []
            for aDict in listGroundtruthDicts:
                if aDict['file_name'] == pathImageValidation:
                    for annot in aDict['annotations']:
                        segments.append(annot['segmentation'])
                        classNumbers.append(annot['category_id'])
                        boxes.append(annot["bbox"])
            return classNumbers, segments, boxes

        ####################################################################################
        def bitMasksIOU(m1, m2):
            inx = np.sum(np.logical_and(m1,m2))
            unx = np.sum(np.logical_or(m1,m2))
            iou = inx / unx
            iou = int(round(iou, 2) * 100)
            return inx, unx, iou

        ####################################################################################
        def bitMaskFrom(segment, height, width):
            bitMask = detectron2.structures.masks.polygons_to_bitmask(segment,height,width)
            return bitMask

        ####################################################################################
        def bitMaskArrayFrom(segmentArray, height, width):
            bitMaskArray = []
            for segment in segmentArray:
                bitMask = bitMaskFrom(segment, height, width)
                bitMaskArray.append(bitMask)
            return bitMaskArray

        ####################################################################################
        def boundingBoxIOU(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
            if interArea == 0:
                return 0
            boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
            boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou

        ####################################################################################
        def boundingBoxXYWH2XYXY(box):
            x1 = box[0]
            y1 = box[1]
            w = box[2]
            h = box[3]
            x2 = x1 + w
            y2 = y1 + h
            return [x1, y1, x2, y2]

        ####################################################################################
        def accuracy(var, varTest):
            if (varTest == 0):
                return 0
            return int(round((var / varTest), 2) * 100)

        ####################################################################################
        def getContours(image, results, P_V):
            if len(results) > 0:
                imageHeight, imageWidth, c = image.shape
                if P_V == "V":
                    bitMasks = bitmasksFromResults(results, image)
                else:
                    bitMasks = results[1]          
                contours = contoursArrayFrom(bitMasks)
            return contours, bitMasks 

        ####################################################################################
        def bitmasksFromResults(results, image):
            imageHeight, imageWidth, c = image.shape
            segments = results[1]
            bitMasks = bitMaskArrayFrom(segments, imageHeight, imageWidth)
            return bitMasks

        ####################################################################################
        def drawImage(image, results, P_V, contours, boundingBoxMeasurements):
            if len(results) > 0:
                classDict = LM.classData(cfg.OUTPUT_DIR)
                classNumbers = results[0]
                if len(contours) > 0:
                    image = drawMasks(image, contours, classNumbers, P_V, boundingBoxMeasurements)
            return image

        ####################################################################################
        def drawContour(image, contour, colour):
            noprint = cv2.drawContours(image, [contour], 0, colour, 1)
            return image

        ####################################################################################
        def boundingBoxContourFromSegmentContour(contour):
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            contour = np.int0(box)
            return contour

        ####################################################################################
        def drawRotatedBoundingBox(image, x, y, w, h, contour, angle, cx, cy):
            red = (0,0,255)
            noprint = cv2.rectangle(image, (x, y), (x + w, y + h), red, 1)
            noprint = cv2.drawContours(image,[contour],0,red,1)
            
        ####################################################################################
        def drawBoundBox(image, contour, colour):
            boxcontour = boundingBoxContourFromSegmentContour(contour)
            noprint = cv2.drawContours(image,[boxcontour],0, colour ,1)

        ####################################################################################
        def drawContourLabel(image, text, contour, colour):
            M = cv2.moments(contour)
            if (M["m00"] == 0.0):
                return image
            if (M["m01"] == 0.0):
                return image
            if (M["m10"] == 0.0):
                return image
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            bottomLeftCornerOfText = (cX, cY)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            lineType = 2
            cv2.putText(image, text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            colour,
            lineType)
            return image

        ####################################################################################
        def drawMasks(image, contours, classNumbers, P_V, boundingBoxMeasurementsArray):
            for counter in range(len(contours)):
                contour = contours[counter]
                area = getContourArea(contour)
                classNumber = classNumbers[counter]
                screenCounter = counter + 1
                colour = getColour(classNumber, P_V)
                className = LM.getClassName(classNumber, cfg.OUTPUT_DIR)
                labelText = P_V + "_" + str(screenCounter) + "_" + str(className)
                widthbox, heightbox, angle, area, contourNULL_ATM, areaMask = boundingBoxMeasurementsArray[counter]
                if P_V == "V":
                    labelText = '\n' + labelText #try to make the validation title not overlap the predication label
                if LM.goodClass(className):
                    image = drawContour(image, contour, colour)
                    image = drawContourLabel(image, labelText, contour, colour)
                    if pref.drawBoundingBoxPredictionMasksFile() == True:
                        drawBoundBox(image, contour, colour) # a min rectangle bounding box before being rotated
            return image

        ####################################################################################
        def boundingBoxIOUDetails(boxPrediction, boxValidation, classNumber, classNumberValidation, validationsCounter):
            bBoxIOU = 0
            classNameValidation = LM.getClassName(classNumberValidation, cfg.OUTPUT_DIR)
            screenvalidationsCounter = validationsCounter + 1
            if (classNumber == classNumberValidation):
                convertedBox = boundingBoxXYWH2XYXY(boxValidation)
                bBoxIOU = boundingBoxIOU(boxPrediction, convertedBox)
                bBoxIOU = int(round(bBoxIOU , 2) * 100)
            print("             " + "V_" + str(screenvalidationsCounter) + "   Class: " + str(classNumberValidation) + ", Label: " + classNameValidation + ", Bounding box IOU: " + str(bBoxIOU) + " %")
            return bBoxIOU

        ####################################################################################
        def compareBoxes(boxPrediction, boxesValidation, classNumber, classNumbersValidation):
            boundingBoxMaxIOU = 0
            boundingBoxIndex = -1
            if (len(boxesValidation) > 0):
                for counter in range(len(boxesValidation)):
                    classNumberValidation = classNumbersValidation[counter]
                    boxValidation = boxesValidation[counter]
                    bBoxIOU = boundingBoxIOUDetails(boxPrediction, boxValidation, classNumber, classNumberValidation, counter)
                    if bBoxIOU > boundingBoxMaxIOU:
                        boundingBoxMaxIOU = bBoxIOU
                        boundingBoxIndex = counter
            return boundingBoxMaxIOU, boundingBoxIndex

        ####################################################################################
        def processMasks(imageNumber, imageNameNoExtension, image, results, resultsValidation, writer):
            classNumbers, _, boxes, scores = results
            classNumbersValidation, _, boxesValidation = resultsValidation
            imageLabelMe = image.copy()
            #draw the masks onto the image
            print("     Drawing masks on image...")
            contours, bitMasks = getContours(image, results, "P")
            contoursValidation, bitMasksValidation = getContours(image, resultsValidation, "V")
            predictionsLength = len(contours)
            validationsLength = len(contoursValidation)
            #Get measurements from the masks
            print("     Getting mask measurements...\n")
            measurementsArray = getContourMeasurementsArray(image, contours, bitMasks) #, results)
            measurementsArrayValidation = getContourMeasurementsArray(image, contoursValidation, bitMasksValidation) #, results)

            matchedValidations = [] #build a list of matched validations so you can print out a list of not found validations
            #Loop through predicted boxes and find matching validation mask
            #These are used to build up the text for the logs and to print on screen for user feedback
            emptySet = ["", "", "", "", ""]
            logNoPredication = [imageNumber] + [imageNameNoExtension] + emptySet + [""] + [""] + [""] + [""] + [""] 
            logNoValidation = emptySet
            logNoAccuracy = emptySet 
            indentSpace = "             "

            #draw the measurementsArray results onto the image
            imagePath = os.path.join(imagesOutDirectory, str(imageNumber) + "_" + imageNameNoExtension + "_P" + str(predictionsLength) + "_V" + str(validationsLength) + ".png") 
            
            indentLine = "\n" + indentSpace + "-------------------------------------------------"

            #draw the masks on the image #SAVE THE IMAGE HERE
            imagePreview = drawImage(image, results, "P", contours, measurementsArray) 
            imagePreview = drawImage(image, resultsValidation, "V", contoursValidation, measurementsArrayValidation) 


            if imagePreview.size !=  0:
                if pref.savePredicationImage()  == True:
                    print("     Saving predication image...")
                    noprint = cv2.imwrite(imagePath, imagePreview)

            print("     Saving LabelMe files...")
            LM.saveAsLabelMe(imageNameNoExtension, imageLabelMe, contours, classNumbers, cfg.OUTPUT_DIR, pref.pathClassifyDirectory())

            for counter in range(predictionsLength):
                classNumber = classNumbers[counter]
                className = LM.getClassName(classNumber, cfg.OUTPUT_DIR)
                screenCounter = counter + 1
                score = int(scores[counter] * 100)
                widthbox, heightbox, angle, _, _, areaMask =  measurementsArray[counter]
                widthboxConverted = convertWidthHeight(widthbox)
                heightboxConverted = convertWidthHeight(heightbox)
                areaMaskConverted = convertArea(areaMask)
                imageDetails = "\n" + indentSpace + "Image_" + str(imageNumber) + ", Predication_" + str(screenCounter)
                confidenceDetails = indentSpace + imageNameNoExtension + '_P' + str(screenCounter) + ', ' + str(className) + ', Confidence: ' + str(score) + " %"
                measurementsPrediction = indentSpace + "Predication: Area Pixels: " + str(int(areaMask)) + ', Height Pixels: ' + str(int(heightbox)) + ', Width Pixels: ' + str(int(widthbox)) + ', Rotated Angle: ' + str(angle)
                #these are written to the log csv
                logPredication = [str(imageNumber), imageNameNoExtension, str(screenCounter), className,  str(score),  str(int(areaMask)),  str((areaMaskConverted)), str(int(heightbox)), str(heightboxConverted), str(widthbox), str(widthboxConverted), str(angle)]
                logValidation = logNoValidation
                logAccuracy = logNoAccuracy
                P_yes = 1
                #set the global counter
                self.counterPrediction += 1
                V_yes = 0
                P_V_yes = 0 #predication and validation
                P_V_no = 0 #predication without validation
                V_P_no = 0 #validation without prediction
                #ignore unwanted Classes
                if LM.goodClass(className):
                #validations

                    if validationsLength > 0:
                        boxPrediction = boxes[counter]
                        classNumber = classNumbers[counter]
                        print(imageDetails + ", Compare with " + str(validationsLength) + " validation masks.\n")
                        print(indentSpace + "P_" + str(screenCounter) + "   Class: " + str(classNumber) + ", Label: " + className)
                        boundingBoxMaxIOU, validationMaskIndex = compareBoxes(boxPrediction, boxesValidation, classNumber, classNumbersValidation)
                        print(indentLine)
                        print(indentSpace + "Max Bounding box IOU: " + str(boundingBoxMaxIOU) + " %")
                        measurementsValidation = indentSpace + "Validation: -"
                        measurementsAccuracy = indentSpace + "Accuracy: -"
                        maskIOU = 0
                        #matching validation bounding box found
                        if boundingBoxMaxIOU > 0:
                            #P_yes = 1
                            
                            bitMask = bitMasks[counter]
                            bitMaskValidation = bitMasksValidation[validationMaskIndex]
                            maskInter, maskUnion, maskIOU = bitMasksIOU(bitMask, bitMaskValidation)
                        #matching validation mask found   
                            if  maskIOU > pref.accuracyValidationMaskIOU():
                                #("IOU: " + str(maskIOU))
                                V_yes = 1
                                #set the global counter
                                self.counterValidation += 1
                                P_V_yes = 1 #there's a predication and validation
                                #set the global counter
                                self.counterPredictionAndValidation += 1
                                matchedValidations.append(validationMaskIndex)
                                screenCounterValidation = validationMaskIndex + 1
                                widthboxValidation, heightboxValidation, angleValidation, _, _, areaMaskValidation = measurementsArrayValidation[validationMaskIndex]
                                widthboxAccuracy, heightboxAccuracy, angleAccuracy, areaAccuracy = measurementsArrayAccuracy(measurementsArray[counter], measurementsArrayValidation[validationMaskIndex])
                                measurementsValidation = indentSpace + "Validation:  Area Pixels: " + str(int(areaMaskValidation)) + ', Height Pixels: ' + str(int(heightboxValidation)) + ', Width Pixels: ' + str(int(widthboxValidation)) + ', Rotated Angle: ' + str(angleValidation)
                                measurementsAccuracy = indentSpace + "Accuracy:    Area Pixels: " + str(int(areaAccuracy)) + ' %, Height Pixels: ' + str(int(heightboxAccuracy)) + ' %, Width Pixels: ' + str(int(widthboxAccuracy)) + " %"
                                #these are written to the log csv
                                logValidation = [str(screenCounterValidation), str(int(areaMaskValidation)), str(int(heightboxValidation)), str(int(widthboxValidation)), str(angleValidation)]
                                logAccuracy = [str(int(areaAccuracy)), str(int(heightboxAccuracy)), str(int(widthboxAccuracy)), str(angleAccuracy), str(maskIOU)]
                        if maskIOU == 0: #there's a predication but no validation
                            P_V_no = 1
                            #set the global counter
                            self.counterPredictionNoValidation += 1
                        #no matching validation mask found
                        print(indentSpace + "Mask IOU: " + str(maskIOU) + " %")
                        print(confidenceDetails)
                        print(measurementsPrediction)
                        print(measurementsValidation)
                        print(measurementsAccuracy)
                    else:
                    #VALIDATIONS without a matching prediction
                        print(imageDetails)
                        print(confidenceDetails)
                        print(measurementsPrediction)
                    print(indentLine)
                    #print(indentSpace + "Log string:")
                    logString = logPredication + logValidation + logAccuracy + [P_yes] + [V_yes] + [P_V_yes] + [P_V_no] + [V_P_no]
                    #print(logString)
                    print("############################################")
                    writer.writerow(logString)
                    #TODO IF NO MATCH WRITE THE IMAGE NAME AND THE VALIDATION IMAGE DETAILS TO LOG.

            


            print(indentSpace + "Validations without matching prediction:")
            for counter in range(len(measurementsArrayValidation)):
                if counter not in matchedValidations:
                    screenCounterValidation = counter + 1
                    widthboxValidation, heightboxValidation, angleValidation, _, _, areaMaskValiation = measurementsArrayValidation[counter]
                    logValidation = [str(screenCounterValidation), str(int(areaMaskValiation)), str(int(heightboxValidation)), str(int(widthboxValidation)), str(angleValidation)]
                    logStringNotFound = logNoPredication + logValidation + logNoAccuracy + [0] + [1] + [0] + [0] + [1] #last 5 are [P_yes] + [V_yes] + [P_V_yes] + [P_V_no] + [V_P_no]
                    #set the global counters
                    self.counterValidation += 1
                    self.counterValidationNoPrediction += 1
                    writer.writerow(logStringNotFound)
                    print(logStringNotFound)


        ####################################################################################
        def mainLoop(validationDictsArray):
            PIL.Image.MAX_IMAGE_PIXELS = None
            #remove the old output 
            #if os.path.isdir(imagesOutDirectory):
                #shutil.rmtree(imagesOutDirectory)
            #put it back
            removeOldPredicationImages()
            os.makedirs(imagesOutDirectory, exist_ok=True)
            
            logName = "Log_measurements.csv"
            #logFile = os.path.join(imagesOutDirectory, "Log_"  + pref.nameClassifyDirectory() + ".csv") # + "_"  + str(timestamp) + ".csv")
            logFile = os.path.join("output", logName)
            if os.path.isfile(logFile):
                os.remove(logFile)
            with open(logFile, 'w', encoding='UTF8', newline='') as logFile:
                writer = csv.writer(logFile)
                header = ['Image Number', 'Image', 'Predication Number', 'Label Name', 'Confidence', 'Pixel Area Mask', pref.conversionUnit() + ' Area Mask', 'Pixel Length', pref.conversionUnit() + ' Length ', 'Pixel Width', pref.conversionUnit() + ' Width ', 'Box Rotation Degrees', 'Validation Number', 'Pixel Area Validation', 'Pixel Length Validation', 'Pixel Width Validation', 'Box Rotation Degrees Validation', 'Pixel Area Accuracy %', 'Pixel Length Accuracy %', 'Pixel Width Accuracy %', 'Box Rotation Degrees Accuracy %','IOU %', "Prediction", "Validation", "Prediction & Validation", "Predication No Validation", "Validation No Predication"]
                writer.writerow(header)
                counterFile = 0
                for file in os.listdir(pref.nameClassifyDirectory()):
                    if file.endswith(".png"): #ONLY USE PNG downstream labelme format functions expect png. It's been hardcoded in...
                        print("\n----------------------------------------------------------------")
                        screenCounter = counterFile + 1
                        print(str(screenCounter) + "_" + str(file) + "\n")
                        pathImage = os.path.join(pref.nameClassifyDirectory(), file)
                        image = cv2.imread(pathImage)
                        results = classifyImage(image)
                        nameImageNoExtension = fileNameWithoutExt(pathImage)
                        pathValidationImage = os.path.join('.', pref.nameValidationDirectory(), file)
                        resultsValidation = ""
                        existsFile = os.path.isfile(pathValidationImage)
                        counterFile += 1
                        if (existsFile == True):
                            print("     There's a validation image...")
                            resultsValidation = classifyValidationImage(pathValidationImage, validationDictsArray)
                        else:
                            resultsValidation = [], [], []
                        if not (results is None):
                            processMasks(counterFile, nameImageNoExtension, image, results, resultsValidation, writer)
                footer = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]  + [self.counterPrediction] + [self.counterValidation] + [self.counterPredictionAndValidation] + [self.counterPredictionNoValidation] + [self.counterValidationNoPrediction] 

                writer.writerow(footer)
            logFile.close()
            #rename the output folder then add the name to the log
            name = renameOutputDir()
            saveValidationResults(name)
            

        #RUN
        validationDictsArray = setUpconfigs()
        mainLoop(validationDictsArray)
        #save the global counters
        





def main():
   
    baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(baseDir)
    classifyDetectron2()
    

####################################################################################
if __name__ == "__main__":
    main()
    
    


