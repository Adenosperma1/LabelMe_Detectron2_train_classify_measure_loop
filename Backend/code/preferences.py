#Measurement preferences################################################

def conversionPixelLength():
    return 30 # an int, the amount of pixels along a certain distance that you want to convert to the unit above

def conversionUnitLength():
    return  1 #the 'real world' size of the pixel length. i.e 1 (cm) 4 (inches)

def conversionUnit():
    return "cm" #i.e. a string, mm, cm, inches






#Train preferences################################################


def iterations():
    return 500 
   
def cfgNUM_WORKERS():
    return 2

def cfgIMS_PER_BATCH():
    return 2

def cfgBASE_LR():
    return 0.00025 

#Classify preferences################################################

def accuracy(): #dectron's predication accuracy 0 to 1
    return 0.7

def accuracyValidationMaskIOU(): # 0 to 100
    return 50

def labelMe_SaveFileJson(): #saves the labelme json file for each image 
    return True

def savePredicationImage(): #save a png file with the predications drawn on to it
    return False

def drawBoundingBoxPredictionMasksFile():
    #shows the min rect, the rotated min rectangle (RED) used to make measurements, and the rotated leaf
    return False

def getPointCountForPath(): #how many points should be on the labelme paths. 
    return 30

def labelMeVersion():
    return "4.5.7"

def ignoreClassNames():
    #A list of names that you don't want to see in the classification results
    #TODO code it so adding an _hide to a classname in Labelme will automatically hide them
    return ["partLeaf", "background", "groupLeaf", "fruit"]

#def savePredicationImage():
    #return True

def moveCompleteImageToTrain():
    return True 
    #if you open a classify image in labelme, you can change the mask classes from prediction to say full, leaf or draw a new mask
    #next time you train the png and json file will be copied from classify to train and added to the training set
    #false will put each labelled item onto their own page with a 0 background to the original image size, could be useful if you don't want to label everything in the image
    
    #don't bother with false, the code works but it doesn't seem to help training!
   #"Crop" #True #'Crop' to crop masks to a bounding box labeled 'crop', else False will crop each labeled object to their own page


def labelMe_SaveImageDataInJson():
    #labelme by default saves a copy of the image in the json, so this turns that off
    return False

def labelMe_SaveImageFilePNG():
    #this saves blank image that can be used with the labelme json file... because it now saves the json to the classify dir
    #it expects to find the image there already
    return False


##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################


#ignore everything below here...
###Probably leave as is:

def nameTrainDir():
    return "train"

def nameTrainJsonFile():
    return "images"

def nameClassifyDirectory():
   return "classify"

def nameValidationDirectory(): 
    return "validate"

def nameValidationJsonFile():
   return "images"

def nameValidationResultsFile():
   return "Log_validation"

def nameOutDirectory(): 
    return "images" 

def nameModel(): 
    return "model_final.pth"


import os


def pathClassifyDirectory():
    return os.path.join(".", nameClassifyDirectory())


def pathValidationDirectory():
   return os.path.join(".", nameValidationDirectory())


def pathValidationJson():
    return os.path.join(pathValidationDirectory(), nameValidationJsonFile() + '.json')

def segmentOrBox():
    #do you want to do image segmentation or just bounding box classification
    return "s" #"s" or "b" b doesn't work atm? need to look into it


def pathYaml(): 
    if segmentOrBox() == "s":
        return "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    if segmentOrBox() == "b":
        return "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" 


def cocoFilePath(): #the code currently expects a zoo file
   if segmentOrBox() == "s":
        return "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
   if segmentOrBox() == "b":
        return "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" 


def conversionHeightWidth(pixels):
    #7cm = 200 pixels long when measured in Photoshop. #or 3cm = 87 pixels #or 1cm = 30 pixels
    return pixels * (conversionUnitLength() / conversionPixelLength())

def conversionArea(pixels):
    return pixels / (conversionPixelLength() * conversionPixelLength())