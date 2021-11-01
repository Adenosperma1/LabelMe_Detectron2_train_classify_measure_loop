import moveToTrain
import convert
import train
import classify
import asyncio
import os


#all files look at the preferences file for variables 

async def one():
    #this looks at the classification images to see if there are any new masks that need to be moved to the train directory, then moves them and creates json
    moveToTrain.main()

async def two():
    #this creates the images.json files for the train and validation directories
    convert.main()

async def three():
    #does the training with detectron2, results go into the output dir
     train.main()


async def four():
    #does the classification, validation, measures, saves labelme json in the classify dir, creates preview png files and a spreadsheet in the output/images dir
    classify.main()



if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    codeDirPath = os.path.join(baseDir, "code")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(one()) 
    loop.run_until_complete(two())
    try:
        #need to check if the model file is there etc...
        loop.run_until_complete(three()) #this always errors and then stops the next part, so as a workaround it expects to fail then run the next script...
    except:
        loop.run_until_complete(four())
    loop.close()

