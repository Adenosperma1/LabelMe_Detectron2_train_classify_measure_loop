
import json
import os
import pickle
import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
import detectron2.modeling.anchor_generator 
#detectron set up unique to trainer
from detectron2.engine import DefaultTrainer
import preferences as pref
import shutil





class trainDetectron2():
    def __init__(self, baseDir):
        nameTrainDir = pref.nameTrainDir() 
        nameJsonFile = pref.nameTrainJsonFile() + ".json" 
        trainDir = os.path.join(baseDir, nameTrainDir)
        jsonFilePath = os.path.join(trainDir, nameJsonFile)
        cocoFilePath = pref.cocoFilePath() 

        ####################################################################################
        #test if the train folder exists...
        dirTest = os.path.isdir(trainDir)
        if dirTest == False:
            print("ERROR: Can't find train directory?")
            exit()
        else:
            print("Found training directory...")
       

        ####################################################################################
        #count categories in the json data
        register_coco_instances(nameTrainDir, {}, jsonFilePath, trainDir)
        fileTest = os.path.isfile(jsonFilePath)
        if fileTest == False:
            print("ERROR: Can't find json data??")
            exit()
        else:
            print("Found training Json...")

        ####################################################################################
        #get a list of categories from the json file
        categoryData = {}
        with open(jsonFilePath) as f:
            jsonData = json.load(f)
            print(jsonFilePath)
        for each in jsonData['categories']:
            categoryData[each['id']] = each['name']

        ####################################################################################
        classCount = len(categoryData)
        if (classCount == 0):
            print("ERROR: Issue with Json Data?")
            exit()
        else:
            print("Categories: " + str(classCount))

        ####################################################################################
        #detectron configs
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cocoFilePath))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cocoFilePath)
        cfg.DATASETS.TRAIN = (nameTrainDir,)
        cfg.DATALOADER.NUM_WORKERS = pref.cfgNUM_WORKERS() #2
        cfg.SOLVER.IMS_PER_BATCH = pref.cfgIMS_PER_BATCH() #2 
        cfg.SOLVER.BASE_LR = pref.cfgBASE_LR() #0.00025 
        cfg.SOLVER.MAX_ITER = pref.iterations()
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = classCount# + 1
        #print(cfg.MODEL.ANCHOR_GNERATOR.SIZES) # = [[32, 64, 128]]
        
        ####################################################################################
        #delete or rename old output directory
        '''
        keepOldOutputDir = pref.keepOldOutputDir()
       #if keepOldOutputDir == False:
        #    if os.path.isdir(cfg.OUTPUT_DIR):
        #        shutil.rmtree(cfg.OUTPUT_DIR)   
        #else: 
        #    if os.path.isdir(cfg.OUTPUT_DIR):
         #       count = 0
         #       files = os.listdir(baseDir)
          #      for aFile in files:
          #          if os.path.isdir(aFile):
          #              if ("output" in aFile): 
           #                 count += 1
           #     print("Backing up last output dir as : output_" +  str(count))
          #      outputPath = os.path.join(baseDir, "output_" + str(count))
           #     os.mkdir(outputPath)
           #     try:
          #          shutil.move(cfg.OUTPUT_DIR,outputPath)
          #      except:
           #         print("Is the Log_classify file open?")
           #         quit()
           #         #os.rename(cfg.OUTPUT_DIR, outputPath)
        '''           
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        out_config_file = os.path.join(cfg.OUTPUT_DIR, "configs.yaml")
        f = open(out_config_file, 'w')
        f.write(cfg.dump())
        f.close()
        #save categories so the classification script can read it in
        out_category_file = os.path.join(cfg.OUTPUT_DIR, "categories.dict")
        f1 = open(out_category_file, "wb")
        pickle.dump(categoryData, f1)
        f1.close()


        ####################################################################################
        #train
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

def main():
    baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #print(baseDir)
    os.chdir(baseDir)
    trainDetectron2(baseDir)


if __name__ == "__main__":
    main()