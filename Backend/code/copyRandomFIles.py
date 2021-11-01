#move some random files
python

import os
import shutil
import random

num = 35
root_src = "C:\\Users\\brendan\\Desktop\\backend\\classify\\"

files = os.listdir(root_src)
if len(files) > num:
    for x in range(num):
        file = random.choice(files)
        file_path = os.path.join(root_src, file)
        new_path = os.path.join(root_src, 'NEW', file) 
        shutil.copy(file_path, new_path) 
