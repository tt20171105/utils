# coding: utf-8
import os, io, tarfile, zipfile
import numpy as np
from PIL import Image

"""
Compression file should be the following architecture.

Compression file
  |-- Label A
  |     |-- image file
  |     |-- image file
  |-- Label B
  |     |-- image file
  |     |-- image file
"""

extentions = ["jpg", "JPG", "png", "PNG"]

def load_zip_file(path, resize):
    """
      This function can load the compression format, '.zip'.
    """
    if type(resize) is not tuple or len(resize) != 2:
        print("Parameter 'resize' should be tuple and the length should be three.")
        return
    
    dict = {}
    loaded_num, error_num = 0, 0
    with zipfile.ZipFile(path, 'r') as zf:
        for idx, f in enumerate(zf.namelist()):
            if f[-3:] not in extentions:
                label_name  = f[f[:-1].rfind("/")+1:-1]
                dict[label_name] = np.array([])
                print("Start loading. Label name is %s." % label_name)
                continue
            try:
                img = np.array(Image.open(io.BytesIO(zf.read(f))).resize(resize))
                dict[label_name] = np.append(dict[label_name], img).reshape(-1, *img.shape)
                loaded_num += 1
            except:
                print("File '%s' was not able to load." % f)
                error_num  += 1
                
    for key, imgs in dict.items():
        print("Label %s's shape is %s" % (key, imgs.shape))
        dict[key] = imgs.astype(int)
    print("The number of loaded file was %s." % loaded_num)
    print("The number of error file was %s." % error_num)
    return dict

def load_tar_file(path, resize):
    """
      This function can load the compression format, '.tar' and '.tar.gz'.
    """
    if type(resize) is not tuple or len(resize) != 2:
        print("Parameter 'resize' should be tuple and the length should be three.")
        return
    
    dict = {}
    loaded_num, error_num = 0, 0
    tar  = tarfile.open(path, 'r')
    for idx, f in enumerate(tar):
        if idx==0:  # Index 0 is tar file name.
            continue
        if f.name[-3:] not in extentions:
            label_name  = f.name[f.name.rfind("/")+1:]
            dict[label_name] = np.array([])
            print("Start loading. Label name is %s." % label_name)
            continue
        try:
            img = tar.extractfile(f.name)
            img = np.array(Image.open(io.BytesIO(img.read())).resize(resize))
            dict[label_name] = np.append(dict[label_name], img).reshape(-1, *img.shape)
            loaded_num += 1
        except:
            print("File '%s' was not able to load." % f)
            error_num  += 1
            
    tar.close()
    for key, imgs in dict.items():
        print("Label %s's shape is %s" % (key, imgs.shape))
        dict[key] = imgs.astype(int)
    print("The number of loaded file was %s." % loaded_num)
    print("The number of error file was %s." % error_num)
    return dict

