
# coding: utf-8
import os, zipfile
import numpy as np
from PIL import Image

def crop_center(img, crop_size=128):
    """
     This function can crop center of image. 
     If image size is smaller than crop_size, image will be padded "0".
    """
    if type(img) is np.ndarray:
        img = Image.fromarray(np.uint8(img))
    ratio = crop_size / min(img.size)
    img.thumbnail(list(map(lambda x: int(x * ratio), img.size)))
    x, y  = map(lambda x: (x - crop_size) / 2, img.size)
    img   = img.crop((x, y, x + crop_size, y + crop_size))
    return np.array(img)

def img2gif(inputs, save_path, duration=60):
    """
     This function can create gif file. 
     It is very useful to visualize the generated images by GANs etc.
    """
    if save_path[-4:] != ".gif":
        print("The parameter 'save_path' should be full path(including file name), and the extension should be '.gif'")
        return
    
    pil_imgs = []
    if type(inputs) in [list, np.ndarray]:
        for img in inputs:
            pil_imgs.append(Image.fromarray(np.uint8(img)))
    else:
        for f in os.listdir(inputs):
            pil_imgs.append(Image.open(inputs + f))
        
    pil_imgs[0].save(save_path, save_all=True, append_images=pil_imgs[1:], optimize=False, duration=duration, loop=0)
    print("Completed.")

def create_zip_files(input_path, save_path):
    """
     This function can create zip file. 
    """
    if save_path[-4:] != ".zip":
        print("The parameter 'save_path' should be full path(including file name), and the extension should be '.zip'")
        return

    with zipfile.ZipFile(save_path, 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        for f in os.listdir(input_path):
            new_zip.write(input_path + f)

