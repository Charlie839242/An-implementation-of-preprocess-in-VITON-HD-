import cv2
import os

#转换图片尺寸
from PIL import Image
import os.path
import glob
def convertjpg(jpgfile, outdir, width= 720, height= 960):
    img=Image.open(jpgfile)
    new_img=img.resize((width,height),Image.BILINEAR)
    new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))

for jpgfile in glob.glob(r"./cloth-seg/input_images/*.jpg"):
    convertjpg(jpgfile,r"./cloth-seg/input_images")

print('Resize Done !')



