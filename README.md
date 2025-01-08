# Usage

Step 1: Cloth

将衣服图片放进**./cloth-seg/input_images/**下。

```
python .\cloth-seg\infer.py
python color.py --type 1
cd cloth-seg
python clean_mask.py
cd ..
```



Step 2: Person

将人物图片放进**./person-seg/image_segmentation/human_part_segmentation/input/**下。

```
cd ./person-seg/image_segmentation/human_part_segmentation
python human_part_segmentation_atr.py
python human_part_segmentation_lip.py
python palette.py
python clean_mask.py
cd ..
cd ..
cd ..
```



Step 3: Openpose

在[Windows Portable Version](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#windows-portable-demo)下载Openpose。点击models/getBaseModels.bat和models/getCOCO_and_MPII_optional.bat来下载需要的模型。

```
bin\OpenPoseDemo.exe --image_dir examples\media --hand '
          '--write_images output\ --write_json output/ --disable_blending'
```



















































