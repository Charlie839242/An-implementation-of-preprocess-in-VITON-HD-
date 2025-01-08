```
# 这个项目用来记录在复现VITON-HD的时候，是如何得到输入VITON-HD的模型的数据的。

## Check VITON-HD Here! [VITON-HD](https://github.com/shadow2496/VITON-HD)

这里，我们的目标是如何将一张有背景的人物图和一张由背景的衣服图片，转化成VITON-HD模型所需要的输入数据。

在VITON-HD的官方代码里，可以看见输入的数据总共有六个, cloth, cloth-mask, image, image-parse, openpose-img and openpose-json，其实际对应的具体图片如下。

| cloth         | 去除了背景的只有衣服的衣服图片    |
| ------------- | --------------------------------- |
| cloth-mask    | 用黑白表示的衣服分割图 (白衣黑底) |
| image         | 去除了背景的只有人物的人物图片    |
| image-parse   | 对人体不同部位的语义分割          |
| openpose-img  | 以不同颜色呈现的人体关键点检测图  |
| openpose-json | 人体关键点的坐标数据              |



<!-- # 复现背景

最近在做一个华为的比赛，想要制作一个换装照相机，即制作一个基于树莓派的照相机，通过拍摄人物图片和衣服图片，并将图片上传到华为云服务器上进行处理后，返回换装过后的结果图片到用户端。

因此，重点就是如何在云服务器上进行虚拟换装（Virtual Try On），我在查阅了相关资料后，发现现有的虚拟换装存在分辨率较低的问题，普遍为256×256，分辨率低即用户的体验不会太好。直到看到了[VITON-HD](https://github.com/shadow2496/VITON-HD)，发现该模型能够输出1024×768的图片，分辨率很不错，因此决定部署VITON-HD到云端服务器上。

 -->

# 复现思路



### 1. 衣服分割



#### 1.1 衣服分割

对于实现衣服分割，我们采用了U2-Net来进行分割。[U2-Net](https://github.com/xuebinqin/U-2-Net)

因为要分割的是衣服，因此我们选择了数据集[[iMaterialist (Fashion) 2019 at FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data)]。这个数据集对不同的衣服的不同部位进行了不同的标记，所实现的精度已经超过了我们所需要的目标，但为了方便就选择这个数据集了。

在输出的分割图中，不同的衣服部位用不同的颜色标记。关于具体的实现，可以参考[U2-Net](https://github.com/xuebinqin/U-2-Net)官网。我自己在笔记本电脑上训练的模型精度还不错，但是[Clothes Segmentation using U2NET](https://github.com/Charlie839242/cloth-segmentation)这个仓库中提供了训练了很多轮模型，比我自己训练的模型效果要好，有需要可以从上面下载。



衣服的原图和分割后的效果如下：

<p align="middle">     
    <img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/original_cloth.jpg" width="200">     
    <img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/cloth_seg.jpg" width="200">    
</p>






#### 1.2 把图片中非黑的像素都转化成白色

由于最终模型输出会给衣服的不同部位画上不同颜色，而VITON-HD需要的模型输入是黑白的分割图，因此需要转化。

```
img = Image.open(args.cloth_path + "/0.jpg")  # 读取照片
img = img.convert("RGB")  					  # 转换格式
width, height = img.size  					  # 长度和宽度
print(width, height)
for i in range(0, width):  					  # 遍历
for j in range(0, height):  				  
data = img.getpixel((i, j))  				  # 获取一个像素
if (data[0] >= 64):
img.putpixel((i, j), (255, 255, 255))
img = img.convert('L')
img.save(args.cloth_path + "/0.jpg")  		  # 保存图片
```

得到的效果如下：

<p align="middle">   
    <img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/cloth_color.jpg" width="200">     
</p>






#### 1.3 利用获得的黑白分割图来去除衣服图片的背景

这一步我们要将输入衣服原图中的背景通过获得的分割图去除。

```
edge_path='output_images/0.jpg'
content_path='input_images/0.jpg'

img_edge = cv2.imread(edge_path)
img_content = cv2.imread(content_path)				# 直接将img_edge和img_content相乘像素会溢出，出现不正常的颜色

img_inverse_edge=255*np.ones_like(img_edge)-img_edge		# 翻转分割图的颜色 
img0=img_edge/255

image=img0*img_content+img_inverse_edge
image=np.array(image,dtype=np.uint8)
cv2.imwrite(content_path, image)
```

得到的效果如下：

<p align="middle">   
    <img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/cloth_no_background.jpg" width="200">     
</p>




至此，我们通过输入的带有背景的衣服图片，获得了没有背景的衣服图片以及黑白的分割图。



## 2. 人体分割

#### 2.1 对脖子Neck的疑惑

对于人体分割，在查看了作者的测试集后，我发现作者得到的Image-Parse图片都是将脖子用棕色给分割出来了。

然而，作者在论文中提到了他用的ACGPN中的分割模型来进行的人体分割。当我去查阅的时候，我发现这个分割方面有三个主要数据集，LIP，ATR和Pascal。然而，这三个数据集中都没有包含脖子（Neck）的标签。然而如果分割出来没有脖子，那么最终的虚拟换衣效果就不会好。因为所有训练集的分割图都有脖子的标签，而我们分割出来的却没有。如下：（左边为原图，中间为作者图，右为我们的图）

<p align="middle">   
	<img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/original_person.jpg" width="200">
    <img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/sample.png" width="200">   
    <img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/ATR.png" width="200">
</p>



就在要走投无路的时候，我在另一篇虚拟换衣领域的重要论文[DeepFashion_Try_On](https://github.com/switchablenorms/DeepFashion_Try_On)中发现了这样一句话:

*"[2021-12-3] The light point artifacts seem to be caused by the variance of the imprecise human parsing when we rearrange the data for open-sourcing. We recommend to use the **ATR** model in https://github.com/PeikeLi/Self-Correction-Human-Parsing to get the human parsing with neck label to stablize training. "*

这句话的意思大概就是说他们原先的模型在人的脖子上会产生一些亮点，为了克服这个亮点的缺点，作者推荐我们使用[Self-Correction-Human-Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing)里的ATR训练集来进行训练。可是我刚刚查阅的资料显示ATR数据集里根本没有Neck的标签。抱着怀疑的态度，我从[Self-Correction-Human-Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing)上下载了基于ATR数据集和LIP数据集的模型，并进行了推理，结果如下：

<p align="middle">   
    <img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/ATR.png" width="200">   
    <img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/LIP.png" width="200">
</p>






很明显，基于LIP数据集得到的图片，在脖子那一块是黑色的；而基于ATR数据集得到的图片，脖子被用和脸一样的颜色给标记出来了。原来ATR数据集并不是单独给Neck制作了一个标签，而是对Neck进行了标注，但是是用和脸一样的标签来标注的。

这样的话就有办法得到人物的脖子的区域了，我们只需要将从LIP和ATR产生的图片中脸的部分进行相减，就可以获取脖子的部分。



#### 2.2 开始分割

这里由于我的目标部署平台只有CPU，没有CUDA，而[Self-Correction-Human-Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing)中只提供了利用GPU推理的方式，因此我选择了另一个集成了许多AI模型的库, [AILIA](https://github.com/axinc-ai/ailia-models)。关于如何安装这个库，详见[Tutorial](https://github.com/axinc-ai/ailia-models/blob/master/TUTORIAL.md)。这个库恰好具备了我们所需要的ATR模型和LIP模型。而且为我们提供了转换好的ONNX模型，可以在CPU上进行推理。得到的图片如下：

<p align="middle">   
	<img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/original_person.jpg" width="200">
    <img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/ATR.png" width="200">   
    <img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/LIP.png" width="200">
</p>







#### 2.3 转化颜色

可以发现，在2.2中得到的人体分割图和官方给出的用于测试的人体分割图，在颜色上明显不同，因此，在这一步我们需要转换得到的人体分割图的颜色，以使得和测试集中的颜色一致。



这里笔者在处理图片的时候遇见了一个坑：ATR和LIP模型得到的图片都是 ***P 模式*** 的！由于之间没有接触过P模式的图片，所以在这里浪费了很多时间。

那么什么是P模式呢？我们用PIL来打开图片看看：

```
from PIL import Image
img0 = Image.open('./output/0.png')
print(img0)
```

可以看到结果是：

```
<PIL.PngImagePlugin.PngImageFile image mode=P size=768x1024 at 0x15227E09040>
```

打印出来的结果表明： **image mode = P**。

在查阅资料后，终于搞明白了：P模式的图片是一种基于调色盘的图片，这种储存方式是为了减小图片的内存而发明的。在图片的开头，存有256个RGB颜色，我们用numpy数组来打印看看：

```
palette0 = np.array(img0.getpalette(),dtype=np.uint8).reshape((256,3))
```

 得到的结果如下

```
[[  0   0   0]
 [128   0   0]
 [254   0   0]
 [  0  85   0]
 ···
 [253 253 253]
 [254 254 254]
 [255 255 255]]				# totally 256 rows
```

在调色盘之后，才是图片的颜色。后面每一个像素只对应一个数字，代表调色盘中的储存的颜色，而不是像RGB图像对应一个[255, 255, 255]这样的numpy数组。这里记录一些操作P模式图片的操作：

```
img0 = Image.open('./output/0.png')				# 读取图片
palette11 = img1.getpalette()					# 获取调色盘
palette0 = np.array(img0.getpalette(),dtype=np.uint8).reshape((256,3)) # 打印调色盘
img0.putpalette(palette11)						# 将palette11置为img0的调色盘
color0 = img0.getcolors()						# 获取图片中的像素及个数
```



下面我们进行颜色转化，先看看我们通过ATR模型得到的图片和作者提供的sample进行对比（左边为Sample）:

<p align="middle">   
	<img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/sample.png" width="200">
    <img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/ATR.png" width="200">   
</p>




通过打印这两张图片所包含的像素和调色盘 我找到了这两张图片中不同颜色与不同部位的对应关系:

| ATR得到的图  |                       |           | Sample图     |                      |          |
| ------------ | --------------------- | --------- | ------------ | -------------------- | -------- |
| 调色盘对应值 | 颜色                  | 对应部位  | 调色盘对应值 | 颜色                 | 对应部位 |
| 0            | [0, 0, 0]: 黑         | 背景      | 0            | [0, 0, 0]: 黑        | 背景     |
| 2            | [0, 128, 0]: 绿       | 头发      | 2            | [254, 0, 0]: 红      | 头发     |
| 4            | [0, 0, 128]: 蓝       | 衣服      | 5            | [254, 85, 0]: 橘色   | 衣服     |
| 5            | [128, 0, 128]: 紫     | 裤子      | 9            | [0, 85, 85]: 深绿    | \        |
| 11           | [192, 128, 0]: 棕黄   | 脸 + 脖子 | 10           | [85, 51, 0]: 棕      | 脖子     |
| 12           | [64, 0, 128]: 深紫    | 右腿      | 12           | [0, 128, 0]: 绿      | 裤子     |
| 13           | [192, 0, 128]: 粉     | 左腿      | 13           | [0, 0, 254]: 蓝      | 脸       |
| 14           | [64, 128, 128]: 浅蓝  | 右手      | 14           | [51, 169, 220]: 浅蓝 | 右手     |
| 15           | [192, 128, 128]: 肉色 | 左手      | 15           | [0, 254, 254]: 亮蓝  | 左手     |
|              |                       |           | 16           | [85, 254, 169]:浅绿  | 右腿     |
|              |                       |           | 17           | [169, 254, 85]: 亮绿 | 左腿     |

因此, 从ATR得到的图转化到Sample图, 我们先将Sample图的调色盘放进ATR得到的图,之后我们只需要进行如下像素值的变化:

4&rArr;5		5&rArr;12		11&rArr;13		12&rArr;16		13&rArr;17



具体实现如下:

```
img0 = Image.open('./output/0.png')			# ATR Image
img1 = Image.open('sample.png')				# Sample Image
width, height = img0.size

palette11 = img1.getpalette()
img0.putpalette(palette11)					# 将ATR Image的调色盘置为Sample Image的

load0 = img0.load()

for i in range(width):						# 遍历所有像素进行颜色转换
    for n in range(height):
        if(load0[i, n] == 4):
            load0[i, n] = 5
        elif(load0[i, n] == 5):
            load0[i, n] = 12
        elif(load0[i, n] == 11):
            load0[i, n] = 13
        elif(load0[i, n] == 12):
            load0[i, n] = 16
        elif(load0[i, n] == 13):
            load0[i, n] = 17

img0.save('./output/0.png')
```



现在我们得到了转化颜色后的图片，如下：

<p align="middle">   
	<img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/person_color.png" width="200">
</p>






#### 2.4 添加脖子区域

但是我们目前的图片脖子Neck是和脸Face用一个颜色标记出来的. 正如之前讨论的, 我对此的解决方法是基于ATR输出的图片中脸部标记包括了脖子,而LIP模型输出的图片中脸部标记没有包括脖子, 将这两张图片中的脸部区域相减, 即可得到脖子Neck的区域. 

假定我们现在已经同过ATR和LIP分别得到了两张图片0.png和1.png, 我们首先将这两张图片通过上面一步的代码转化颜色, 再进行对脖子Neck的标记. 具体实现如下:

```
# -------------- add neck by brown color -------------- #
small_face, big_face, real_face = [], [], []      # 储存着从不同照片中读取的脸部像素坐标
img0 = Image.open('./output/0.png')	
no_neck_img = Image.open('./output/1.png')        
no_neck_load = no_neck_img.load()
for i in range(width):                          # width = 768
    for n in range(height):                     # height = 1024
        if no_neck_load[i, n] == 13:			# pixel 13 means face
            small_face.append([i, n])			# 获取LIP输出图片中脸部的坐标


for i in range(width):                          # width = 768
    for n in range(height):                     # height = 1024
        if load0[i, n] == 13:
            big_face.append([i, n])				# 获取ATR输出图片中脸部的坐标

for i in big_face:								# 存在在ATR脸部中但不在LIP脸部
	if i not in small_face:						# 中的像素就是脖子Neck的像素
        load0[i[0], i[1]] = 10					# 变成棕色,即Sample中脖子的颜色
        
img0.save('./output/0.png')
```

得到图片的效果如下:

<p align="middle">   
	<img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/person_neck.png" width="200">
</p>






同时,我们基于上面转化好颜色并添加了脖子标签的图片,再获取一张一人是全白的图片, 背景是全黑的分割图,用于去除原人物图中的背景:

获取分割图, 即将当前所有非黑像素转化成白色即可:

```
for i in range(width):
    for n in range(height):
        if load0[i, n] != 0:					# 若不是黑色
            load0[i, n] = 255					# 则转化成白色
img0.save('./output/1.png')

```

获取的图片如下:

<p align="middle">   
	<img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/person_white.png" width="200">
</p>






下一步就是将原人物图的背景去除, 这一步的代码和第一部中去除衣物背景时的背景用到的方法相同. 需要注意的一个点是我们在上一步保存的黑白分割图是以***mode P***保存的, 用PIL读出来是单通道图, 但当我们用cv2来读取的时候,会自动读取RGB的格式, 所以将该图片与原图片相乘的话不会出现两个图片的dimension不同的报错. 具体实现如下:

```
edge_path='output/1.png'
content_path='input/0.jpg'

img_edge = cv2.imread(edge_path)
img_content = cv2.imread(content_path)
# 直接将img_edge和img_content相乘像素会溢出，出现不正常的颜色

img_inverse_edge=255*np.ones_like(img_edge)-img_edge		# 翻转分割图的颜色 
img0=img_edge/255

image=img0*img_content+img_inverse_edge
image=np.array(image,dtype=np.uint8)
cv2.imwrite(content_path, image)
```

<p align="middle">   
	<img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/person_no_background.jpg" width="200">
</p>




至此, 我们已经实现了获得没有背景的人物图, 已经加上了脖子标签并修正了颜色的人物分割图.



### 3. 姿态检测

姿态检测的实现比较简单，是基于[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)实现的。

对于生成VITON-HD需要的姿态图，我这里选择了简单地用[Windows Portable Version](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#windows-portable-demo)来实现。

从这里下载源码[Windows Portable Version](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#windows-portable-demo), 然后运行‘models/getBaseModels.bat’和‘models/getCOCO_and_MPII_optional.bat’，会自动的下载相关模型。下载完成后，将我们需要的图片放在./examples/media/0.jpg:

在主目录执行以下指令：

```
bin\OpenPoseDemo.exe --image_dir examples\media --hand --write_images output\ --write_json output/ --disable_blending
```

这样会将骨架图保存在output里，同时也会将关键点坐标等信息保存在json文件里：

<p align="middle">   
	<img src="https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-/blob/main/img/openpose.png" width="200">
</p>
```


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



















































