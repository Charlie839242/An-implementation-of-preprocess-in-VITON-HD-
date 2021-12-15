from PIL import Image
import numpy as np

img0 = Image.open('./output/0.png')
img1 = Image.open('sample.png')
width, height = img0.size

print(width, height)

palette11 = img1.getpalette()
img0.putpalette(palette11)

load0 = img0.load()

for i in range(width):
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


# -------------- add neck by brown color -------------- #
small_face, big_face, real_face = [], [], []      # real = big - small
no_neck_img = Image.open('./output/1.png')        # pixel 13 means face
no_neck_load = no_neck_img.load()
for i in range(width):                          # width = 768
    for n in range(height):                     # height = 1024
        if no_neck_load[i, n] == 13:
            small_face.append([i, n])


for i in range(width):                          # width = 768
    for n in range(height):                     # height = 1024
        if load0[i, n] == 13:
            big_face.append([i, n])

for i in big_face:
    if i not in small_face:
        load0[i[0], i[1]] = 10


img0.save('./output/0.png')
#
#
#
#
#
#
# 获取仅有黑白图片,以去掉Mask
for i in range(width):
    for n in range(height):
        if load0[i, n] != 0:
            load0[i, n] = 255
img0.save('./output/1.png')




# #---------------- 获取调色盘 ----------------#
# # 获取我们的图的调色盘
# palette0 = np.array(img0.getpalette(),dtype=np.uint8).reshape((256,3))
# print('palette0 obtained. Shape:', palette0.shape)
#
# # 获取目标图1的调色盘
# palette1 = np.array(img1.getpalette(),dtype=np.uint8).reshape((256,3))
# print('palette1 obtained. Shape:', palette1.shape)
#
# #---------------- 获取图片中用了哪些颜色 ----------------#
# color_list0, color_list1 = [], []
# pixel0, pixel1 = [], []
# color0, color1 = img0.getcolors(), img1.getcolors()
#
# for i in range(len(color0)):
#     number, count = color0[i]
#     color_list0.append(count)
#     pixel0.append(palette0[count])
# print('color_list0 : ', color_list0, '\n', 'corresponding pixel :', pixel0)
# for i in range(len(color1)):
#     number, count = color1[i]
#     color_list1.append(count)
#     pixel1.append(palette1[count])
# print('color_list1 : ', color_list1, '\n', 'corresponding pixel :', pixel1)










# color = []
# width, height = img.size
# print('width and height:', width, height)
# for i in range(width):
#     for n in range(height):
#         pixel[i, n] = 0
# img.save('pixel=0.png')







