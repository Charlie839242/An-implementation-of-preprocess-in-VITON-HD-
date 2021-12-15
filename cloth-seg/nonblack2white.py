from PIL import Image

img = Image.open("1.png")  # 读取照片
img = img.convert("RGB")    # 转换格式，确保像素包含alpha通道
width, height = img.size     # 长度和宽度
for i in range(0, width):     # 遍历所有长度的点
    for j in range(0, height):       # 遍历所有宽度的点
        data = img.getpixel((i,j))  # 获取一个像素
        if (data != (0, 0, 0)):  # RGBA都是255，改成透明色
            img.putpixel((i,j),(255,255,255))

img.save("2.png")  # 保存图片

print('Convert To White !')