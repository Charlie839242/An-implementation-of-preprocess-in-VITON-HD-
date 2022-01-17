from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=int, required=True, help='class = 0 is for person, class = 1 is for cloth')
parser.add_argument('--person_path', type=str, default='./person-seg/output', required=False)
parser.add_argument('--cloth_path', type=str, default='./cloth-seg/output_images', required=False)
args = parser.parse_args()

if(args.type == 0):
    img = Image.open(args.person_path + "/0.png")  # 读取照片
    x,y = img.size
    p = Image.new('RGBA',img.size,(255,255,255))
    p.paste(img,(0,0,x,y),img)
    p = p.convert('RGB')
    p.save(args.person_path + "/0.jpg")

    print('Person Convert To White !')

elif(args.type == 1):
    img = Image.open(args.cloth_path + "/0.jpg")  # 读取照片
    img = img.convert("RGB")  # 转换格式
    width, height = img.size  # 长度和宽度
    print(width, height)
    for i in range(0, width):  # 遍历所有长度的点
        for j in range(0, height):  # 遍历所有宽度的点
            data = img.getpixel((i, j))  # 获取一个像素
            if (data[0] >= 64):
                img.putpixel((i, j), (255, 255, 255))
    img = img.convert('L')
    img.save(args.cloth_path + "/0.jpg")  # 保存图片

    print('Cloth Convert To White !')
else:
    pass
