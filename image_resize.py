import os
from PIL import Image


path = r'E:\data\flower_photos\roses\\'
copy_path = r'E:\data\flower_photos\roses_cp\\'


if not os.path.exists(copy_path):
    os.mkdir(copy_path)

listDir = os.listdir(path)  # 获取当前目录下的所有内容
for image_name in listDir[:]:

    try:
        image_path = path+image_name
        im = Image.open(image_path).resize((640,640))
        im.save(os.path.join(copy_path, image_name))

    except:
        pass

