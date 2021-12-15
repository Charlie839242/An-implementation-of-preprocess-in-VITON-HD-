from PIL import Image
import numpy as np
img0 = Image.open('./output/0.png')
palette0 = np.array(img0.getpalette(),dtype=np.uint8).reshape((256,3))
print(palette0)