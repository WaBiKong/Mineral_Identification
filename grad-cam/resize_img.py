import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import center_crop_img

img_name = "161_25_18"

img_path = img_name + ".jpg"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img = Image.open(img_path).convert('RGB')
img = np.array(img, dtype=np.uint8)
img = center_crop_img(img, 384)

plt.imshow(img)
plt.axis('off')   # 去坐标轴
plt.xticks([])    # 去 x 轴刻度
plt.yticks([])    # 去 y 轴刻度
plt.savefig(img_name + "_resize384.jpg")