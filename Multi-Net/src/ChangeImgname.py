# ChangeImgname.py
import os

path = 'RTM/'
classes = ['QH/', 'SX/', 'XZ/', 'ZY/']
type = ['Train/', 'Test/']

for i in range(4):
    for j in range(2):
        img_dir = path + classes[i] + type[j]
        cnt = 0
        for img_name in os.listdir(img_dir):
            print(cnt)
            os.rename(os.path.join(img_dir,img_name), os.path.join(img_dir, str(cnt) +'.png'))
            cnt = cnt + 1
