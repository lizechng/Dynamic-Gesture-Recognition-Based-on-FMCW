import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def f(img):
    height, width, channels = img.shape
    cnt_h_a, cnt_h_b = 0, 0
    cnt_w_l, cnt_w_r = 0, 0
    state_h, state_w = True, True
    for i in range(height):
        if img[i,int(width/2),0] == 255 and img[i,int(width/2),1] == 255 and img[i,int(width/2),2] == 255 and state_h:
            cnt_h_a = cnt_h_a + 1
        elif img[i,int(width/2),0] == 255 and img[i,int(width/2),1] == 255 and img[i,int(width/2),2] == 255 and not state_h:
            cnt_h_b = cnt_h_b + 1
        else:
            state_h = False

    for i in range(width):
        if img[int(height/2),i,0] == 255 and img[int(height/2),i,1] == 255 and img[int(height/2),i,2] == 255 and state_w:
            cnt_w_l = cnt_w_l + 1
        elif img[int(height/2),i,0] == 255 and img[int(height/2),i,1] == 255 and img[int(height/2),i,2] == 255 and not state_w:
            cnt_w_r = cnt_w_r + 1
        else:
            state_w = False

    return cnt_h_a, cnt_h_b, cnt_w_l, cnt_w_r

def main():
    dataset_dir = './RTM_DTM/'
    saveset_dir = './IMG/'
    list1 = ['DTM/', 'RTM/']
    list2 = ['QH/', 'SX/', 'XZ/', 'ZY/']
    list3 = ['CH1/', 'CH2/', 'CH3/', 'CH4/']

    # for i in range(2):
        # for j in range(4):
            # for k in range(4):
                # img_dir = dataset_dir + list1[i] + list2[j] + list3[k]
                # save_dir = saveset_dir + list1[i] + list2[j] + list3[k]
                # for img_name in os.listdir(img_dir):
                    # img_path = img_dir + img_name
                    # img = Image.open(img_path)
                    # img=np.array(img)
                    # height, width, channels = img.shape
                    # h1,h2,w1,w2 = f(img)
                    # img = img[h1:height-h2,w1:width-w2]
                    # img = Image.fromarray(np.uint8(img))
                    # img.save(save_dir + img_name)
                   
    img_dir = '../XZ/'
    save_dir = './XZ/'
    for img_name in os.listdir(img_dir):
        img_path = img_dir + img_name
        img = Image.open(img_path)
        img=np.array(img)
        height, width, channels = img.shape
        h1,h2,w1,w2 = f(img)
        img = img[h1:height-h2,w1:width-w2]
        img = Image.fromarray(np.uint8(img))
        img.save(save_dir + img_name)

if __name__ == '__main__':
    main()
