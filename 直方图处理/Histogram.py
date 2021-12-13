import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_hist(input_img):

    b, g, r = cv2.split(input_img)
    b_hist = np.zeros(256,dtype=np.int)
    g_hist = np.zeros(256,dtype=np.int)
    r_hist = np.zeros(256, dtype=np.int)

    for i in range(0,512):
        for j in range(0, 512):
            number = b[i,j]
            b_hist[number] = b_hist[number] + 1

    for i in range(0,512):
        for j in range(0, 512):
            number = g[i,j]
            g_hist[number] = g_hist[number] + 1

    for i in range(0,512):
        for j in range(0, 512):
            number = r[i,j]
            r_hist[number] = r_hist[number] + 1

    output_image = b_hist+g_hist+r_hist

    return output_image

def img_negative(input_img):

    b, g, r = cv2.split(input_img)
    b_reverse = 255 - b
    g_reverse = 255 - g
    r_reverse = 255 - r

    output_img = cv2.merge((b_reverse,g_reverse,r_reverse))

    return output_img

def img_threshold(input_img, threshold):

    output_img = 255*(input_img>=threshold).astype(np.int)

    return output_img

def img_equalization(input_img):

    input_img_hist = generate_hist(input_img)
    image_pro = generate_hist(input_img)/(512*512*3)

    image_accumulate_pro = np.cumsum(image_pro)

    output_img_hist = np.round(image_accumulate_pro*255).astype(np.int)
    output_img = np.zeros([512,512,3]).astype(np.int)

    for i in range(0,512):
        for j in range(0,512):
            for k in range(0,3):
                output_img[i,j,k] = output_img_hist[input_img[i,j,k]]

    output_img_hist = generate_hist(output_img)

    return input_img_hist, output_img_hist, output_img

# 加载图片
input_img_RGB_astronaut = cv2.imread('./astronaut.jpg')
input_img_RGB_cameraman = cv2.imread('./cameraman.jpg')

indexs = np.arange(0,256,1)

# 图像处理
input_img_RGB_hist_astronaut,\
output_img_RGB_hist_astronaut,\
output_img_RGB_astronaut = img_equalization(input_img_RGB_astronaut)
input_img_RGB_reverse_astronaut = img_negative(input_img_RGB_astronaut)
input_img_RGB_threshold_astronaut = img_threshold(input_img_RGB_astronaut,128)

input_img_RGB_hist_cameraman,\
output_img_RGB_hist_cameraman,\
output_img_RGB_cameraman = img_equalization(input_img_RGB_cameraman)
input_img_RGB_reverse_cameraman = img_negative(input_img_RGB_cameraman)
input_img_RGB_threshold_cameraman = img_threshold(input_img_RGB_cameraman,128)

# 绘制直方图
# fig, axes = plt.subplots(nrows=2, ncols=2, facecolor='cornsilk')
# axes[0,0].bar(indexs,input_img_RGB_hist_cameraman,color = "blue",width = 1)
# axes[0,0].set_title("input_img_hist_cameraman")
# axes[0,1].bar(indexs,output_img_RGB_hist_cameraman,color = "green",width = 1)
# axes[0,1].set_title("output_img_hist_cameraman")
# axes[1,0].bar(indexs,input_img_RGB_hist_astronaut,color = "blue",width = 1)
# axes[1,0].set_title("input_img_hist_astronaut")
# axes[1,1].bar(indexs,output_img_RGB_hist_astronaut,color = "green",width = 1)
# axes[1,1].set_title("output_img_hist_astronaut")
fig, axes = plt.subplots(nrows=2, ncols=1, facecolor='cornsilk')
axes[0].bar(indexs,generate_hist(input_img_RGB_reverse_cameraman),color = "blue",width = 1)
axes[1].bar(indexs,generate_hist(input_img_RGB_reverse_astronaut),color = "green",width = 1)
# axes[1].set_title("output_img_hist_cameraman")
plt.show()

# 保存图片
# cv2.imwrite("./astronaut_img_equ.jpg", output_img_RGB_astronaut)
# cv2.imwrite("./astronaut_img_reverse.jpg", input_img_RGB_reverse_astronaut)
# cv2.imwrite("./astronaut_img_threshold.jpg", input_img_RGB_threshold_astronaut)
# cv2.imwrite("./cameraman_img_equ.jpg", output_img_RGB_cameraman)
# cv2.imwrite("./cameraman_img_reverse.jpg", input_img_RGB_reverse_cameraman)
# cv2.imwrite("./cameraman_img_threshold.jpg", input_img_RGB_threshold_cameraman)
