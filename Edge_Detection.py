import cv2
import numpy as np
import matplotlib.pyplot as plt

def sub_Sobel(input_img,threshold):

    # 计算边界模板
    sobelX = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    sobelY = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    m, n = np.shape(input_img)
    gx = 0
    gy = 0
    # 初始化与原图像相同数组
    output_edge = np.zeros_like(input_img)
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            gx = np.sum(input_img[i - 1:i + 2, j - 1:j + 2] * sobelX)
            gy = np.sum(input_img[i - 1:i + 2, j - 1:j + 2] * sobelY)
            output_edge[i][j] = abs(gx) + abs(gy)
            if output_edge[i][j] >= int(255*threshold):
                output_edge[i][j] = 255
            else:
                output_edge[i][j] = 0

    return output_edge

def Sobel(input_img,threshold):

    b, g, r = cv2.split(input_img)
    b_sobel = sub_Sobel(b,threshold)
    g_sobel = sub_Sobel(g,threshold)
    r_sobel = sub_Sobel(r,threshold)

    output_edge = cv2.merge((b_sobel, g_sobel, r_sobel))

    return output_edge

def sub_Prewitt(input_img,threshold):

    prewitX = [[-1, -1, -1],
               [0, 0, 0],
               [1, 1, 1]]
    prewitY = [[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]]
    m, n = np.shape(input_img)
    output_edge = input_img[:]
    gx = 0
    gy = 0
    for i in range(m - 3):
        for j in range(n - 3):
            gx = np.sum(input_img[i:i + 3, j:j + 3] * prewitX)
            gy = np.sum(input_img[i:i + 3, j:j + 3] * prewitY)
            output_edge[i][j] = abs(gx) + abs(gy)
            if output_edge[i][j] >= int(255*threshold):
                output_edge[i][j] = 255
            else:
                output_edge[i][j] = 0

    return output_edge

def Prewitt(input_img,threshold):

    b, g, r = cv2.split(input_img)
    b_Prewitt = sub_Prewitt(b,threshold)
    g_Prewitt = sub_Prewitt(g,threshold)
    r_Prewitt = sub_Prewitt(r,threshold)

    output_edge = cv2.merge((b_Prewitt,g_Prewitt,r_Prewitt))

    return output_edge

def sub_Roberts(input_img,threshold):

    # 获取边界模板
    roberX = roberY = np.zeros((2, 2))
    roberX[0][0] = -1
    roberX[1][1] = 1
    roberY[0][1] = -1
    roberY[1][0] = 1
    # 计算图像大小
    m, n = np.shape(input_img)
    gx = 0
    gy = 0
    output_edge = input_img[:]
    for i in range(m):
        for j in range(n):
            gx = np.sum(input_img[i:i + 2, j:j + 2] * roberX)
            gy = np.sum(input_img[i:i + 2, j:j + 2] * roberY)
            output_edge[i][j] = abs(gx) + abs(gy)
            if output_edge[i][j] >= int(255*threshold):
                output_edge[i][j] = 255
            else:
                output_edge[i][j] = 0

    return output_edge

def Roberts(input_img,threshold):

    b, g, r = cv2.split(input_img)
    b_Roberts = sub_Roberts(b,threshold)
    g_Roberts = sub_Roberts(g,threshold)
    r_Roberts = sub_Roberts(r,threshold)

    output_edge = cv2.merge((b_Roberts,g_Roberts,r_Roberts))
    return output_edge

def sub_Laplacian(input_img,threshold):

    Laplacian = [[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]]

    m, n = np.shape(input_img)
    output_edge = input_img[:]
    gx = 0
    gy = 0
    for i in range(m - 3):
        for j in range(n - 3):
            gx = np.sum(input_img[i :i + 3, j :j + 3] * Laplacian)
            gy = np.sum(input_img[i :i + 3, j :j + 3] * Laplacian)
            output_edge[i][j] = abs(gx) + abs(gy)

            if (output_edge[i][j] >= int(255*threshold)):
                output_edge[i][j] = 255
            else:
                output_edge[i][j] = 0

    return output_edge

def Laplacian(input_img,threshold):

    b, g, r = cv2.split(input_img)
    b_Laplacian = sub_Laplacian(b,threshold)
    g_Laplacian = sub_Laplacian(g,threshold)
    r_Laplacian = sub_Laplacian(r,threshold)

    output_edge = cv2.merge((b_Laplacian, g_Laplacian, r_Laplacian))

    return output_edge

# 加载图片
input_image_RGB = cv2.imread('./cameraman.jpg')

# 阈值设定
threshold = 0.3

# 图片处理
input_image_RGB_sobel = Sobel(input_image_RGB,threshold)
input_image_RGB_prewitt = Prewitt(input_image_RGB,threshold)
input_image_RGB_roberts = Roberts(input_image_RGB,threshold)
input_image_RGB_laplacian = Laplacian(input_image_RGB,threshold)

fig, axes = plt.subplots(nrows=2, ncols=2, facecolor='cornsilk')
axes[0,0].imshow(input_image_RGB_sobel)
axes[0,0].set_title("sobel")
axes[0,1].imshow(input_image_RGB_prewitt)
axes[0,1].set_title("prewitt")
axes[1,0].imshow(input_image_RGB_roberts)
axes[1,0].set_title("roberts")
axes[1,1].imshow(input_image_RGB_laplacian)
axes[1,1].set_title("laplacian")
plt.show()

# 保存图片
cv2.imwrite("./cameraman_sobel.jpg",input_image_RGB_sobel)
cv2.imwrite("./cameraman_prewitt.jpg",input_image_RGB_prewitt)
cv2.imwrite("./cameraman_roberts.jpg",input_image_RGB_roberts)
cv2.imwrite("./cameraman_laplacian.jpg",input_image_RGB_laplacian)
