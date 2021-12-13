import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_gaussian_noise(input_img, mean, var):

 image = np.array(input_img / 255, dtype=float)
 noise = np.random.normal(mean, var ** 0.5, image.shape)
 noised_img = image + noise
 noised_img = np.uint8(noised_img * 255)

 return noised_img

def generate_salt_pepper_noise(input_img, prob1, prob2):

 noised_img = np.zeros(input_img.shape, np.uint8)

 for i in range(input_img.shape[0]):
  for j in range(input_img.shape[1]):
   rdn = np.random.random()

   if rdn < prob1:
    noised_img[i][j] = 0
   elif rdn > prob2:
    noised_img[i][j] = 255
   else:
    noised_img[i][j] = input_img[i][j]

 return noised_img

def mean_filter(noised_img, kernel_size):

 b, g, r = cv2.split(noised_img)
 b = sub_mean_filter(b,kernel_size)
 g = sub_mean_filter(g,kernel_size)
 r = sub_mean_filter(r,kernel_size)

 output_img = cv2.merge((b,g,r))

 return output_img

def sub_mean_filter(noised_img, kernel_size):

 noised_image_cp = np.copy(noised_img)  # 输入图像的副本
 filter_template = np.ones((kernel_size, kernel_size))  # 空间滤波器模板

 pad_num = int((kernel_size - 1) / 2)  # 输入图像需要填充的尺寸

 noised_image_cp = np.pad(noised_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)  # 填充输入图像

 m, n = noised_image_cp.shape  # 获取填充后的输入图像的大小

 output_img = np.copy(noised_image_cp)  # 输出图像

 # 空间滤波
 for i in range(pad_num, m - pad_num):
  for j in range(pad_num, n - pad_num):
    output_img[i, j] = np.sum(
     filter_template * noised_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1]) / (
                                   kernel_size ** 2)

 output_img = output_img[pad_num:m - pad_num, pad_num:n - pad_num]  # 裁剪

 return output_img

def median_filter(noised_img, kernel_size):

 b, g, r = cv2.split(noised_img)
 b = sub_median_filter(b, kernel_size)
 g = sub_median_filter(g, kernel_size)
 r = sub_median_filter(r, kernel_size)
 output_img = cv2.merge((b, g, r))

 return output_img

def sub_median_filter(noised_img, kernel_size):

 noised_image_cp = np.copy(noised_img)  # 输入图像的副本
 filter_template = np.ones((kernel_size, kernel_size))  # 空间滤波器模板

 pad_num = int((kernel_size - 1) / 2)  # 输入图像需要填充的尺寸

 noised_image_cp = np.pad(noised_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)  # 填充输入图像

 m, n = noised_image_cp.shape  # 获取填充后的输入图像的大小

 output_img = np.copy(noised_image_cp)  # 输出图像

 # 空间中值滤波
 for i in range(pad_num, m - pad_num):
  for j in range(pad_num, n - pad_num):
   output_img[i, j] = np.median(
    filter_template * noised_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1])

 output_img = output_img[pad_num:m - pad_num, pad_num:n - pad_num]  # 裁剪

 return output_img

# 加载图片
input_image = cv2.imread("./cameraman.jpg")

# 参数设定
mean = 0
var = 0.01
prob1 = 0.1
prob2 = 1-prob1
kernel_size = 5

#图像处理
output_image_gaussian = generate_gaussian_noise(input_image,mean,var)
output_image_salt_pepper = generate_salt_pepper_noise(input_image,prob1,prob2)
output_image_gaussian_mean_filter = mean_filter(output_image_gaussian,kernel_size)
output_image_salt_pepper_mean_filter = mean_filter(output_image_salt_pepper,kernel_size)
output_image_gaussian_median_filter = median_filter(output_image_gaussian,kernel_size)
output_image_salt_pepper_median_filter = median_filter(output_image_salt_pepper,kernel_size)

# 保存图片
cv2.imwrite("./cameraman_gaussian.jpg",output_image_gaussian)
cv2.imwrite("./cameraman_salt_pepper.jpg",output_image_salt_pepper)
cv2.imwrite("./cameraman_salt_pepper_mean_filter.jpg",output_image_salt_pepper_mean_filter)
cv2.imwrite("./cameraman_gaussian_mean_filter.jpg",output_image_gaussian_mean_filter)
cv2.imwrite("./cameraman_salt_pepper_median_filter.jpg",output_image_salt_pepper_median_filter)
cv2.imwrite("./cameraman_gaussian_median_filter.jpg",output_image_gaussian_median_filter)


