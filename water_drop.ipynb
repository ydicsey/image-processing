import cv2
import numpy as np

# 读取彩色图像
image_name = 'image3.jpg'
img = cv2.imread(image_name)

# 将图像转换为YUV颜色空间
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# 对Y通道进行直方图均衡化
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# 将图像转换回BGR颜色空间
img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# 将图像转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(1, 1))  # 创建自适应直方图均衡化器
equalized = clahe.apply(gray_img)  # 应用自适应直方图均衡化

gray = cv2.cvtColor(img_equalized, cv2.COLOR_BGR2GRAY)

# 使用Otsu's二值化方法
# threshold 二值化圖形
_, threshold = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def Sobel_edge_detection(f):
    grad_x = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize = 3)
    grad_y = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize = 3)
    magnitude = abs(grad_x) + abs(grad_y)
    g = np.uint8(np.clip(magnitude, 0, 255))
    ret, g = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return g
#sobel edge
img2 = Sobel_edge_detection(threshold)



    
# 轮廓提取
contours, hierarchy = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 形状和平均值检查
valid_droplets = []  # 存储符合要求的水滴
total_area = 0       # 累计水滴面积

for contour in contours:
    # 计算形状特征
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    (x, y, w, h) = cv2.boundingRect(contour)
    
    # 执行形状检查
    if area > 1 and area < 1000 and w < 150 and h < 150:
        # 将符合要求的水滴添加到列表中
        valid_droplets.append(contour)
        total_area += area

# 绘制轮廓和计算平均值
output_image = np.zeros_like(img2)
cv2.drawContours(output_image, valid_droplets, -1, 255, thickness=cv2.FILLED)
#cv2.fillPoly(output_image, valid_droplets, (255, 255, 255))


# 显示结果
kernel = np.ones((1,1),np.uint8)
result = cv2.morphologyEx(output_image,cv2.MORPH_OPEN ,kernel)

#res_1 = cv2.inpaint(img, space, 5, cv2.INPAINT_NS)
res_2 = cv2.inpaint(img, result, 5, cv2.INPAINT_TELEA)




# 显示原始图像和均衡化后的图像
cv2.imshow('Original Image', img)
cv2.imshow('gray', gray)
cv2.imshow('equalized', equalized)
cv2.imshow('threshold', threshold)
cv2.imshow("img2", img2)
#cv2.imshow('space', space)
cv2.imshow('Output Image', output_image)
cv2.imshow("result", result)
#cv2.imshow("res_1", res_1)
cv2.imshow("res_2", res_2)

#cv2.imshow("res_3", res_3)
#cv2.imshow("res_4", res_4)

cv2.waitKey(0)
cv2.destroyAllWindows()
