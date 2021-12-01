import cv2
import numpy as np
import math
from PIL import Image

def pre_faces():
    faces = np.zeros((400, 112, 92), dtype=np.float64)
    j = 0
    for num in range(1, 41):
        path = "./pre_faces/s" + str(num)

        for i in range(1, 11):
            imagePath = path + '/' + str(i) + '.pgm'
            image_array = cv2.imread(imagePath, -1)

            with open(path+ '/' + str(i) + '.txt', "r") as file:
                data = file.read().split(' ')
                dy = float(data[3]) - float(data[1])
                dx = float(data[2]) - float(data[0])
                angle = math.atan2(dy, dx)*180. / math.pi
                left_eye_x = float(data[0])
                left_eye_y = float(data[1])
                right_eye_x = float(data[2])
                right_eye_y = float(data[3])
                eye_center = ((float(data[0]) + float(data[2]))//2, (float(data[1]) + float(data[3])) // 2)

                # 眼间距
                eye_distance = math.sqrt((eye_center[0] - left_eye_x)**2 + (eye_center[1] - left_eye_y)**2)
                # 旋转
                rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
                rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))

                center = (48, 53)
                # 平移
                mat_translation = np.float32([[1,0,center[0]-eye_center[0]],[0,1,center[1]-eye_center[1]]])
                dst = cv2.warpAffine(rotated_img, mat_translation, (image_array.shape[1], image_array.shape[0]))  # 变换函数

                # 缩放
                factor = 17/eye_distance
                dst = cv2.resize(dst, None, fx=factor , fy=factor)
                factor_center =(center[0]*factor, center[1]*factor)

                # 裁剪
                left = factor_center[0] - 30
                right = factor_center[0] + 30
                top = factor_center[1] - 40
                down = factor_center[1] + 50
                dst = Image.fromarray(dst)
                cropped_img = dst.crop((left, top, right, down ))
                
                img = cv2.cvtColor(np.asarray(cropped_img),cv2.COLOR_RGB2BGR)
                img = img[:,:,0]
                faces[j] = cv2.resize(img, (92, 112))
                # plt.imshow(faces[i], cmap='gray')
                # plt.show()
                j += 1

    return faces  


def pre_one_faces():
    image_array = cv2.imread("test-mine.pgm", -1)
    with open('6.txt', "r") as file:
        data = file.read().split(' ')
        dy = float(data[3]) - float(data[1])
        dx = float(data[2]) - float(data[0])
        angle = math.atan2(dy, dx)*180. / math.pi
        left_eye_x = float(data[0])
        left_eye_y = float(data[1])
        right_eye_x = float(data[2])
        right_eye_y = float(data[3])
        eye_center = ((float(data[0]) + float(data[2]))//2, (float(data[1]) + float(data[3])) // 2)

        # 眼间距
        eye_distance = math.sqrt((eye_center[0] - left_eye_x)**2 + (eye_center[1] - left_eye_y)**2)
        # 旋转
        rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
        rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))

        center = (48, 53)
        # 平移
        mat_translation = np.float32([[1,0,center[0]-eye_center[0]],[0,1,center[1]-eye_center[1]]])
        dst = cv2.warpAffine(rotated_img, mat_translation, (image_array.shape[1], image_array.shape[0]))  # 变换函数

        # 缩放
        factor = 17/eye_distance
        dst = cv2.resize(dst, None, fx=factor , fy=factor)
        factor_center =(center[0]*factor, center[1]*factor)

        # 裁剪
        left = factor_center[0] - 30
        right = factor_center[0] + 30
        top = factor_center[1] - 40
        down = factor_center[1] + 50
        dst = Image.fromarray(dst)
        cropped_img = dst.crop((left, top, right, down ))
                
        img = cv2.cvtColor(np.asarray(cropped_img),cv2.COLOR_RGB2BGR)
        img = img[:,:,0]
        img = cv2.resize(img, (92, 112))
        # plt.imshow(faces[i], cmap='gray')
        # plt.show()

    return img
