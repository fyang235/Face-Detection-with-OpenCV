import os				# 系统包，主要用为文件操作
from PIL import Image	# pillow 主要用于从路径中打开图片和resize图片
import numpy as np
import cv2
import pickle			# 用于压缩dict

# 获取当前文件所在路径，构建图像所在路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, 'images')

# 获得haar filter函数和初始化人脸识别器
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() 

# 初始化存储和中间变量
current_id = 0
label_dict = {}
x_train = []
y_labels = []

# 从当前路径中获得路径，文件
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
			# 对每一张图片获得其路径和标签，标签即图片所在文件夹的名称，用‘-’替换空格，统一使用lowercase
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(' ','-').lower()
			
			# 构建标签dict
            if not label in label_dict:
                label_dict[label] = current_id
                current_id += 1
            id_ = label_dict[label] 
			
			# 读取图片并转化为灰度图
            pil_img = Image.open(path).convert('L')
            # resize 图片，并转化为np数组，‘uint8’类型
            size = (550, 550)
            final_img = pil_img.resize(size, Image.ANTIALIAS)
            img_array = np.array(final_img, 'uint8')
            
			# 采用1.1的较密集的框，扫描人脸
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=15)
            print('id_, label, file, faces: ',id_,label,file,faces)
			
			# 获取人脸图片
            for x,y,w,h in faces:
                roi = img_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
     
print(label_dict) 
print(y_labels) 

# 保存标签dict         
with open('labels.pickle', 'wb') as f:
    pickle.dump(label_dict, f)

# 训练识别器并保存    
recognizer.train(x_train, np.array(y_labels))
recognizer.save('recognizers/trainner.yml')    
    
    
    
    
    
    
    
    
    
    