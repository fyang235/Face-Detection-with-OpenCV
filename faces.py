import numpy as np
import cv2		# OpenCV 工具包
import pickle	# 用于打包可解压标签词典

# 从指定的文件位置获取haar filter 函数
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

# 指定识别器的类型并从训练好的模型中读取出来
recognizer = cv2.face.LBPHFaceRecognizer_create() 
recognizer.read('recognizers/trainner.yml')

# 创建视频捕捉器
cap = cv2.VideoCapture(0)
labels = {}

# 从训练时构建的词典中读取标签的信息并置换key和value
with open('labels.pickle', 'rb') as f:
    org_labels = pickle.load(f)
    labels = {v:k for k, v in org_labels.items()}

# 视频图像抓取    
while(True):
    # Capture frame-by-frame 抓取的图像存放在frame中，彩色
	# 将彩图转化为灰度图，并用函数获取所有人脸的坐标
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)
	
	# 获得所有的人脸坐标
    for (x, y, w, h) in faces:
        # print the cord of face
        print(x, y, w, h)
        
        # save gray face img 保存灰度图
        roi_gray = gray[y:y+h, x:x+w]        
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)
        
        # sace color face img 保存彩图
        img_item_color = "my-image-color.png"
        roi_color = frame[y:y+h, x:x+w, :]
        cv2.imwrite(img_item_color, roi_color)
        
        # draw a rectangle 绘制人脸矩形框
        color = (255, 0, 0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        
        # implement recognizer 用训练好的识别器对人脸进行实时识别
        id_, conf = recognizer.predict(roi_gray)
        if conf >=45 :# and conf <= 85:
            print(id_)
            print(labels[id_])
        
        # front 在视频中标注标签名称
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
		
    # 眼部识别，与上面的内容相似    
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)
    for (ex, ey, ew, eh) in eyes:       
        # draw a rectangle
        color = (0, 255, 0) #BGR
        stroke = 2
        eend_cord_x = ex + ew
        eend_cord_y = ey + eh
        cv2.rectangle(frame, (ex, ey), (eend_cord_x, eend_cord_y), color, stroke)
        
    # Display the resulting frame 显示图片和退出方法
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()