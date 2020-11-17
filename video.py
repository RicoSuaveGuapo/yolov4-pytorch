#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
yolo = YOLO()
# 调用摄像头
capture=cv2.VideoCapture('/home/rico-li/Job/1st-DL-CVMarathon_Rico/Kangaroo.mp4')

# fps = 0.0
# while(True):
#     t1 = time.time()
#     # 读取某一帧
#     ref,frame=capture.read()
#     # 格式转变，BGRtoRGB
#     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     # 转变成Image
#     frame = Image.fromarray(np.uint8(frame))

#     # 进行检测
#     frame = np.array(yolo.detect_image(frame))

#     # RGBtoBGR满足opencv显示格式
#     frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

#     fps  = ( fps + (1./(time.time()-t1)) ) / 2
#     print("fps= %.2f"%(fps))
#     frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow("video",frame)


#     c= cv2.waitKey(1) & 0xff 
#     if c==27:
#         capture.release()
#         break


fps = 0.0
frame_index = 0
while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame = np.array(yolo.detect_image(frame))

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if frame_index < 10:
        cv2.imwrite(f'predictions/Kangaroo/kangaroo_000{frame_index}.jpg', frame)
    elif frame_index < 100:
        cv2.imwrite(f'predictions/Kangaroo/kangaroo_00{frame_index}.jpg', frame)
    elif frame_index < 1000:
        cv2.imwrite(f'predictions/Kangaroo/kangaroo_0{frame_index}.jpg', frame)
    else:
        cv2.imwrite(f'predictions/Kangaroo/kangaroo_{frame_index}.jpg', frame)
    frame_index += 1
    
    if not ref:
        break
