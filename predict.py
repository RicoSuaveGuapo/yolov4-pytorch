#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import cv2
import os

# yolo = YOLO()

# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = yolo.detect_image(image)
#         r_image.show()

def images2video(image_dir_path, output_name, fps):
    img_array = []
    filenames = [os.path.join(image_dir_path,filename) for filename in os.listdir(image_dir_path) if filename.endswith('.jpg')]
    filenames.sort()
    path = filenames[0]
    img = cv2.imread(path)
    h, w, _ = img.shape
    size = (w, h)
    for filename in filenames:
        img = cv2.imread(filename)
        img_array.append(img)
    out = cv2.VideoWriter(f'/home/rico-li/Job/yolov4-pytorch/predictions/Kangaroo/{output_name}.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


# path = 'VOCdevkit/VOC2007/JPEGImages'
# img_path = [os.path.join(path, img) for img in os.listdir(path)]

# for img in img_path:
#     image = Image.open(img)
#     result = yolo.detect_image(image)
#     result.save(os.path.join('predictions',os.path.basename(img)))
#     print(f'Prediction on {os.path.basename(img)} is saved')


fps = 22
pred_path = 'predictions/Kangaroo'
images2video(image_dir_path=pred_path, output_name='Prediction', fps=fps)