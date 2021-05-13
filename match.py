import numpy as np
import torch
import argparse
import cv2
import torch
from model import *

ap = argparse.ArgumentParser()
ap.add_argument('-p1','--path1',type=str,help='path to image 1',required=True)
ap.add_argument('-p2','--path2',type=str,help='path to image 2',required=True)
args = vars(ap.parse_args())

weightpath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
prototxtpath = 'face_detector/deploy.prototxt'


#loading the face detection model
net = cv2.dnn.readNet(weightpath,prototxtpath)
#loading the face recognition model   
model = torch.load('model')

#function to detect faces
def face_detect(net,image):
  img = cv2.resize(image,(300,300))
  blob = cv2.dnn.blobFromImage(img,1,(300,300),100)
  net.setInput(blob)
  detections=net.forward()
  for i in range(0,detections.shape[2]):
    if detections[0,0,i,2]>=0.5:
      box = detections[0,0,i,3:7]*np.array([300,300,300,300])
      (startx,starty,endx,endy)= box.astype('int')
      face = img[starty:endy,startx:endx]
      if face is not None: 
        return face
  return None

#transformations to be applied to training data for training model
trans = Compose([ToTensor(),Normalize(mean=0.4,std=0.2),Scale((240,240))])

  
#function to calculate cosine similarity between two images
def similarity(model,face1,face2):
  face1 = torch.unsqueeze(face1,0)
  face2 = torch.unsqueeze(face2,0)
  face1_f = model(face1)
  face2_f = model(face2)
  return np.cos(np.linalg.norm((face1_f-face2_f).detach().numpy()))
  
# function to give similarity between two images given paths
def sim(net,model,path1,path2):
  face1=trans(face_detect(net,cv2.imread(path1)))
  face2 = trans(face_detect(net,cv2.imread(path2)))
  s = similarity(model,face1,face2)
  if s>=0.7 :
    print('Match with confidence of similarity: %.2f %%'%(s*100))
  else:
    print('No Match with confidence of similarity : %.2f %%'%(s*100))
    

sim(net,model,args['path1'],args['path2'])
  
