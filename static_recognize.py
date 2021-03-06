import cv2
thres = 0.55 # Threshold to detect object

# cap = cv2.VideoCapture(1)
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,70)

img = cv2.imread('photos/personholdingball.jpg')


classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


classIds, confs, bbox = net.detect(img,confThreshold=thres)
print(classIds,confs,bbox)

for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
        print(box)
        cv2.putText(img,classNames[classId-1].upper()+'- Conf: '+str(round(confidence,3)),(box[0]+10,box[1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        # cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
        # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
# imO = cv2.resize(img,(960,540))
cv2.imshow('Output',img)
cv2.waitKey(0)