
from darkflow.net.build import TFNet
import cv2
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold':0.3,
    'gpu' : 0.7 }


tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(3)]


capture = cv2.VideoCapture(0)


capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    stime = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    if ret:
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            
            label = result['label']
           # if label == 'laptop':
          #      label='Laptop now $999'
          #  elif label == 'person':
         #       label='Awesome Audience'
         #   elif label == 'mouse':
         #       label = 'Mouse 50% OFF'
         #   elif label == 'cell phone':
         #       label = 'Phone belongs to Noel'
         #   elif label == 'remote':
        #        label = 'Phone belongs to Noel'

                
            confidence = result['confidence'] 
#             text = '{}: {:.0f}%'.format(label,confidence * 100)
            text = '{}'.format(label)  #,confidence * 100)

            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyALLWindows()

