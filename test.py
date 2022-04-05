import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0)

prev = None

def vec2graph(flow):
    h,w,ch = flow.shape
    x_values = flow[:,:,0].reshape(-1)
    #print(x_values.shape)
    y_values = flow[:,:,1].reshape(-1)
    plt.scatter(x_values,y_values)
    plt.show()

while cap.isOpened():
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    h,w,ch = frame.shape

    if prev is not None:
        flow = cv2.calcOpticalFlowFarneback(prev,gray,None,\
            0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        #flow is 2channel(y,x vector)
        print(flow.shape)
        vec2graph(flow)
    else : 
        prev = gray

    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(33) == 27:
        break

cap.release()
cv2.destroyAllWindows()