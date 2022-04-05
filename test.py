import cv2


cap = cv2.VideoCapture(0)

prev = None
while cap.isOpened():
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    h,w,ch = frame.shape

    if prev is not None:
        flow = cv2.calcOpticalFlowFarneback(prev,gray,None,0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        print(flow.shape)
    else : 
        prev = gray

    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(33) == 27:
        break

cap.release()
cv2.destroyAllWindows()