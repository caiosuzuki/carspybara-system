from darkflow.net.build import TFNet
import cv2
import numpy as np

options = {"pbLoad": "built_graph/yolo-mor-4c.pb", "metaLoad": "built_graph/yolo-mor-4c.meta", "threshold": 0.3, "gpu": 0.5}

tfnet = TFNet(options)

cap = cv2.VideoCapture("/home/takashi/exp/entradas/mor2.mp4")
counter = 0
fps = cap.get(cv2.CAP_PROP_FPS)


ret, frame = cap.read()
height, width, channels = frame.shape
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('count.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))


while (cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # define ROI da contagem
        roithickness = 0.025
        roixmin = 0
        roixmax = width
        roiymin = int(height/2 - roithickness*height)
        roiymax = int(height/2 + roithickness*height)
        cv2.rectangle(frame, (roixmin,roiymin), (roixmax,roiymax), (0,0,255), 2)

        boxesInfo = tfnet.return_predict(frame)

        for boxInfo in boxesInfo:
            # checa se centro da BB atual está na ROI, se sim, aumenta contador
            xcenter = (boxInfo["topleft"]["x"]+boxInfo["bottomright"]["x"])/2
            ycenter = (boxInfo["topleft"]["y"]+boxInfo["bottomright"]["y"])/2
            if(roixmin<=xcenter<=roixmax and roiymin<=ycenter<=roiymax):
                counter += 1
            cv2.rectangle(frame, (boxInfo["topleft"]["x"], boxInfo["topleft"]["y"]), (boxInfo["bottomright"]["x"], boxInfo["bottomright"]["y"]), (255,0,0), 2)
        
        cv2.putText(frame, 'counter:'+str(counter), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
        out.write(frame)  
    else:
        break

cap.release()
cv2.destroyAllWindows()

#cv2.waitKey(0)
    

