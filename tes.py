import cv2
import numpy as np

def Calculate_Euclidean_Distance(p, q):
    p = np.array(p)
    q = np.array(q)
    return np.linalg.norm(p-q)

def DetectMousePoint(event,x,y,flags,params):
    global current_point, point_selected, previous_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_point=(x,y)
        point_selected = True
        previous_points = np.array([[x,y]],dtype=np.float32)

#calcOpticalFlowPyrLK(previous_frame, next_frame, previous_points, search_window_size,max_level, termination_criteria, no_of_iterations, epsilon_value)
optFlowparams = dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1))

cv2.namedWindow("Main Window")
cv2.setMouseCallback("Main Window", DetectMousePoint)

point_selected = False
current_point = ()
previous_points = np.array([[]],dtype=np.float32)
colorforresults = (255,0,0)
colorforresultstext = (0,0,255)

# get the pixel per cm (camera specific)
PIXEL_PER_CM = 15
# get the FPS of the camera
FPS = 29

video = cv2.VideoCapture(0)
print(video.get(3),video.get(4)) 

_,initialFrame = video.read()
initialGrayFrame = cv2.cvtColor(initialFrame,cv2.COLOR_BGR2GRAY)
resultMaskImage = np.zeros_like(initialFrame)


while video.isOpened():
    _, frames = video.read()
    grayimg = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    resultMaskText = np.zeros_like(initialFrame)

    if point_selected:

        new_points, status, error = cv2.calcOpticalFlowPyrLK(initialGrayFrame,grayimg,previous_points,None,**optFlowparams)
        
        for i,(new,old) in enumerate(zip(new_points,previous_points)):
            new_X,new_Y = new.ravel()
            old_X,old_Y = old.ravel()

            distance = Calculate_Euclidean_Distance((old_X,old_Y),(new_X,new_Y))
            speed = "  " + str(round(distance / PIXEL_PER_CM * FPS,2)) + " cm/s"
            #km/h = cm/s x 0.036

            resultMaskImage = cv2.line(resultMaskImage,(old_X,old_Y),(new_X,new_Y),colorforresults,2)
            frames = cv2.circle(frames,(new_X,new_Y),5,colorforresults,2)
            resultMaskText = cv2.putText(resultMaskText, speed, (new_X,new_Y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, colorforresultstext)

        frames = cv2.add(frames,resultMaskImage)
        result = cv2.add(frames,resultMaskText)
        initialGrayFrame = grayimg.copy()
        previous_points = new_points
    else:
        result=frames
    

    cv2.imshow("Main Window",result)

    key = cv2.waitKey(1)

    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
