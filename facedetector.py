import cv2
import dlib
import imutils
from imutils import face_utils
import winsound
from scipy.spatial import distance

detector=dlib.get_frontal_face_detector()
predict=dlib.shape_predictor("C:/Users/kushal asn/Downloads/shape_predictor_68_face_landmarks.dat")
def eye_aspect_ratio(Eye):
    A=distance.euclidean(Eye[1],Eye[5])
    B=distance.euclidean(Eye[2],Eye[4])
    C=distance.euclidean(Eye[0],Eye[3])
    ear=(A+B)/(2*C)
    return ear


thresh=0.30
frame_rate=30
duration=1000
frequency=2500
(lstart,lend)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart,rend)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0
while(True):
    ret,frame=cap.read()
    frame=imutils.resize(frame,width=500)
    if ret:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        subjects=detector(gray,0)
        for subject in subjects:
            shape=predict(gray,subject)
            shape=face_utils.shape_to_np(shape)
            leye=shape[lstart:lend]
            reye=shape[rstart:rend]
            lear=eye_aspect_ratio(leye)
            rear=eye_aspect_ratio(reye)
            lhull=cv2.convexHull(leye)
            rhull=cv2.convexHull(reye)
            ear=(lear+rear)/2
            if(ear<thresh):
                flag+=1
                print(flag)
                if(flag>frame_rate):
                    winsound.Beep(frequency,duration)
                    print("drowsy alert")
            else:
                flag=0
    cv2.imshow("Frame",frame)
    if(cv2.waitKey(1)==ord("q")):
        break
    
cv2.destroyAllWindows()
cap.release()


