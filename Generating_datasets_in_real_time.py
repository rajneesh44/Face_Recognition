import cv2

def generate_dataset(img,id,img_id):
    cv2.imwrite("D:/images/user."+str(id)+"."+str(img_id)+".jpg",img)

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img,text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


# Method to detect the features
def detect(img, faceCascade,img_id):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['white'], "New_person")
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords)==4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        user_id =4
        generate_dataset(roi_img,user_id,img_id)
    return img

# Loading face classifier
faceCascade = cv2.CascadeClassifier('C:/Users/Rajneesh/Downloads/Compressed/FaceDetection-master/frontalface.xml')
video_capture = cv2.VideoCapture(0)
img_id=0 

while True:
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    img = detect(img, faceCascade,img_id)
    cv2.imshow("face detection", img)
    img_id+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
