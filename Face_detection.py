""" here in this code I have done face detection and detection of other features of face using haar cascades files."""


import cv2
import dlib
# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    detector = dlib.get_frontal_face_detector();
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    coords = []
    
    featur = detector(gray_img)
    # drawing rectangle around the feature and labeling it
    if classifier=="faceCascade":
        for f in featur:
            landmark = predictor(gray_img, f)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

    for (x, y, w, h) in features:
         cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
         cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
         coords = [x, y, w, h]
    return coords


# Method to detect the features
def detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords)==4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Passing roi, classifier, scaling factor, Minimum neighbours, color, label text
        coords = draw_boundary(roi_img, eyeCascade, 1.1, 5, color['red'], "Eye")
        coords = draw_boundary(roi_img, noseCascade, 1.1, 5, color['green'], "Nose")
        coords = draw_boundary(roi_img, mouthCascade, 1.1, 10, color['white'], "Mouth")
    return img


# Loading classifiers
faceCascade = cv2.CascadeClassifier('C:/Users/Rajneesh/Downloads/Compressed/FaceDetection-master/frontalface.xml')
eyesCascade = cv2.CascadeClassifier('C:/Users/Rajneesh/Desktop/eyes.xml')
noseCascade = cv2.CascadeClassifier('C:/Users/Rajneesh/Downloads/Compressed/Nose25x15.1/Nariz.xml')
mouthCascade = cv2.CascadeClassifier('C:/Users/Rajneesh/Downloads/Compressed/Mouth25x15.1/Mouth.xml')

video_capture = cv2.VideoCapture(0)

frame_width = int(video_capture.get(9))
frame_height = int(video_capture.get(16))

while True:
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade)
    # Writing processed image in a new window
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()
