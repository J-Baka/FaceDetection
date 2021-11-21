import cv2.cv2 as cv
from random import randrange


face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Basic smile detector. As long as teeth are showing it pretty much assumes a smile
smile_detector = cv.CascadeClassifier('haarcascade_smile.xml')

# Choosing an image to detect
img = cv.imread('Profile.jpg')

# If wanted to use with default live video feed keep it 0
# Otherwise insert static video file location (i.e. filename.mp4)
webcam = cv.VideoCapture(0)

# Iterate over video frames
while True:

    # read current frame
    # returns if it is successful and actual image
    success_frame_read, frame = webcam.read()

    # Precaution in case of errors. Will abort instead of infinite loop
    if not success_frame_read:
        break

    # Converting an image to grayscale (Must do)
    # Haar algorithm looks for dark to light differences, color is irrelevant
    # Also efficient, colors add more data
    frame_grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect Faces of all sizes
    face_coordinates = face_detector.detectMultiScale(frame_grayscale)

    # Drawing rectangles around the face in the provided image
    # RGB is backwards, last number is for thickness of line
    # (img, (x, y), (x + w, y + h), (B, G, R), line thickness)

    # cv.rectangle(img, (125, 37), (125 + 131, 37 + 131), (0, 255, 0), 5)

    # Looping through possible faces in an image
    for (x, y, w, h) in face_coordinates:
        cv.rectangle(frame, (x, y), (x + w, y + h),
                     (100, 200, 50),
                     5)

        # Obtain all data in the frame. N dimensional array slicing for sub-coordinates
        face = frame[y:y + h, x:x + w]

        face_grayscale = cv.cvtColor(face, cv.COLOR_BGR2GRAY)

        # slight image blur for better contrasting, how many neighbors for success pass
        smile_coordinates = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.3, minNeighbors=20)

        # Looping through searching for smiles in face box
        # for (x_, y_, w_, h_) in smile_coordinates:
        #     cv.rectangle(face, (x_, y_), (x_ + w_, y_ + h_),
        #                  (50, 50, 200),
        #                  5)

        # Labeling smile
        if len(smile_coordinates) > 0:
            cv.putText(frame, 'Great Smile', (x, y + h + 40), fontScale=1,
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))

    # Displaying the image you are importing
    cv.imshow('Face_Detector', frame)

    # Pauses execution of code for x milliseconds
    key = cv.waitKey(1)

    # Stops program if Q key is pressed. Represented in ascii number
    if key == 81 or key == 113:
        break

# Releases video capture object
webcam.release()
cv.destroyAllWindows()
