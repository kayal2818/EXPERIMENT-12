# EXPERIMENT-12
## Face Detection using Haar Cascades with OpenCV and Matplotlib
## Name : KAYALVIZHI.V
## Reg no : 212225040182
## Aim
To write a Python program using OpenCV to perform the following image manipulations:
i) Extract ROI from an image.
ii) Perform face detection using Haar Cascades in static images.
iii) Perform eye detection in images.
iv) Perform face detection with label in real-time video from webcam.

## Software Required
Anaconda - Python 3.7 or above
OpenCV library (opencv-python)
Matplotlib library (matplotlib)
Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)
## Algorithm
## I) Load and Display Images
Step 1: Import necessary packages: numpy, cv2, matplotlib.pyplot
Step 2: Load grayscale images using cv2.imread() with flag 0
Step 3: Display images using plt.imshow() with cmap='gray'
## II) Load Haar Cascade Classifiers
Step 1: Load face and eye cascade XML files
## III) Perform Face Detection in Images
Step 1: Define a function detect_face() that copies the input image
Step 2: Use face_cascade.detectMultiScale() to detect faces
Step 3: Draw white rectangles around detected faces with thickness 10
Step 4: Return the processed image with rectangles
## IV) Perform Eye Detection in Images
Step 1: Define a function detect_eyes() that copies the input image
Step 2: Use eye_cascade.detectMultiScale() to detect eyes
Step 3: Draw white rectangles around detected eyes with thickness 10
Step 4: Return the processed image with rectangles
## V) Display Detection Results on Images
Step 1: Call detect_face() or detect_eyes() on loaded images
Step 2: Use plt.imshow() with cmap='gray' to display images with detected regions highlighted
## VI) Perform Face Detection on Real-Time Webcam Video
Step 1: Capture video from webcam using cv2.VideoCapture(0)
Step 2: Loop to continuously read frames from webcam
Step 3: Apply detect_face() function on each frame
Step 4: Display the video frame with rectangles around detected faces
Step 5: Exit loop and close windows when ESC key (key code 27) is pressed
Step 6: Release video capture and destroy all OpenCV windows

## Program
```
# Name : Iniya E
# Reg no : 212224230096

import cv2
import matplotlib.pyplot as plt
%matplotlib inline

withglass = cv2.imread('/content/Screenshot 2025-11-13 212818.png', 0)
group = cv2.imread('/content/Screenshot 2025-11-13 213045.png', 0)

plt.imshow(withglass, cmap='gray')
plt.title("With Glasses")
plt.show()

plt.imshow(group, cmap='gray')
plt.title("Group Image")
plt.show()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty():
    raise IOError("Error loading face cascade XML file")
if eye_cascade.empty():
    raise IOError("Error loading eye cascade XML file")

def detect_face(img, scaleFactor=1.1, minNeighbors=5):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

def detect_eyes(img):
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

result_withglass_faces = detect_face(withglass)
plt.imshow(result_withglass_faces, cmap='gray')
plt.title("Faces in With Glasses Image")
plt.show()

result_group_faces = detect_face(group)
plt.imshow(result_group_faces, cmap='gray')
plt.title("Faces in Group Image")
plt.show()

result_withglass_eyes = detect_eyes(withglass)
plt.imshow(result_withglass_eyes, cmap='gray')
plt.title("Eyes in With Glasses Image")
plt.show()

result_group_eyes = detect_eyes(group)
plt.imshow(result_group_eyes, cmap='gray')
plt.title("Eyes in Group Image")
plt.show()
```
## Output
<img width="489" height="516" alt="image" src="https://github.com/user-attachments/assets/b1b0c5a9-87fd-4eb9-9ecf-5f2c45bdac96" />
<img width="582" height="513" alt="image" src="https://github.com/user-attachments/assets/28039820-1d32-4956-80b8-bd4cfdd3abe5" />
<img width="562" height="524" alt="image" src="https://github.com/user-attachments/assets/8dc6fa47-359c-45ec-8429-7ae2b0f506c0" />
<img width="585" height="529" alt="image" src="https://github.com/user-attachments/assets/7cf6a7b2-b60e-4e53-bda0-9aa589b56f5b" />
<img width="511" height="513" alt="image" src="https://github.com/user-attachments/assets/6bbac208-ee56-4be6-813c-19e4d632310d" />
<img width="476" height="521" alt="image" src="https://github.com/user-attachments/assets/5f0f4372-786f-49ed-ad36-1d2ba55ad8e5" />
<img width="432" height="516" alt="image" src="https://github.com/user-attachments/assets/7fba0293-9395-4e25-ae1f-b5bedc4e1128" />
<img width="432" height="516" alt="image" src="https://github.com/user-attachments/assets/05ae8f8e-ab58-46d4-b455-895e5f53f6d4" />
<img width="553" height="532" alt="image" src="https://github.com/user-attachments/assets/544e3aba-f3a4-40c1-a8df-0ac6aeafef4f" />
<img width="715" height="533" alt="image" src="https://github.com/user-attachments/assets/1c6fd2f6-d1bb-4ce3-a35b-fd139311355d" />

## Result
Thus executed successfully


