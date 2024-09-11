import cv2
def detect_single_face(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))[0]
    return x, y, w, h
image_path = 'datasets/Prasanna/gt/00009502.png'
img = cv2.imread(image_path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# x, y, w, h = detect_single_face(img, face_cascade)
x, y, w, h = [306, 192, 454, 454]
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
print(x, y, w, h)
cv2.imwrite('faces_1.jpg', img)