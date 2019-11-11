import cv2
import os

classifierFace = cv2.CascadeClassifier("classifiers\\haar\\haarcascade_frontalface_default.xml")
eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50)
eigenface.read("classifiers\\classifierEigenface.yml")

height, width = 220, 220
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def getImageName(faceId):
    paths = [os.path.join('images\\samples', i) for i in os.listdir('images\\samples')]
    imageName = ''

    for pathImage in paths:
        imageName = os.path.split(pathImage)[1].split('.')[0]

    return imageName.upper()

while True:
    isConnected, image = webcam.read()

    if isConnected:
        imageConverted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceDetected = classifierFace.detectMultiScale(imageConverted, scaleFactor=1.5, minSize=(30, 30))

        for (x, y, w, h) in faceDetected:
            faceImage = cv2.resize(imageConverted[y:y + h, x:x + w], (width, height))
            faceId, accuracy = eigenface.predict(faceImage)
            faceName = getImageName(faceId)

            if faceName == 'THIAGO':
                color = (0, 255, 51)
            else:
                color = (0, 0, 255)

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f'ID: {faceId}', (x, y + (h + 30)), cv2.FONT_HERSHEY_PLAIN, 1, color)
            cv2.putText(image, f'NAME: {faceName}', (x, y + (h + 50)), cv2.FONT_HERSHEY_PLAIN, 1, color)
            cv2.putText(image, f'ACCURACY: {accuracy}', (x, y + (h + 70)), cv2.FONT_HERSHEY_PLAIN, 1, color)

        cv2.imshow("FACE", image)
        if cv2.waitKey(1) == ord('q'):
            break
