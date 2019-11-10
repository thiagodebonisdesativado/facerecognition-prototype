import cv2

classifierFace = cv2.CascadeClassifier("classifiers\\haar\\haarcascade_frontalface_default.xml")
eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50)
eigenface.read("classifiers\\classifierEigenface.yml")

height, width = 220, 220
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    isConnected, image = webcam.read()

    if isConnected:
        imageConverted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceDetected = classifierFace.detectMultiScale(imageConverted, scaleFactor=1.5, minSize=(30, 30))

        for (x, y, w, h) in faceDetected:
            faceImage = cv2.resize(imageConverted[y:y + h, x:x + w], (width, height))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            faceId, accuracy = eigenface.predict(faceImage)
            print(faceId)
            cv2.putText(image, '', (x, y + (h + 30)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        cv2.imshow("FACE", image)
        if cv2.waitKey(1) == ord('q'):
            break