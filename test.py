import cv2


smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


video_capture = cv2.VideoCapture(0)  # Use 0 for webcam or provide the path to a video file

while True:
    # Read the current frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("====gray", gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=24, minSize=(45, 45))
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Perform smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=24, minSize=(18, 18))

        # If a smile is detected, draw a rectangle around the face and display text
        if len(smiles) > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Smiling', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Smiling Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    video_capture.release()
    cv2.destroyAllWindows()
