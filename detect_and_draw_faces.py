import cv2

cascade_path = "face_cascades/haarcascade_profileface.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_face(frame):
    """
    detect human faces in image using haar-cascade
    Args:
        frame:
    Returns:
    coordinates of detected faces
    """
    #detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]])
    faces = face_cascade.detectMultiScale(frame, 1.1, 2, 0, (20, 20))
    return faces


def draw_faces(frame, faces):
    """
    draw rectangle around detected faces
    Args:
        frame:
        faces:
    Returns:
    face drawn processed frame
    """
    for (x, y, w, h) in faces:
        xA = x
        yA = y
        xB = x + w
        yB = y + h
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    return frame