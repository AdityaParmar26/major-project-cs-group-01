import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yaml")

def recognize_face(frame_orginal, faces):
    """
    recognize human faces using LBPH features
    Args:
        frame_orginal:
        faces:
    Returns:
        label of predicted person
    """
    predict_label = []
    predict_conf = []
    for x, y, w, h in faces:
        frame_orginal_grayscale = cv2.cvtColor(frame_orginal[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
        cv2.imshow("cropped", frame_orginal_grayscale)
        predict_tuple = recognizer.predict(frame_orginal_grayscale)
        a, b = predict_tuple
        predict_label.append(a)
        predict_conf.append(b)
        print("Predition label, confidence: " + str(predict_tuple))
    return predict_label


def put_label_on_face(frame, faces, labels):
    """
    draw label on faces
    Args:
        frame:
        faces:
        labels:
    Returns:
        processed frame
    """
    i = 0
    for x, y, w, h in faces:
        cv2.putText(frame, str(labels[i]), (x, y), font, 1, (255, 255, 255), 2)
        i += 1
    return frame