import cv2
from imutils.object_detection import non_max_suppression

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_people(frame):
    """
    detect humans using HOG descriptor
    Args:
        frame:
    Returns:
        processed frame
    """
    # features : https://pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
    # winstride, step size in x and y direction  : https://pyimagesearch.com/wp-content/uploads/2014/10/sliding_window_example.gif
    # padding sizes in x and y to ignore
    (rects, _) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.06)
    rects = non_max_suppression(rects, probs=None, overlapThresh=0.5)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame