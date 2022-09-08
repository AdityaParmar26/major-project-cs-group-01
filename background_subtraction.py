import cv2

def background_subtraction(previous_frame, frame_resized_grayscale, min_area):
    """
    This function returns 1 for the frames in which the area
    after subtraction with previous frame is greater than minimum area
    defined.
    Thus expensive computation of human detection face detection
    and face recognition is not done on all the frames.
    Only the frames undergoing significant amount of change (which is controlled min_area)
    are processed for detection and recognition.
    """
    frameDelta = cv2.absdiff(previous_frame, frame_resized_grayscale)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Thresh",thresh)
    cv2.waitKey(200)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cv2.imshow("Thresh dialtetd",thresh)
    cv2.waitKey(200)
    countours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp = 0
    for c in countours:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) > min_area:
            temp = 1
    return temp