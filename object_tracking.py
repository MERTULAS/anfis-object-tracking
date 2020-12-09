from cv2 import cv2
import imutils


class ObjectDetection:
    @staticmethod
    def init_hardware():
        web_cam = cv2.VideoCapture(0)
        return web_cam

    @staticmethod
    def read_capture(vid):
        _, frame = vid.read()
        return frame

    @staticmethod
    def image_processing(frame):
        green_lower = (29, 86, 6)
        green_upper = (64, 255, 255)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, green_lower, green_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        if len(contour) > 0:
            c = max(contour, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            m = cv2.moments(c)
            center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
            if radius > 15:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (255, 0, 0), -1)
        return frame

    @staticmethod
    def show(frame):
        cv2.imshow("FRAME", frame)


Object = ObjectDetection()
video = Object.init_hardware()
if __name__ == "__main__":
    while True:
        cam = Object.read_capture(video)
        cam = Object.image_processing(cam)
        Object.show(cam)
        k = cv2.waitKey(25) & 0xFF
        if k == 27 or k == ord("q"):
            break
    cv2.destroyAllWindows()
