import cv2
import numpy as np

video = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame = video.read()

    if ret == False: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if i == 20:
        bgGray = gray
    if i > 20:
        dif = cv2.absdiff(gray, bgGray)
        _, thresh = cv2.threshold(dif, 40, 255, cv2.THRESH_BINARY)
        # find contours and hierarchy to get
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # draw contours image
        # cv2.drawContours(frame, cnts, -1, (0, 255, 0), 2)

        # Using cv2.imshow() method
        # Displaying the image
        # cv2.imshow('dif', dif)
        cv2.imshow('thresh', thresh)

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 9000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    i = i + 1

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#
video.relace()