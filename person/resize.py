import cv2
img=cv2.imread('0.jpg')
img = cv2.resize(src=img, dsize=(768, 1024), fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
cv2.imwrite('0.jpg', img)