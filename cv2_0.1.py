import cv2 as cv
img = cv.imread('/home/pi/test.jpg')
#cv.imshow('123_img',img)
print('img', img.shape)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
resize_img = cv.resize(gray_img, dsize=(500,400))
cv.imshow('resize_img',resize_img)
print('resize_img', resize_img.shape)

#cv.imwrite('gray.jpg',gray_img)
cv.waitKey(0)
cv.destroyAllWindows()