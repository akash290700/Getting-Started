import cv2

img_cascade = cv2.CascadeClassifier('C:\\Users\\jiaka\\OneDrive\\Desktop\\Face Detection Python\\Example\\haarcascade_frontalface_default.xml')

img = cv2.imread('C:\\Users\\jiaka\\OneDrive\\Desktop\\Face Detection Python\\Example\\news.jpg',1)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = img_cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5)

for x,y,w,h in faces:
    img1 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,15),2)

resized_img = cv2.resize(img1,(int(img1.shape[1]//1.2),int(img1.shape[0]//1.2)))

cv2.imshow('gray',resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
