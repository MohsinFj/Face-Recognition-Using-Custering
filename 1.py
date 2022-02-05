import numpy as np
import cv2
import glob

AuthorizedPath = "C:/Users/Mohammad/PycharmProjects/ttest/Images/Authorized/*.*"
UnauthorizedPath = "C:/Users/Mohammad/PycharmProjects/ttest/Images/Unauthorized/*.*"

AuthorizedSavedPath = "C:/Users/Mohammad/PycharmProjects/ttest/Images/AuthorizedDataset/"
UnauthorizedSavedPath = "C:/Users/Mohammad/PycharmProjects/ttest/Images/UnauthorizedDataset/"

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

pathOfIamges = glob.glob(AuthorizedPath)
print(len(pathOfIamges))

sift = cv2.xfeatures2d.SIFT_create()

xx = 0
yy = 0
for bb,file in enumerate(pathOfIamges, 1):
    #print(bb, file)
    features = list()
    img = cv2.imread(file)
    #colorSpace  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(img , 1.1, 4)
    for (x, y, w, h) in face:
        detect = cv2.rectangle(img , (x,y),(x+w, y+h), (255,0,0))
        roi_color = detect[y:y-5 + h, x:x-5 + w]
        cv2.imwrite(AuthorizedSavedPath +'{}F.png'.format(bb), roi_color)
        kp, des = sift.detectAndCompute(roi_color,None)
        """
        for k in kp:
            print(k)
            xx += 1
        
        for d in des:
            for dd in d:
                yy += 1
            print(d)
        """
        np.savetxt("Images/AuthorizedDataset/{}Feature.txt".format(bb), des, delimiter='  ', fmt="%s")
        siftResult = cv2.drawKeypoints(roi_color, kp, None)
        cv2.imwrite(AuthorizedSavedPath + '{}FS.png'.format(bb), siftResult)
        #print(xx)
        #print(yy)

    cv2.imshow("Images" , detect)
    cv2.imwrite(AuthorizedSavedPath +"/{}.png".format(bb), img)

    cv2.waitKey(1000)


cv2.destroyAllWindows()
print("Finish ")


