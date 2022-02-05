import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import pyplot as plt

def printKP_DES(kp1, des1, kp2, des2):
    print("Number of Keypoints Detected In The Image1: ", len(kp1))
    x = 0
    for dd in des1:
        for d in dd:
            x += 1
    print("Number of Descriptor In The Image1: ", x)

    print("Number of Keypoints Detected In The Image2: ", len(kp2))
    x = 0
    for dd in des2:
        for d in dd:
            x += 1
    print("Number of Descriptor In The Image1: ", x)


def test1(matches,kp1,kp2):
    print("test1")
    number_keypoints = 0
    if len(kp1) <= len(kp2):
        number_keypoints = len(kp1)
    else:
        number_keypoints = len(kp2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_points.append(m)

    numMatches = len(good_points)
    percent = (numMatches / number_keypoints) * 100

    print("\nNumber of matches ", len(matches))
    print("Good Matcher:", len(good_points))
    print("Percentage of matches :", math.floor(percent), "%\n -----------------------------\n")

    #result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, flags=2)
    #cv2.imshow("result",result)

def test2(matches,kp1,kp2):
    print("test2")
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    count_matches = 0
    for i in range(len(matches)):
        if matchesMask[i] == [1, 0]:
            count_matches += 1

    number_keypoints = 0
    if len(kp1) <= len(kp2):
        number_keypoints = len(kp1)
    else:
        number_keypoints = len(kp2)

    print("Number of matches ", count_matches)
    print("Percentage :" , math.floor(count_matches / number_keypoints * 100) ,"% \n -----------------------")
    #res = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    #cv2.imshow("result", res)



img1 = cv2.imread('Images/AuthorizedDataset/2F.png',0)
img2 = cv2.imread('Images/AuthorizedDataset/7F.png',0)

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)

#cv2.imshow("img1", cv2.resize(img1,None, fx=0.5 , fy=0.5))
#cv2.imshow("img2", cv2.resize(img2,None, fx=0.5 , fy=0.5))

"""
#Apply Clustering

Z1 = img1.reshape((-1,3))
# convert to np.float32
Z1 = np.float32(Z1)

Z2 = img2.reshape((-1,3))
# convert to np.float32
Z2 = np.float32(Z2)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 20
ret1,label1,center1=cv2.kmeans(Z1,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
ret2,label2,center2=cv2.kmeans(Z2,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center1 = np.uint8(center1)
center2 = np.uint8(center2)

res1 = center1[label1.flatten()]
res21 = res1.reshape((img1.shape))

res2 = center2[label2.flatten()]
res22 = res2.reshape((img2.shape))
"""

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

printKP_DES(kp1,des1, kp2,des2)  #print Kep points And Desecreptors

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

#bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
#extract_bow = cv2.BOWImgDescriptorExtractor(sift, flann)

#voc = bow_kmeans_trainer.cluster()
#extract_bow.setVocabulary( voc )


#test1(matches,kp1,kp2)
test1(matches,kp1,kp2)


cv2.waitKey(0)
cv2.destroyAllWindows()



