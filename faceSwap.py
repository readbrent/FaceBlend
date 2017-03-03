import cv2
import os
import random
import numpy as np

def findFace(haarPath, testImage):
    faceCascade = cv2.CascadeClassifier(haarPath)

    inMemImage = cv2.imread(testImage)
    greyScale = cv2.cvtColor(inMemImage, cv2.COLOR_BGR2GRAY)

    #Detect the faces
    faces = faceCascade.detectMultiScale(
        greyScale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    #Select a random face
    THE_CHOSEN_ONE = random.choice(faces)
    fudgeFactor = 10

    # Art is not safe
    crop_img = inMemImage[(THE_CHOSEN_ONE[1] - fudgeFactor):(THE_CHOSEN_ONE[1] + THE_CHOSEN_ONE[3] + fudgeFactor),
                        (THE_CHOSEN_ONE[0] - fudgeFactor):(THE_CHOSEN_ONE[0] + THE_CHOSEN_ONE[2] + fudgeFactor)]

    return crop_img

# Places 'face' onto a hardcoded part of the boohbah and then displays the image
def placeOntoBoohbah(face, boohbahPath):
    boohbahImage = cv2.imread(boohbahPath)
    #Hardcoded coordinates for the boohbah's head
    y = 010
    x = 125
    h = 175
    w = 175
    #subBoohbah = boohbahImage[y:(y+h), x:(x+w)]
    headCenter = ((x + h) / 2, (y + w) / 2)

    faceMask = np.zeros(face.shape, face.dtype)
    width =  len(face[0])
    height =  len(face[1])
    poly = np.array( [[0,0], [width, 0], [width, height], [0, height]], np.int32)
    cv2.fillPoly(faceMask, [poly], (255, 255, 255))
    # Scale the image to be the same size as the boohbah
    resizedFace = cv2.resize(face, (w, h))
    outputImage = cv2.seamlessClone(face, boohbahImage, faceMask, headCenter, cv2.NORMAL_CLONE)

    cv2.imshow("Final Monstrosity", outputImage)
    cv2.waitKey(0)



def BeierNeely():
    print "jk"

def quickHull():
    print "jk"

def sweepline():
    print "jk"

def main():
    cd = os.getcwd()
    haarPath = cd + "/haarcascade_frontalface_default.xml"
    testImage = cd + "/faceTest.jpg"
    firstFace = findFace(haarPath, testImage)
    boohbah = cd + "/boohbah.jpg"
    placeOntoBoohbah(firstFace, boohbah)

if __name__ == "__main__":
    main()

