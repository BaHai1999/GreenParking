import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from joblib import dump, load
from sklearn import svm

def listBoundingRect(img):
    MIN_AREA = 3000
    MAX_AREA = 8000
    rs = []
    # giam nhieu
    gray = cv2.GaussianBlur(img.copy(), (7, 7), 0)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 255)
    # sobelx = cv2.Sobel(gray, cv2.CV_8U, 0, 1)
    ret2, th2 = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Closing is reverse of Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 4))
    close_morh = th2.copy()
    # cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel, close_morh)
    cv2.morphologyEx(close_morh, cv2.MORPH_CLOSE, kernel, close_morh)
    contours, _ = cv2.findContours(close_morh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if(w/h < 1.8 and (w/h > 0.9) and (MIN_AREA <= w*h <= MAX_AREA)):
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            rs = non_maximum_suppresstion(rs, [x, y, w, h], threshold=0.3)
            # rs.append([x, y, w, h])
    img = cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    # plt.imshow(img), plt.show()
    return rs

# rect = (x, y, w, h)
def iou(rect1, rect2):
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
    interArea = max(0,x2-x1)*max(0,y2-y1)
    s_Rect1 = rect1[2]*rect1[3]
    s_Rect2 = rect2[2]*rect2[3]
    iou = interArea / (s_Rect1 + s_Rect2 - interArea)
    return iou
def non_maximum_suppresstion(rs, rect, threshold=0):
    for box in rs:
        IOU = iou(box, rect)
        if(IOU > threshold):
            return rs
    rs.append(rect)
    return rs
def crop(img, x, y, w, h):
    img_copy = img.copy()
    return img_copy[y:y+h, x:x+w]
def sort(chars):
    chars = sorted(chars, key=lambda x: x[1], reverse=False)
    for i in range(len(chars)-1):
        for j in range(i+1, len(chars)):
            if(-5 <= chars[j][1] - chars[i][1] <= 5):
                if(chars[j][0] < chars[i][0]):
                    chars[j], chars[i] = chars[i], chars[j]
    return chars

def OCR(img, box):
    _box = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    # _box = cv2.GaussianBlur(_box, (5,5), 0)
    ret, mask = cv2.threshold(_box, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret, mask = cv2.threshold(_box, 110, 255, cv2.THRESH_BINARY)
    # nen den chu trang
    mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    chars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if(box.shape[0]/6 <= h <= box.shape[0]):
            if (1.5 < h/w < 5):
                non_maximum_suppresstion(chars, [x, y, w, h])
    chars = sort(chars)
    clf = load('svm_model_green_parking')
    X_predict = []
    for x, y, w, h in chars:
        x_predict = crop(mask, x, y, w, h)
        x_predict = cv2.resize(x_predict, (30, 60))
        x_predict = np.reshape(x_predict, 30*60)
        X_predict.append(x_predict)
        box = cv2.rectangle(box, (x, y), (x+w, y+h), (255, 0, 0), 1)
        char1 = crop(mask, x, y, w, h)
        cv2.imshow('.', char1), cv2.waitKey(0)
    X_predict = np.array(X_predict)
    y_predict = clf.predict(X_predict)
    plt.imshow(box), plt.show()
    return y_predict

if __name__ == '__main__':
    img =  cv2.imread('dataset/25.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = listBoundingRect(img.copy())
    # print(output)
    for x, y, w, h in output:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        box = img[y:y + h, x:x + w, :].copy()
        y_predict = OCR(img, box)
        text = ''
        print(y_predict)
        for y_output in y_predict:
            if(y_output > 9):
                text = text + chr(y_output)
            else:
                text = text + str(y_output)
        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # plt.imshow(box, 'gray'), plt.show()
    #59 S114883
    plt.imshow(img), plt.show()
