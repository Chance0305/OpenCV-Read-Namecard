from tkinter import filedialog
from tkinter import *
import cv2 as cv
import numpy as np
import sys
import pytesseract

def reorderPts(pts):
    idx = np.lexsort((pts[:, 1], pts[:, 0]))  # 칼럼0 -> 칼럼1 순으로 정렬한 인덱스를 반환
    pts = pts[idx]  # x좌표로 정렬

    if pts[0, 1] > pts[1, 1]:
        pts[[0, 1]] = pts[[1, 0]]

    if pts[2, 1] < pts[3, 1]:
        pts[[2, 3]] = pts[[3, 2]]

    return pts

filename = Tk().dirName=filedialog.askopenfilename() # GUI 통해 이미지 탐색
img = cv.imread(filename)

if img is None:
    print('Image load failed!')
    sys.exit()

# 출력 영상 설정
dw, dh = 720, 400
imgQuad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)
dstQuad = np.array([[0, 0], [0, dh], [dw, dh], [dw, 0]], np.float32)
dst = np.zeros((dh, dw), np.uint8)

# 입력 영상 전처리
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # grayscale
_, img_bin = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) # 이진화

contours, _ = cv.findContours(img_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # 외곽선 검출
# img_contours = cv.drawContours(img, contours, -1, (0,255,0), 3) # (debugging)

for pts in contours:
    # 너무 작은 객체는 제외
    if cv.contourArea(pts) < 1000:
        continue

    # 외곽선 근사화
    approx = cv.approxPolyDP(pts, cv.arcLength(pts, True)*0.02, True)

    if not cv.isContourConvex(approx) or len(approx) != 4:
        continue

    cv.polylines(img, [approx], True, (0, 255, 0), 2, cv.LINE_AA) # 외곽선 폴리곤 렌더
    imgQuad = reorderPts(approx.reshape(4, 2).astype(np.float32))

    pers = cv.getPerspectiveTransform(imgQuad, dstQuad)
    dst = cv.warpPerspective(img, pers, (dw, dh), flags=cv.INTER_CUBIC)

    dst_rgb = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
    print(pytesseract.image_to_string(dst_rgb, lang="hangul+eng"))


# cv.imshow('img', img)
# cv.imshow('img_gray', img_gray)
# cv.imshow('img_bin', img_bin)
# cv.imshow('img_contours', img_contours)
# cv.imshow('img_contours_filtered', img_contours_filtered)
cv.imshow('dst', dst)
cv.waitKey()
cv.destroyAllWindows()