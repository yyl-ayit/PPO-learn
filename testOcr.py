import Myclass as My
import cv2
import pytesseract
if __name__ == '__main__':
    gray_image = cv2.imread('testImg/cut0.jpg')
    print(pytesseract.image_to_string(gray_image, config='--psm 6'))