import pytesseract
import cv2
from Myclass import My
if __name__ == '__main__':
    image = cv2.imread('testImg/main5.png')
    # print(My.gameEnd(image))
    print(pytesseract.image_to_string(image, config='--psm 11'))
