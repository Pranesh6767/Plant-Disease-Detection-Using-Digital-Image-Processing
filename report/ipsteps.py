import cv2
import numpy as np

filename = '1_g8.jpg'
main_img = cv2.imread(filename)
img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
#except:
#    return "Invalid"

#Preprocessing


gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#cv2.imwrite('greayscale.png',gs)

blur = cv2.GaussianBlur(gs, (25,25),0)
#cv2.imwrite('blur.png',blur)

ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#cv2.imwrite('otsus.png',im_bw_otsu)

kernel = np.ones((25,25),np.uint8)
closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('closing.png',closing)


#Shape features
contours, _ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.imwrite('contours.png',contours)
cnt = contours[0]

current_frame = main_img
filtered_image = closing/255

#Elementwise Multiplication of range bounded filtered_image with current_frame
current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 0] = np.multiply(current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 0], filtered_image) #B channel
current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 1] = np.multiply(current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 1], filtered_image) #G channel
current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 2] = np.multiply(current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 2], filtered_image) #R channel

cv2.imwrite('final_output.png',current_frame)
