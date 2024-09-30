

##i)Image Translation
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_img=cv2.imread("color image of flower.jpg")
input_img=cv2.cvtColor(input_img,cv2.)
plt.axis('off')
plt.imshow(input_img)
plt.show()
rows,cols,dim=input_img.shape
M=np.float32([[1,0,20],
             [0,1,50],
             [0,0,1]])
translated_img=cv2.
plt.axis('off')



##ii)Image Scaling

scaled_img=cv2.warpPerspective(input_img,M,(cols,rows))
plt.axis('off')
plt.imshow(scaled_img)
plt.show()


##iii)Image Shearing
M_x=np.float32([[1,0.2,0],
               [0,1,0],
               [0,0,1]])
M_y=np.
sheared_img_xaxis=cv2.warpPerspective(input_img,M_x,(cols,rows))
sheared_img_yaxis=
plt.axis('off')
plt.imshow(sheared_img_xaxis)
plt.show()
plt.axis
plt.show()
