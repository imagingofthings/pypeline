# Necessary imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Creating our image with coordinates (500,500)
img = np.zeros((500, 500, 3), dtype="uint8")

# Change the color of the image to white
img[:] = (255,255,255)

# Drawing a red rectangle. Negative parameter -1
# indicates that we want to draw filled shape
cv2.rectangle(img, (100,350), (400,400), (0,0,255), (-1))

cv2.imshow("name",img)
cv2.waitKey() 
cv2.destroyAllWindows()