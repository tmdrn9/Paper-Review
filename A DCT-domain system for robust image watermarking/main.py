import cv2
import matplotlib.pyplot as plt
import numpy as np
import functions as fc

alpha = 0.35
m = 4500

X = np.random.randn(1000, m)
image = cv2.imread("./boat.PNG", cv2.IMREAD_GRAYSCALE)
watermark_img = fc.putWatermark(image, X[400, :])

# flg2
plt.imshow(image, cmap='gray')
plt.show()
plt.imshow(watermark_img, cmap='gray')
plt.show()

# flg3
wm = fc.dct(watermark_img)
dd = fc.detector(wm)
t = fc.makeZone_flatten(wm)
plt.axhline((alpha/(2*m)) * np.sum(abs(t)), color='lightgray', linestyle='--' )
plt.plot(dd)
plt.show()

# #flg5
lowfilterImg = cv2.GaussianBlur(watermark_img, (3, 3), 0)
plt.imshow(lowfilterImg, cmap='gray')
plt.show()
wm = fc.dct(lowfilterImg)
dd = fc.detector(wm)
t = fc.makeZone_flatten(wm)
plt.axhline((alpha/(3*m)) * np.sum(abs(t)), color='lightgray', linestyle='--' )
plt.plot(dd)
plt.show()

#fig14
multi = [X[600, :], X[800, :], X[900, :], X[200, :]]
multiWatermark_img = fc.putmultiwatermark(watermark_img, multi)
plt.imshow(multiWatermark_img, cmap='gray')
plt.show()
wm = fc.dct(multiWatermark_img)
dd = fc.detector(wm)
t = fc.makeZone_flatten(wm)
plt.axhline((alpha/(2*len(t))) * np.sum(abs(t)), color='lightgray', linestyle='--' )
plt.plot(dd)
plt.show()