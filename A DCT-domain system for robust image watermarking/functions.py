from Common.dct2d import dct, idct
import numpy as np

def makeZone_flatten(img):
    img = img.copy()
    zone0 = img[100:150, 80:150].flatten()
    zone1 = img[80:100, 100:150].flatten()
    full_zone = np.append(zone0, zone1)
    return full_zone

def Watermark(zone, xx):
    global alpha
    zero = np.zeros_like(zone)
    zero = zone + alpha*abs(zone)*xx
    return zero

def inputWm(wm, tp):
    wm[100:150, 80:150] = tp[:3500].reshape(50, 70)
    wm[80:100, 100:150] = tp[3500:].reshape(20, 50)
    return wm

def detector(dct_img):
    zz = makeZone_flatten(dct_img)
    z = np.zeros(1000)
    for jj in range(1000):
        z[jj] += np.dot(X[jj, :], zz)/m
    return z

def putWatermark(img, x):
    indct = dct(img)
    indctt = indct.copy()
    crop_zone = makeZone_flatten(indctt)
    water_markzone = Watermark(crop_zone, x)
    return idct(inputWm(indct, water_markzone))

def putmultiwatermark(img, x):
    indct = dct(img)
    indctt = indct.copy()
    crop_zone = makeZone_flatten(indctt)
    water_markzone = Watermark(crop_zone, x[0])
    for i in range(1, len(x)):
        water_markzone = Watermark(water_markzone, x[i])
    return idct(inputWm(indct, water_markzone))
