#P_FA : https://blog.naver.com/hjh4119/222087541573
#PCE :

import src.Functions as Fu
import cv2 as cv
import src.Filter as Ft
import numpy as np
import src.maindir as md
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt

# extracting Fingerprint from same size images in a path
# Images = sorted(glob.glob('cameraImage/train/Sony-NEX-7/*.JPG', recursive=True))
# Images = Images[:110]
# #테스트용으로 쓸 데이터 제거
# Images.remove('cameraImage/train/Sony-NEX-7\\(Nex7)1.JPG')
# Images.remove('cameraImage/train/Sony-NEX-7\\(Nex7)2.JPG')
# Images.remove('cameraImage/train/Sony-NEX-7\\(Nex7)3.JPG')
# Images.remove('cameraImage/train/Sony-NEX-7\\(Nex7)4.JPG')
# Images.remove('cameraImage/train/Sony-NEX-7\\(Nex7)5.JPG')
# Images.remove('cameraImage/train/Sony-NEX-7\\(Nex7)6.JPG')
# Images.remove('cameraImage/train/Sony-NEX-7\\(Nex7)7.JPG')
# Images.remove('cameraImage/train/Sony-NEX-7\\(Nex7)8.JPG')
# Images.remove('cameraImage/train/Sony-NEX-7\\(Nex7)9.JPG')
# Images.remove('cameraImage/train/Sony-NEX-7\\(Nex7)10.JPG')
#
# #color 채널별로 fingerprint 추출하고 파일로 저장
# RP,_,_ = gF.getFingerprint(Images)
# sigmaRP1, sigmaRP2, sigmaRP3 = np.std(RP[:,:,0]),np.std(RP[:,:,1]),np.std(RP[:,:,2])
# Fingerprint0 = Fu.WienerInDFT(RP[:,:,0], sigmaRP1)
# Fingerprint1 = Fu.WienerInDFT(RP[:,:,1], sigmaRP2)
# Fingerprint2 = Fu.WienerInDFT(RP[:,:,2], sigmaRP3)
# Fingerprint=np.array([Fingerprint0,Fingerprint1,Fingerprint2])
# sigmaRP=np.array([sigmaRP1, sigmaRP2, sigmaRP3])

# To save RP in a '.mat' file:
# import scipy.io as sio
# sio.savemat('Fingerprint_3channel.mat', {'RP': RP, 'sigmaRP': sigmaRP, 'Fingerprint': Fingerprint})

#저장한 mat파일 불러오기
mat_file=sio.loadmat('Fingerprint_3channel.mat')
Fingerprint = mat_file['Fingerprint']

#각도 바꿔주는 함수
def changeHue(img,i): #바꿔줄이미지,바꿀 각도
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV) #hsv로 변환
    a = img[:, :, 0] #hue채널만 추출

    #본래 이미지가 uint형에 0~180범위의 값을 갖기 때문에 변환각도가 +,-일때 나눠서 계산
    if i > 0:
        # 만약 a+i가 181이면 1이 되야하므로 180을 빼줘야함
        a = np.where(a > 180-i,  (a + i)-180, a+i)
    elif i < 0:
        # 만약 a+i가 -1이면 180이 되야하므로 181을 더해줘야함
        a = np.where(a < -i, 181 + (a + i), a + i)

    img[:, :, 0] = a
    img = cv.cvtColor(img, cv.COLOR_HSV2RGB) #rgb로 변환
    return img

#바뀐 각도 찾는 함수
def detector(changeImage):
    im_corr = []
    for i in tqdm(range(180)):
        #i도씩 각도 변환
        img=changeHue(changeImage, i)
        #fingerprint 추출
        Noisex0 = Ft.NoiseExtractFromImage(img, sigma=2.)
        corr = []

        #color 채널별로 pce구하고 더해주기
        for i in range(3):
            Noisex = Fu.WienerInDFT(Noisex0[:,:,i], np.std(Noisex0[:,:,i]))
            C = Fu.crosscorr(Noisex,np.multiply(img[:,:,i],  Fingerprint[i]))
            det,_ = md.PCE(C)
            corr.append(det['PCE'])
        im_corr.append(sum(corr))
    result=np.argmax(np.array(im_corr))
    return result #result==0이면 변환 안 한 사진

# 원본이미지
im=cv.imread('cameraImage/train/Sony-NEX-7\\(Nex7)1.JPG')
origin=cv.cvtColor(im, cv.COLOR_BGR2RGB)

#각도 바꾼 이미지
change=changeHue(origin,-60)

seta=detector(change)
if seta==0:
    print("색조 수정이 안 된 이미지 입니다")
else:
    print("{0}도 색조 수정이 된 이미지 입니다".format(-seta))


plt.subplot(1, 3, 1)
plt.imshow(origin)
plt.subplot(1, 3, 2)
plt.imshow(change)
plt.subplot(1, 3, 3)
re_change=changeHue(change,seta)
plt.imshow(re_change)
plt.show()