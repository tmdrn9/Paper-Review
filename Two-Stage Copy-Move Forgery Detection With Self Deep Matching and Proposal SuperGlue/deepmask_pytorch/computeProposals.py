
import models
import numpy as np
import time
import cv2
from PIL import Image
import torch
import albumentations as A
from tools.InferDeepMask import Infer
from utils.load_helper import load_pretrain
# from dataset import CustomDataset

import pandas as pd
import os
from torch.utils.data import DataLoader
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/home/mmc/Paper/Copy-move/two_stage/'))))
# sys.path.append(['/home/mmc/Paper/Copy-move/two_stage'])
from dataset import CustomDataset
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__") and callable(models.__dict__[name]))

# parser = argparse.ArgumentParser(description='PyTorch DeepMask/SharpMask evaluation')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='DeepMask', choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names) + ' (default: DeepMask)')
# parser.add_argument('--resume', default='exps/deepmask/train/model_best.pth.tar',
#                     type=str, metavar='PATH', help='path to checkpoint')
# parser.add_argument('--img', default='data/testImage.jpg',
#                     help='path/to/test/image')
# parser.add_argument('--nps', default=10, type=int,
#                     help='number of proposals to save in test')
# parser.add_argument('--si', default=-2.5, type=float, help='initial scale')
# parser.add_argument('--sf', default=.5, type=float, help='final scale')
# parser.add_argument('--ss', default=.5, type=float, help='scale step')
data_dir = '/home/mmc/Paper/Copy-move/'
data_folder = '/home/mmc/Paper/Copy-move/dataset/' # dataset/
val_dir = os.path.join(data_dir, data_folder, 'val2014/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
aug = A.Compose([
                 A.Resize(512, 512)
                    ])
val_annotation = pd.read_csv(os.path.join(val_dir, 'annotation.csv'))
valid_dataset =  CustomDataset(val_annotation, val_dir, aug)

valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)                    


def range_end(start, stop, step=1):
    return np.arange(start, stop+step, step)

def p_generation_test(si = -2.5, sf = .5, ss= .5, nps=5, img=None, model_path='/home/mmc/Individual/HandsomeMJ/deepmask-pytorch/pretrained/deepmask/DeepMask.pth.tar', arch_path='DeepMask'):
    # global args
    # args = parser.parse_args()
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup Model
    from collections import namedtuple
    Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'batch'])
    config = Config(iSz=160, oSz=56, gSz=112, batch=1)  # default for training

    model = (models.__dict__[arch_path](config))
    model = load_pretrain(model, model_path)
    model = model.eval().to(device)

    scales = [2**i for i in range_end(si, sf, ss)]
    meanstd = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    infer = Infer(nps=nps, scales=scales, meanstd=meanstd, model=model, device=device)

    # print('| start'); tic = time.time()
    # im = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
    # print(img.shape)
    h, w = img.shape[2:]
    # print(h,w)
    # h, w = im.shape[:2]
    # img = np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0).astype(np.float32)
    # img = torch.from_numpy(img / 255.).to(device)
    # print(img.shape)
    # img = torch.from_numpy(img)
    infer.forward(img)
    masks, scores = infer.getTopProps(.2, h, w)
    # toc = time.time() - tic
    pred_box_r = []
    # print('masks.shape : ', masks.shape)
    # print('| done in %05.3f s' % toc)
    # print(masks.shape)
    for i in range(masks.shape[2]):
        # print('img',img[0].shape)
        # res = im[:,:,::-1].copy().astype(np.uint8)
        res = img[0].cpu().numpy()
        # print("res", res.shape)
        res = np.transpose(res,(2,1,0))
        res[:, :, 2] = masks[:, :, i] * 255 + (1 - masks[:, :, i]) * res[:, :, 2]

        mask = masks[:, :, i].astype(np.uint8)
        
        contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        #print(contours)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        cnt_max_id = np.argmax(cnt_area)
        contour = contours[cnt_max_id]
        polygons = contour.reshape(-1, 2)

        predict_box = cv2.boundingRect(polygons)
        predict_rbox = cv2.minAreaRect(polygons)
        rbox = cv2.boxPoints(predict_rbox)
        pred_box = [[predict_box[0],predict_box[1]], [predict_box[0]+predict_box[2], predict_box[1]+predict_box[3]]]
        pred_box_r.append(pred_box)
        # print("return is .. ", pred_box_r)    
    print("return is .. ", (pred_box_r))
    return torch.tensor(pred_box_r)    

def p_generation(si = -2.5, sf = .5, ss= .5, img_path='data/sampleetest.jpg', nps=3,  model_path='/home/mmc/Individual/HandsomeMJ/deepmask-pytorch/pretrained/deepmask/DeepMask.pth.tar', arch_path='DeepMask'):
    # global args
    # args = parser.parse_args()
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_box_r = []
    # Setup Model
    from collections import namedtuple
    Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'batch'])
    config = Config(iSz=160, oSz=56, gSz=112, batch=2)  # default for training

    model = (models.__dict__[arch_path](config))
    model = load_pretrain(model, model_path)
    model = model.eval().to(device)

    scales = [2**i for i in range_end(si, sf, ss)]
    meanstd = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    infer = Infer(nps=nps, scales=scales, meanstd=meanstd, model=model, device=device)

    print('| start'); tic = time.time()
    im = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
    print(im.shape)
    h, w = im.shape[:2]
    img = np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0).astype(np.float32)
    img = torch.from_numpy(img / 255.).to(device)
    print(img.shape)
    infer.forward(img)
    masks, scores = infer.getTopProps(.2, h, w)
    toc = time.time() - tic
    print('| done in %05.3f s' % toc)
    print('masks : ', masks.shape)
    print('im : ', im[:,:,::-1].copy().astype(np.uint8).shape)
    for i in range(masks.shape[2]):
        res = im[:,:,::-1].copy().astype(np.uint8)
        res[:, :, 2] = masks[:, :, i] * 255 + (1 - masks[:, :, i]) * res[:, :, 2]

        mask = masks[:, :, i].astype(np.uint8)
        
        contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        #print(contours)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        cnt_max_id = np.argmax(cnt_area)
        contour = contours[cnt_max_id]
        polygons = contour.reshape(-1, 2)

        predict_box = cv2.boundingRect(polygons)
        predict_rbox = cv2.minAreaRect(polygons)
        rbox = cv2.boxPoints(predict_rbox)
        # pr = np.array([[predict_box[0],predict_box[1]], [predict_box[2],pred_box[3]]])
        # pred_box.append(pr)
        # print(pred_box)
        # pred_box.append(predict_box)
        # print('Segment Proposal Score: {:.3f}'.format(scores[i]))

        res = cv2.rectangle(res, (predict_box[0], predict_box[1]),
                      (predict_box[0]+predict_box[2], predict_box[1]+predict_box[3]), (0, 255, 0), 3)
        res = cv2.polylines(res, [np.int0(rbox)], True, (0, 255, 255), 3)
        pred_box = [[predict_box[0],predict_box[1]], [predict_box[0]+predict_box[2], predict_box[1]+predict_box[3]]]
        pred_box_r.append(pred_box)
        print(pred_box)
        cv2.imshow('Proposal', res)
        cv2.waitKey(0)
    # cv2.imshow('',res)
    # cv2.waitKey(0)
    
    print("result is : ..", pred_box_r)
    

# if __name__ == '__main__':
#     main()
# p_generation()

for data, _ in valid_loader:
    data = data.to(device)
    p_generation_test(img=data)
