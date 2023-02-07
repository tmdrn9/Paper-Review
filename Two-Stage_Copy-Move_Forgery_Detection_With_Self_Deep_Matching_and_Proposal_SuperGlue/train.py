import os
import time
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from util import *
from dataset import CustomDataset
from SelfDeepMatching import self_deep_matching
import albumentations as A
import albumentations.pytorch


Precautions_msg = '(주의사항) ---- \n'

'''
- train.py

모델을 학습하는 전과정을 담은 코드

#### 실행법 ####
Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행
python train.py --kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30

pycharm의 경우: 
Run -> Edit Configuration -> train.py 가 선택되었는지 확인 
-> parameters 이동 후 아래를 입력 -> 적용하기 후 실행/디버깅
--kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30

*** def parse_args(): 실행 파라미터에 대한 모든 정보가 있다.  
*** def run(): 학습의 모든과정이 담긴 함수. 이곳에 다양한 trick을 적용하여 성능을 높혀보자. 
** def main(): fold로 나뉜 데이터를 run 함수에 분배해서 실행
* def train_epoch(), def val_epoch() : 완벽히 이해 후 수정

 MMCLab, 허종욱, 2020

 기본
 python train__.py --kernel-type test --data-dir /home/mmc/Paper/Copy-move/ --data-folder dataset/  --n-epochs 20 --batch-size 16

'''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel-type', type=str, required=True)
    # kernel_type : 실험 세팅에 대한 전반적인 정보가 담긴 고유 이름

    parser.add_argument('--data-dir', type=str, required=True)
    # base 데이터 폴더 ('./data/')

    parser.add_argument('--data-folder', type=str, required=True)
    # 데이터 세부 폴더 예: 'original_stone/'
    # os.path.join(data_dir, data_folder, 'train.csv')

    parser.add_argument('--image-size', type=int, default='512')
    # 입력으로 넣을 이미지 데이터 사이즈

    parser.add_argument('--use-amp', action='store_true')
    # 'A Pytorch EXtension'(APEX)
    # APEX의 Automatic Mixed Precision (AMP)사용
    # 기능을 사용하면 속도가 증가한다. 성능은 비슷
    # 옵션 00, 01, 02, 03이 있고, 01과 02를 사용하는게 적절
    # LR Scheduler와 동시 사용에 버그가 있음 (고쳐지기전까지 비활성화)
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2309

    parser.add_argument('--out-dim', type=int, default=1)
    # 모델 출력 output dimension

    parser.add_argument('--DEBUG', action='store_true')
    # 디버깅용 파라미터 (실험 에포크를 5로 잡음)

    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0,1,2,3,4,7')
    # 학습에 사용할 GPU 번호

    # parser.add_argument('--k-fold', type=int, default=4)
    # # data cross-validation
    # # k-fold의 k 값을 명시

    parser.add_argument('--log-dir', type=str, default='./logs')
    # Evaluation results will be printed out and saved to ./logs/
    # Out-of-folds prediction results will be saved to ./oofs/
    # 분할 했다가 다시 합친 결과

    parser.add_argument('--accumulation_step', type=int, default=2)
    # Gradient accumulation step
    # GPU 메모리가 부족할때, 배치를 잘개 쪼개서 처리한 뒤 합치는 기법
    # 배치가 30이면, 60으로 합쳐서 모델 업데이트함

    parser.add_argument('--model-dir', type=str, default='./weights')
    # weight 저장 폴더 지정
    # best :

    parser.add_argument('--batch-size', type=int, default=2)  # 배치 사이즈
    parser.add_argument('--num-workers', type=int, default=6)  # 데이터 읽어오는 스레드 개수
    parser.add_argument('--init-lr', type=float, default=3e-5)  # 초기 러닝 레이트. pretrained를 쓰면 매우 작은값
    parser.add_argument('--n-epochs', type=int, default=20)  # epoch 수

    args, _ = parser.parse_known_args()
    return args


def train_epoch(model, loader, optimizer):
    model.train()
    train_loss = []
    train_iou=[]

    bar = tqdm(loader)
    for i, (data, target) in enumerate(bar):

        optimizer.zero_grad()
        
        data, target = data.to(device), target.to(device)


        logits = model(data)

        loss = criterion(logits, target)

        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

        logits = logits.sigmoid()
        logits = mask_binarization(logits.detach().cpu(), 0.5)
        iou_np = compute_iou(logits,target)
        train_iou.append(iou_np)

    train_loss = np.mean(train_loss)
    train_iou = np.mean(train_iou)
    return train_loss, train_iou


def val_epoch(model, loader, n_test=1):

    model.eval()
    val_loss = []
    val_iou = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):  # ,patient_id
        
  
            data, target = data.to(device), target.to(device)
            
            logits = model(data)
            
            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

            logits = logits.sigmoid()
            logits = mask_binarization(logits.detach().cpu(), 0.5)
            iou_np = compute_iou(logits, target)
            val_iou.append(iou_np)

    val_loss = np.mean(val_loss)
    val_iou = np.mean(val_iou)

    return val_loss, val_iou


def run():
    '''
    학습 진행 메인 함수
    '''

    # 데이터셋 읽어오기
    data_dir = args.data_dir #/home/mmc/Paper/Copy-move/
    data_folder = args.data_folder # dataset/

    train_dir = os.path.join(data_dir, data_folder, 'train2014/')
    val_dir = os.path.join(data_dir, data_folder, 'val2014/')

    train_annotation = pd.read_csv(os.path.join(train_dir, 'annotation.csv'))
    val_annotation = pd.read_csv(os.path.join(val_dir, 'annotation.csv'))
    
    
    aug = A.Compose([
                 A.Resize(args.image_size, args.image_size)
                    ])

    train_dataset = CustomDataset(train_annotation, train_dir, aug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = CustomDataset(val_annotation, val_dir, aug)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    model = self_deep_matching()

    model = model.to(device)

    val_loss_min = np.inf

    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_bests.pth')

    optimizer = optim.Adadelta(model.parameters(), lr=args.init_lr)

    if DP:
        model = nn.DataParallel(model)

    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Epoch {epoch}')

        train_loss, train_iou = train_epoch(model, train_loader, optimizer)
        val_loss, val_iou = val_epoch(model, valid_loader)

        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, train_iou: {(train_iou):.4f}, valid_iou: {(val_iou):.4f}.'
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        if val_loss_min > val_loss:
            print('val_loss_min ({:.6f} --> {:.6f}). Saving model ...'.format(val_loss_min, val_loss))
            torch.save(model.state_dict(), model_file)
            val_loss_min = val_loss


# def main():
#
#     folds = range(args.k_fold)
#     for fold in folds:
#         run(fold, df_train, meta_features, n_meta_features, transforms_train, transforms_val, target_idx)


if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')

    # argument값 만들기
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    # GPU가 여러개인 경우 멀티 GPU를 사용함
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    # 실험 재현을 위한 random seed 부여하기
    set_seed(2359)
    device = torch.device('cuda')
    criterion = nn.BCEWithLogitsLoss()

    # 메인 기능 수행
    run()