#P shape = P,2,2
import torch
import numpy as np

data = torch.randn((3,512,512))
S = torch.randn((1,512,512))
P = torch.randn((5,2,2))
n = P.shape[0]

s_t = 0.4
iou_t=0.5
inter_t=0.8

def avgs(S,pp):
    temp=S[pp[0,0]:pp[1,0],pp[0,1]:pp[1,1]]
    return torch.mean(temp)


def compute_iou_inter(cand_box, gt_box):
    # Calculate intersection areas
    cand_box=cand_box.flatten()
    gt_box=gt_box.flatten()

    x1 = np.maximum(cand_box[0], gt_box[0])
    y1 = np.maximum(cand_box[1], gt_box[1])
    x2 = np.minimum(cand_box[2], gt_box[2])
    y2 = np.minimum(cand_box[3], gt_box[3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    cand_box_area = (cand_box[2] - cand_box[0]) * (cand_box[3] - cand_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = cand_box_area + gt_box_area - intersection

    iou = intersection / union
    return iou, intersection

def box_size(box):
    h = box[1,1]-box[0,1]
    w = box[1,0]-box[0,0]
    return h*w

def box_merge(box1,box2):
    box1=box1.flatten()
    box2=box2.flatten()

    x1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x2 = np.maximum(box1[2], box2[2])
    y2 = np.maximum(box1[3], box2[3])

    return torch.tensor([[x1,y1],[x2,y2]])

def ProposalSelection(P,S):
    P_t = []
    for i in range(n):
        p_p = P[i]
        s_p = avgs(S, p_p)
        if s_p > s_t:
            flag = 1
            for i_ps in range(len(P_t)):
                v_iou, v_inter = compute_iou_inter(P_t[i_ps],p_p)
                if v_iou > iou_t:
                    if s_p > avgs(S, P_t[i_ps]):
                        P_t[i_ps] = p_p
                        flag = 0
                        break
                if ((v_inter/box_size(p_p)) > inter_t) or ((v_inter/box_size(P_t[i_ps])) > inter_t):
                    p_m = box_merge(p_p, P_t[i_ps])
                    if avgs(S, p_m) > s_t:
                        P_t[i_ps] = p_m
                        flag = 0
                        break
            if flag == 1:
                P_t.append(p_p)

    P_s = []
    while len(P_s) != len(P_t):
        if len(P_s) != 0:
            P_t = P_s
        P_s=[]
        for i_ps1 in range(len(P_t)):
            for i_ps2 in range(i_ps1+1,len(P_t)):
                _, v_inter = compute_iou_inter(P_t[i_ps1], P_t[i_ps2])
                if ((v_inter/box_size(P_t[i_ps1])) > inter_t) or ((v_inter/box_size(P_t[i_ps2])) > inter_t):
                    p_m = box_merge(P_t[i_ps1], P_t[i_ps2])
                    if avgs(S, p_m) > s_t:
                        P_s.append(p_m)
                    else:
                        insert = P_t[i_ps1] if (v_inter/box_size(P_t[i_ps1])) > (v_inter/box_size(P_t[i_ps2])) else P_t[i_ps2]
                        P_s.append(insert)

    return torch.tensor(P_s)
