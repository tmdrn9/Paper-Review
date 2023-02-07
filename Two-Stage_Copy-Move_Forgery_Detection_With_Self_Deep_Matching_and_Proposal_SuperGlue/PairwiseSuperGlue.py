import torch

from superglue.models.matching import Matching
torch.set_grad_enabled(False)

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}

matching = Matching(config).eval().to(device)
keys = ['keypoints', 'scores', 'descriptors']


def PairwiseSuperglue(I, P_s):  # I = image
    M = []

    for c in range(len(P_s)):
        loc0 = P_s[c].flatten()
        img0 = I[loc0[0]:loc0[2], loc0[1]:loc0[3]]

        for cc in range(c + 1, len(P_s)):
            loc1 = P_s[cc].flatten()
            img1 = I[loc1[0]:loc1[2], loc1[1]:loc1[3]]

            pred = matching({'image0': img0, 'image1': img1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            mkpts0, mkpts1 = pred['matches0'], pred['matches1']
            S_m0, S_m1 = pred['matching_scores0'], pred['matching_scores1']

    return M



