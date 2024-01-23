import torch.nn as nn
from renderer_ import renderer_seperate,renderer
from Pointnet_Pointnet2.models.pointnet_sem_seg import get_model
from textureEncoder import TEncoder
import torch.nn.functional as F
from Decoder import Decoder
import torch


class Model_3D2D(nn.Module):
    def __init__(self, img_cropSize=128, N_b=8, pretrained=False, freeze=False):
        super(Model_3D2D, self).__init__()
        self.E_T = TEncoder(img_cropSize, N_b)
        self.E_G = get_model(5)
        self.D = Decoder(N_b)
        self.conv0 = torch.nn.Conv1d(13, 5, 1)
        self.bn0 = nn.BatchNorm1d(5)

        if pretrained:
            pretrained_dict = torch.load('./pretrained/pointnet_sem_seg.pth')["model_state_dict"]
            model_dict = self.E_G.state_dict()

            init_param = ['conv4.weight', 'conv4.bias', 'feat.stn.conv1.weight', 'feat.conv1.weight']
            for name in init_param:
                del pretrained_dict[name]

            model_dict.update(pretrained_dict)
            self.E_G.load_state_dict(model_dict)
            if freeze:
                for name, param in self.E_G.named_parameters():
                    if name == init_param[0] or name == init_param[1] or name == init_param[2] or name == init_param[3]:
                        print(name, 'is not freezing!!')
                        continue
                    param.requires_grad = False

    def forward(self, V, faces_idx, v_normals, T, M, device='cuda'):
        I_w = []
        I_o = []

        # vertex encoding
        expanded_message = M.unsqueeze(-1)  # batch, N_b,1
        expanded_message = expanded_message.expand(-1, -1, V.size()[1])
        V_m = torch.concat([v_normals.permute(0, 2, 1), expanded_message], dim=1)  # V=batch,3,5000
        V_e, _ = self.E_G(V_m)

        # texture encoding
        T_e = self.E_T(T, M)
        T_e = T_e.permute(0, 2, 3, 1)
        T = T.permute(0, 2, 3, 1)

        # renderertexture_map shape:(H,W,C)
        for v_o, v_e, v, f, t1, t0 in zip(v_normals, V_e, V, faces_idx, T_e, T):
            I_w.append(renderer_seperate(v, f, v_e, t1, device)[0, ..., :3])
            I_o.append(renderer_seperate(v, f, v_o, t0, device)[0, ..., :3])

        I_w = torch.stack(I_w, dim=0).permute(0, 3, 1, 2).to(device)
        I_o = torch.stack(I_o, dim=0).permute(0, 3, 1, 2).to(device)
        T_e = T_e.permute(0, 3, 1, 2)
        
        # message decoding
        M_r = self.D(I_w)

        return V_e, T_e, I_o, I_w, M_r
