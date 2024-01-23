import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from pytorch3d.io import load_obj
import torch
from utils import makeMessage
import torch.utils.data as data

def spherical_mapping(verts):
    x, y, z = verts.unbind(1)
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)
    u = phi / (2 * torch.pi)
    v = (torch.pi - theta) / np.pi
    return torch.stack((u, v), dim=1)


def vertex_nomalization(vertices):
    ## 1. 단위 구 내로 스케일링
    centroid = vertices.mean(axis=0)
    translated_vertices = vertices - centroid
    max_distance = torch.max(torch.linalg.norm(translated_vertices, axis=1))
    normalized_vertices_1 = translated_vertices / max_distance

    # 2. Vertex 벡터 정규화
    # magnitudes = torch.linalg.norm(vertices, axis=1)[:, np.newaxis]
    # normalized_vertices_2 = vertices / magnitudes
    return normalized_vertices_1

class CustomDataset(Dataset):
    def __init__(self, mesh_paths, N_b=8):
        self.mesh_paths = mesh_paths
        self.length = N_b

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, index):
        mesh_path = self.mesh_paths[index]

        #load mesh
        vert, face, aux = load_obj(mesh_path)

        #face
        faces_idx = face.verts_idx
        face_uvs = face.textures_idx

        #normals
        tex_coords = aux.verts_uvs
        tex_coords_pad=torch.zeros((5000,2))
        tex_coords_pad[:len(tex_coords)]=tex_coords
        v_normals = aux.normals
        normals=torch.cat([v_normals,tex_coords_pad],dim=1)

        # normalization
        verts = vertex_nomalization(vert)

        #texture
        texture=aux.texture_images['material_0']
        texture = texture[:, :, [2, 1, 0]]
        texture=texture.permute(2,0,1)

        # message making
        message = makeMessage(self.length)

        return verts, faces_idx, face_uvs, normals, texture, message


class CustomDataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = collate_mine


def collate_mine(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    verts_ = []
    face_ = []
    face_uvs_=[]
    normals_ = []
    texture_=[]
    message_ = []

    for i in batch:
        verts_.append(i[0])
        face_.append(i[1])
        face_uvs_.append(i[2])
        normals_.append(i[3])
        texture_.append(i[4])
        message_.append(i[5])

    verts_ = torch.stack(verts_, dim=0)
    normals_ = torch.stack(normals_, dim=0)
    texture_ = torch.stack(texture_, dim=0)
    message_ = torch.stack(message_, dim=0)

    return verts_, face_, face_uvs_, normals_, texture_,  message_

class testDataset(Dataset):
    def __init__(self, mesh_paths, N_b=8):
        self.mesh_paths = mesh_paths
        self.length = N_b

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, index):
        mesh_path = self.mesh_paths[index]
        mesh_name = mesh_path.split("\\")[1]

        #load mesh
        vert, face, aux = load_obj(mesh_path)

        #face
        faces_idx = face.verts_idx
        face_uvs = face.textures_idx

        #normals
        tex_coords = aux.verts_uvs
        tex_coords_pad=torch.zeros((5000,2))
        tex_coords_pad[:len(tex_coords)]=tex_coords
        v_normals = aux.normals
        normals=torch.cat([v_normals,tex_coords_pad],dim=1)

        # normalization
        verts = vertex_nomalization(vert)

        #texture
        texture=aux.texture_images['material_0']
        texture = texture[:, :,[2, 1, 0]]
        texture = texture.permute(2,0,1)

        # message making
        message = makeMessage(self.length)

        return verts, faces_idx, face_uvs, normals, texture, message, mesh_name


class testDataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(testDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = collate_test
def collate_test(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    verts_ = []
    face_ = []
    face_uvs_=[]
    normals_ = []
    texture_=[]
    message_ = []
    mesh_name_=[]

    for i in batch:
        verts_.append(i[0])
        face_.append(i[1])
        face_uvs_.append(i[2])
        normals_.append(i[3])
        texture_.append(i[4])
        message_.append(i[5])
        mesh_name_.append(i[6])

    verts_ = torch.stack(verts_, dim=0)
    normals_ = torch.stack(normals_, dim=0)
    texture_ = torch.stack(texture_, dim=0)
    message_ = torch.stack(message_, dim=0)

    return verts_, face_, face_uvs_, normals_, texture_,  message_, mesh_name_

def collate_mine(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    verts_ = []
    face_ = []
    face_uvs_=[]
    normals_ = []
    texture_=[]
    message_ = []

    for i in batch:
        verts_.append(i[0])
        face_.append(i[1])
        face_uvs_.append(i[2])
        normals_.append(i[3])
        texture_.append(i[4])
        message_.append(i[5])

    verts_ = torch.stack(verts_, dim=0)
    normals_ = torch.stack(normals_, dim=0)
    texture_ = torch.stack(texture_, dim=0)
    message_ = torch.stack(message_, dim=0)

    return verts_, face_, face_uvs_, normals_, texture_,  message_

class CustomDataLoader_separate(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(CustomDataLoader_separate, self).__init__(*args, **kwargs)
        self.collate_fn = collate_mine_seperate


def collate_mine_seperate(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## padd
    verts_ = []
    face_ = []
    normals_ = []
    texture_ = []
    message_ = []
    for i in batch:
        verts_.append(i[0])
        face_.append(i[1])
        normals_.append(i[2])
        texture_.append(i[3])
        message_.append(i[4])
    verts_ = torch.stack(verts_, dim=0)
    normals_ = torch.stack(normals_, dim=0)
    texture_ = torch.stack(texture_, dim=0)
    message_ = torch.stack(message_, dim=0)

    return verts_, face_, normals_, texture_, message_
# _utils.collate.default_collate