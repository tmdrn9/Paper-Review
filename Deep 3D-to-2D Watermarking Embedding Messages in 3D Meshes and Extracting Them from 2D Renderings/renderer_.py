import random
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, PointLights,SplatterPhongShader,SoftPhongShader,
    MeshRenderer, MeshRasterizer,TexturesUV
)


# 정점의 구면 매핑 텍스처 좌표 생성
def spherical_mapping(verts):
    x, y, z = verts.unbind(1)
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)
    u = phi / (2 * torch.pi)
    v = (torch.pi - theta) / np.pi
    return torch.stack((u, v), dim=1)

def renderer(verts, faces_idx, v_normals, texture_image, device='cuda'):

    tex_coords = v_normals[:,3:]
    temp_uvs = faces_idx.to(device)
    v_normals = v_normals[:,:3]

    # 텍스처 초기화
    textures = TexturesUV(maps=[texture_image], verts_uvs=[tex_coords], faces_uvs=[temp_uvs])

    # Meshes 생성
    mesh = Meshes(
        verts=[verts],
        faces=[temp_uvs],
        textures=textures,
        verts_normals=[v_normals]
    ).to(device)


    # 카메라 및 렌더러 설정
    num_views = 1

    # In our experiment, we use random camera locations with ranges, x = 0, y ∈ [-3, -2], z ∈ [2, 4]. Camera is always looking at the object center with fovy = 60◦
    elev = torch.linspace(-3, -2, num_views)
    azim = torch.linspace(2, 4, num_views) 

    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T) #fov default = 60

    #The point light location is also randomly sampled from a Gaussian distribution with mean µ = (2, 1, 2) and standard deviation σ = 0.2.
    location0 = torch.normal(2, 0.2, size=(2,))
    location1 = torch.normal(1, 0.2,size=(1,))
    lights = PointLights(device=device, location=[[location0[0], location1[0], location0[1]]])


    raster_settings = RasterizationSettings(
        image_size=(500,500), ##
        # blur_radius=0.0,
        faces_per_pixel=10,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SplatterPhongShader(cameras=cameras,device=device)
    )

    mesh = mesh.extend(num_views)
    return renderer(mesh, light=lights, cameras=cameras)
