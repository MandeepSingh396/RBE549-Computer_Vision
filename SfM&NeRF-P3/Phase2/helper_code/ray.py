import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# def get_rays(h, w, f, pose, near, far, Nc):

#   pose = pose.cpu().numpy()
# #making meshgrid of the size of image
#   x = np.linspace(0, w-1, w)
#   y = np.linspace(0, h-1, h)
#   xi, yi = np.meshgrid(x, y, indexing='xy')

# # normalized coordinates (i.e. wrt image plane center) - z = 1(assumed)
#   norm_x = (xi - w * 0.5) / f
#   norm_y = (yi - h * 0.5) / f

# #rotation and translation components of Camera to world transformation matrix
#   rot = pose[:3,:3]
#   trans = pose[:3, -1]

# # calculating ray unit vectors and origins
# #ray origin - vectors starting from image plane center to image pixel position
# #ray unit vector - unit vector between the camera aperture and the pixel position
#   ray_dirs_all = []
#   ray_origins_all = []
#   for i in range(h):
#     ray_dirs = []
#     ray_origins = []
#     for j in range(w):
#       #ray direction in Image plane frame
#       Xc = np.array([norm_x[i][j], -norm_y[i][j], -1]).reshape(1,-1)
#       #ray direction in World frame
#       Xw = (rot.dot(Xc.T)).reshape(1,-1)
#       #calculating the unit vector from ray_direction vector
#       ray_dir = Xw/np.linalg.norm(Xw)
#       #origin wrt world frame
#       ray_origin = trans.T
#       ray_dirs.append(ray_dir)
#       ray_origins.append(ray_origin)

#     ray_dirs = np.array(ray_dirs).reshape(w,-1)
#     ray_origins = np.array(ray_origins).reshape(w,-1)
#     ray_dirs_all.append(ray_dirs)
#     ray_origins_all.append(ray_origins)

#   ray_directions = np.array(ray_dirs_all).reshape(h,w,-1)
#   ray_origins = np.array(ray_origins_all).reshape(h,w,-1)
#   ray_directions = torch.from_numpy(ray_directions)
#   ray_origins = torch.from_numpy(ray_origins)

# #sampling the points along the ray
#   depth_val = torch.linspace(near, far, Nc)
# #add noise in otherwise uniform sampling
#   noise_shape = list(ray_origins.shape[:-1]) + [Nc]
#   noise = torch.rand(size = noise_shape) * (far - near)/Nc
#   depth_val = depth_val + noise
# #co-ordinates of the sample points
#   query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_val[..., :, None]

#   return ray_directions, ray_origins, depth_val, query_points


def get_rays(h, w, f, pose, near, far, Nc, device):

#making meshgrid
  x = torch.linspace(0, w-1, w)
  y = torch.linspace(0, h-1, h)
  xi, yi = torch.meshgrid(x, y, indexing='xy')
  xi = xi.to(device)
  yi = yi.to(device)
# normalized coordinates
  norm_x = (xi - w * 0.5) / f
  norm_y = (yi - h * 0.5) / f

#direction unit vectors matrix
  directions = torch.stack([norm_x, - norm_y, -torch.ones_like(xi)], dim = -1)
  directions = directions[..., None,:]

#camera matrix : 3x3 matrix from the 4x4 projection matrix
  rotation = pose[:3, :3]
  translation = pose[:3, -1]

  camera_directions = directions * rotation
  ray_directions = torch.sum(camera_directions, dim = -1)
  ray_directions = ray_directions/torch.linalg.norm(ray_directions, dim = -1, keepdims = True)
  ray_origins =  torch.broadcast_to(translation, ray_directions.shape)

  # print(ray_directions.shape)
  # print(ray_origins.shape)

#get the sample points
  depth_val = torch.linspace(near, far, Nc)
  # print(depth_val.shape)
  noise_shape = list(ray_origins.shape[:-1]) + [Nc]
  noise = torch.rand(size = noise_shape) * (far - near)/Nc
  # print(noise.shape)
  depth_val = depth_val + noise
  depth_val = depth_val.to(device)
  # print(depth_val.shape)
  query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_val[..., :, None]
  # print(query_points.shape)

  return ray_directions, ray_origins, depth_val, query_points