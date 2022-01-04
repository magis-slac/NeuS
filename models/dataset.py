import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import pickle

import diffoptics as optics
from diffoptics import Rays
import sys

# Path to your local clone of the magis simulator
SIMULATOR_PATH = '/sdf/home/s/sgaz/Magis-simulator'
sys.path.insert(0, SIMULATOR_PATH)

from magis.main_helpers import make_scene, get_sensor_index_positions, get_positions
from magis.mirror_utils import get_views_given_fixed_mirrors_smooth


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.dset_name = conf['dset_path']
        self.calib_name = conf['calib_path']
        with open(self.dset_name,'rb') as f:
             in_dataset = pickle.load(f)

        #with open(calib_name, 'rb') as f:
        #     calib_dict = pickle.load(f)

        self.H, self.W = in_dataset.shape[1:3]
        self.image_pixels = self.H * self.W

        in_dataset = in_dataset.reshape(in_dataset.shape[0], -1, 6)
        assert in_dataset.shape == (in_dataset.shape[0], self.H*self.W, 1+1+1+3)
        self.in_dataset = torch.from_numpy(in_dataset)

        self.n_images = len(self.in_dataset)

        #Let's try to put a MAGIS scene in here
        # Get mirror parameters
        m = conf['m']
        f = conf['f']

        xm,ym,zm,mirror_radii,angles,theta,phi,foc,obj = get_views_given_fixed_mirrors_smooth(
            m  = m,
            f  = f * 100,
            fn = 1.23,
            sv = 24/10,
            sh = 24/10,
            skipmirrs=5,
            extreme_view_angle = np.radians(55),
            window_pos = 3.0,
            fixed_radii = [.25],
            num_mirrors = [500])

        #assert len(angles) == self.in_dataset.shape[0]

        normals = torch.zeros((len(angles), 3))
        for i in range(len(theta)):
            normal_angles = angles
            normal = optics.vector(np.cos(normal_angles[i]),
                                   np.cos(theta[i]) * np.sin(normal_angles[i]),
                                   np.sin(theta[i]) * np.sin(normal_angles[i]))
            normals[i] = optics.normalize_vector(normal)


        mirror_parameters = normals, torch.tensor(xm / 100, dtype=torch.float), torch.tensor(ym / 100, dtype=torch.float), torch.tensor(zm / 100, dtype=torch.float), torch.tensor(mirror_radii / 100, dtype=torch.float)

        # @Todo check sensor parameters
        pixel_size = conf['pixel_size']
        self.scene = make_scene(object_x_pos=obj/100, f=f, m=m, na=1 / 1.4, nb_mirror=None, sensor_resolution=(conf['sensor_resolution_x'],conf['sensor_resolution_y']),
                       sensor_pixel_size=(pixel_size, pixel_size), poisson_noise_mean=2, quantum_efficiency=0.77,
                       mirror_parameters=mirror_parameters)

        self.continuous_positions = get_positions(self.scene)

        rad = conf['rad']
        trans_mat = torch.eye(4)
        trans_mat[0][3] = -obj/100* 1/rad

        scale_mat = torch.eye(4)
        scale_mat[0][0] = 1/rad
        scale_mat[1][1] = 1/rad
        scale_mat[2][2] = 1/rad

        full_scale_mat = torch.matmul(trans_mat, scale_mat)[:-1]
        self.full_scale_mat = full_scale_mat

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        self.object_bbox_min = object_bbox_min[:, None][:3, 0]
        self.object_bbox_max = object_bbox_max[:, None][:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        #img = cv.imread(self.images_lis[idx])
        img = self.in_dataset[idx, :, -3:].reshape((self.W,self.W, 3)).numpy()*256
        return (cv.resize(img, (self.W // resolution_level, self.W // resolution_level))).clip(0, 255).astype(np.uint8)
        #return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def gen_rays_at_magis(self, lens, mirror, data_id, mirror_id):
        """
        Generate rays at world space from one camera.
        """
        ind = torch.arange(self.H*self.W)
        # Sampling rays from the sensor to the lens
        origins = torch.zeros(ind.shape[0], 3, device=self.device)
        origins[:, 0] = self.continuous_positions[mirror_id][0]
        origins[:, 1] = self.in_dataset[data_id, ind, 1]
        origins[:, 2] = self.in_dataset[data_id, ind, 2]
        points_on_lens = lens.sample_points_on_lens(ind.shape[0], device=self.device)
        directions = optics.batch_vector(points_on_lens[:, 0] - origins[:, 0],
                                         points_on_lens[:, 1]*0 - origins[:, 1],
                                         points_on_lens[:, 2]*0 - origins[:, 2])
        rays_sensor_to_lens = Rays(origins, directions, device=self.device,
                                   meta = {'target' : self.in_dataset[data_id, ind, -3:].to(self.device),
                                           'ind' : ind})

        # Intersection with lens
        t1 = lens.get_ray_intersection(rays_sensor_to_lens)
        mask_t1 = ~torch.isnan(t1)
        ray_lens_to_mirror = lens.intersect(rays_sensor_to_lens.get_at(mask_t1), t1[mask_t1])

        # Intersection with mirror
        t2 = mirror.get_ray_intersection(ray_lens_to_mirror)
        mask = ~torch.isnan(t2)
        assert mask.shape[0] == ind[mask_t1].shape[0]
        #rays_mirror_to_object = mirror.intersect(ray_lens_to_mirror.get_at(mask), t2[mask])
        rays_mirror_to_object = mirror.intersect(ray_lens_to_mirror, t2)

        color = self.in_dataset[data_id, ind[mask_t1], -3:]

        rays_mirror_to_object.origins = torch.matmul(self.full_scale_mat, 
             torch.cat((rays_mirror_to_object.origins, torch.ones((rays_mirror_to_object.origins.shape[0],1))), dim=1)[:, :, np.newaxis]).squeeze(dim=-1)
        rays_mirror_to_object.directions = torch.matmul(self.full_scale_mat,
             torch.cat((rays_mirror_to_object.directions, torch.zeros((rays_mirror_to_object.origins.shape[0],1))), dim=1)[:, :, np.newaxis]).squeeze(dim=-1)

        rays_mirror_to_object.directions = rays_mirror_to_object.directions/torch.sqrt(torch.sum(rays_mirror_to_object.directions**2, dim=1, keepdim=True))

        return rays_mirror_to_object, color.cuda()


    def gen_random_rays_at_magis(self, lens, mirror, ind, data_id, mirror_id):
        """
        Generate random rays at world space from one camera.
        """
        # Sampling rays from the sensor to the lens
        origins = torch.zeros(ind.shape[0], 3, device=self.device)
        origins[:, 0] = self.continuous_positions[mirror_id][0]
        origins[:, 1] = self.in_dataset[data_id, ind, 1]
        origins[:, 2] = self.in_dataset[data_id, ind, 2]
        points_on_lens = lens.sample_points_on_lens(ind.shape[0], device=self.device)
        directions = optics.batch_vector(points_on_lens[:, 0] - origins[:, 0],
                                         points_on_lens[:, 1]*0 - origins[:, 1],
                                         points_on_lens[:, 2]*0 - origins[:, 2])
        rays_sensor_to_lens = Rays(origins, directions, device=self.device,
                                   meta = {'target' : self.in_dataset[data_id, ind, -3:].to(self.device),
                                           'ind' : ind})

        # Intersection with lens
        t1 = lens.get_ray_intersection(rays_sensor_to_lens)
        mask_t1 = ~torch.isnan(t1)
        ray_lens_to_mirror = lens.intersect(rays_sensor_to_lens.get_at(mask_t1), t1[mask_t1])

        # Intersection with mirror
        t2 = mirror.get_ray_intersection(ray_lens_to_mirror)
        mask = ~torch.isnan(t2)
        assert mask.shape[0] == ind[mask_t1].shape[0]
        rays_mirror_to_object = mirror.intersect(ray_lens_to_mirror.get_at(mask), t2[mask])
        
        color = self.in_dataset[data_id, ind[mask_t1][mask], -3:]

        rays_mirror_to_object.origins = torch.matmul(self.full_scale_mat,
             torch.cat((rays_mirror_to_object.origins, torch.ones((rays_mirror_to_object.origins.shape[0],1))), dim=1)[:, :, np.newaxis]).squeeze(dim=-1)
        rays_mirror_to_object.directions = torch.matmul(self.full_scale_mat,
             torch.cat((rays_mirror_to_object.directions, torch.zeros((rays_mirror_to_object.origins.shape[0],1))), dim=1)[:, :, np.newaxis]).squeeze(dim=-1)

        rays_mirror_to_object.directions = rays_mirror_to_object.directions/torch.sqrt(torch.sum(rays_mirror_to_object.directions**2, dim=1, keepdim=True))
        
        return rays_mirror_to_object, color.cuda()
