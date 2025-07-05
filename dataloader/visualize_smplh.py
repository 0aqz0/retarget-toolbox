import torch
import numpy as np

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp

support_dir = '../reference/amass/support_data'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

amass_npz_fname = osp.join(support_dir, 'github_data/dmpl_sample.npz')
bdata = np.load(amass_npz_fname)

subject_gender = str(bdata['gender'],'utf-8')
print('keys:', list(bdata.keys()))
print('gender:', subject_gender)

from human_body_prior.body_model.body_model import BodyModel

bm_fname = osp.join('./', 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = osp.join('./', 'body_models/dmpls/{}/model.npz'.format(subject_gender))

num_betas = 16
num_dmpls = 8

bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
faces = c2c(bm.f)

time_length = len(bdata['trans'])

body_param = {
    'root_orient': torch.Tensor(bdata['poses'][:,:3]).to(device),
    'pose_body': torch.Tensor(bdata['poses'][:,3:66]).to(device),
    'pose_hand': torch.Tensor(bdata['poses'][:,66:]).to(device),
    'trans': torch.Tensor(bdata['trans']).to(device),
    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(device),
    'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(device)
}

print('body parameters: \n{}'.format(' \n'.join(['{}: {}'.format(k, v.shape) for k, v in body_param.items()])))
print('time length:', time_length)

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

body_pose_beta = bm(**{k:v for k,v in body_param.items() if k in ['pose_body', 'betas']})

def vis_body_pose_beta(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_pose_beta(fId=0)

body_pose_hand = bm(**{k:v for k,v in body_param.items() if k in ['pose_body', 'betas', 'pose_hand']})

def vis_body_pose_hand(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_pose_hand(fId=0)

def vis_body_joints(fId = 0):
    joints = c2c(body_pose_hand.Jtr[fId])
    joints_mesh = points_to_spheres(joints, point_color=colors['red'], radius=0.005)

    mv.set_static_meshes([joints_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_joints(fId=0)

body_dmpls = bm(**{k:v for k,v in body_param.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls']})

def vis_body_dmpls(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_dmpls.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_dmpls(fId=0)

body_trans_root = bm(**{k:v for k,v in body_param.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls', 'trans', 'root_orient']})

def vis_body_trans_root(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_trans_root(fId=0)

def vis_body_transformed(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0,0,1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (1,0,0)))
    
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_transformed(fId=0)

import matplotlib.pyplot as plt
joints = c2c(body_trans_root.Jtr[0])
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.scatter(joints[:,0], joints[:,1], joints[:,2], s=30, c='red')
# plt.show()

amass_dir = '../data/AMASS/AMASS_Complete/'
work_dir = '../data/AMASS/prepared_data'

amass_split = {
    'valid': ['SFU'],
    'test': ['SSM_synced'],
    'train': ['MPI_Limits']
}
from amass.data.prepare_data import prepare_amass
prepare_amass(amass_split, amass_dir, work_dir)

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob

class AMASS_DS(Dataset):
    def __init__(self, dataset_dir, num_betas=16):
        self.ds = {}
        for data_fname in glob.glob(osp.join(dataset_dir, "*.pt")):
            print(data_fname)
            k = osp.basename(data_fname).replace('.pt', '')
            self.ds[k] = torch.load(data_fname)
        self.num_betas = num_betas
    
    def __len__(self):
        print(self.ds['trans'].shape)
        return len(self.ds['trans'])
    
    def __getitem__(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys()}
        data['root_orient'] = data['pose'][:3]
        data['pose_body'] = data['pose'][3:66]
        data['pose_hand'] = data['pose'][66:]
        data['betas'] = data['betas'][:self.num_betas]

        return data
    
num_betas = 16
test_split_dir = osp.join(work_dir, 'stage_III', 'test')
ds = AMASS_DS(dataset_dir=test_split_dir, num_betas=num_betas)
batch_size = 5
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)

import trimesh
from body_visualizer.tools.vis_tools import colors, imagearray2file
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import show_image

# imw, imh = 1600, 1600
# mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

from human_body_prior.body_model.body_model import BodyModel
bm_fname = osp.join('./', 'body_models/smplh/male/model.npz')
num_betas = 16
num_dmpls = 8
bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas).to(device)
faces = c2c(bm.f)

bdata = next(iter(dataloader))
body_v = bm.forward(**{k:v.to(device) for k,v in bdata.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls', 'trans', 'root_orient']}).v
view_angles = [0, 180, 90, -90]
images = np.zeros([len(view_angles), batch_size, 1, imw, imh, 3])
for cId in range(0, batch_size):
    orig_body_mesh = trimesh.Trimesh(vertices=c2c(body_v[cId]), faces=c2c(bm.f), vertex_colors=np.tile(colors['grey'], [6890, 1]))
    for rId, angle in enumerate(view_angles):
        if angle != 0: orig_body_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(angle), (0,1,0)))
        mv.set_meshes([orig_body_mesh])
        images[rId, cId, 0] = mv.render(render_wireframe=False)
        if angle != 0: orig_body_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-angle), (0,1,0)))

img = imagearray2file(images)
show_image(np.array(img)[0])
plt.show()