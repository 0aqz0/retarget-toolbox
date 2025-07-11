"""
animation.py
"""
# Basic animation class. Can use for 
# deep-learning (pre/post)processing 
# and visualization etc.


import numpy as np
import scipy.ndimage as ndimage
from . import quat
from . import dualquat
from .skel import Skel


class Animation:
    def __init__(
        self,
        skel: Skel,
        quats,
        trans,
        positions = None,
        fps: int=30,
        anim_name: str="animation",
    ):
        """ Class for Motions representations.

        Args:
            skel (Skel): Skelton definition.
            quats (np.ndarray): Joints rotations. shape: (T, J, 4)
            trans (np.ndarray): Root transitions. shape: (T, 3)
            positions (np.ndarray): Joints positions. shape: (T, J, 3). Defaults to None.
            fps (int): Frame per seconds for animation. Defaults to 30.
            anim_name (str, optional): Name of animation. Defaults to "animation".
        """
        self.skel = skel
        self.quats = quats
        self.trans = trans
        self.positions = positions
        self.fps = fps
        self.name = anim_name
    
    def __len__(self): return len(self.trans)
    
    def __add__(self, other):
        quats = np.concatenate([self.quats, other.quats], axis=0)
        trans = np.concatenate([self.trans, other.trans], axis=0)
        return Animation(
            skel=self.skel,
            quats=quats,
            trans=trans,
            fps=self.fps,
            anim_name=self.name,
        )
    
    def __getitem__(self, index):
        if isinstance(index, int):
            if index == -1:
                index = len(self) - 1
            return Animation(
                skel=self.skel,
                quats=self.quats[index:index+1],
                trans=self.trans[index:index+1],
                fps=self.fps,
                anim_name=self.name,
            )
        elif isinstance(index, slice):
            return Animation(
                skel=self.skel,
                quats=self.quats[index],
                trans=self.trans[index],
                fps=self.fps,
                anim_name=self.name,
            )
        else:
            raise TypeError
    
    def cat(self, other):
        assert isinstance(other, Animation)

        self.quats = np.concatenate(
            [self.quats, other.quats], axis=0)
        self.trans = np.concatenate(
            [self.trans, other.trans], axis=0
        )
    
    @property
    def parents(self):
        return self.skel.parents
    
    @property
    def joint_names(self):
        return self.skel.names
    
    @property
    def offsets(self):
        return self.skel.offsets
    
    # =====================
    #  Rotation conversion
    # =====================
    @property
    def grot(self):
        """global rotations(quaternions) for each joints. shape: (T, J, 4)"""
        return quat.fk_rot(lrot=self.quats, parents=self.parents)
    
    @property
    def crot(self):
        """Projected root(simulation bone) centric global rotations."""
        
        # Root rotations relative to the forward vector.
        root_rot = self.proj_root_rot
        # Root positions projected on the ground (y=0).
        root_pos = self.proj_root_pos()
        # Make relative to simulation bone.
        lrot = self.quats.copy()
        lpos = self.lpos.copy()
        lrot[:, 0] = quat.inv_mul(root_rot, lrot[:, 0])
        lpos[:, 0] = quat.inv_mul_vec(root_rot, lpos[:, 0] - root_pos)
        return quat.fk_rot(lrot, self.parents)
    
    @property
    def axangs(self):
        """axis-angle rotation representations. shape: (T, J, 3)"""
        return quat.to_axis_angle(self.quats)
    
    @property
    def xforms(self):
        """rotation matrix. shape: (T, J, 3, 3)"""
        return quat.to_xform(self.quats)
    
    @property
    def ortho6ds(self):
        return quat.to_xform_xy(self.quats)
    
    @property
    def sw_tws(self):
        """Twist-Swing decomposition.
           This function is based on HybrIK.
           Remove root rotations.
           quat.mul(swings, twists) reproduce original rotations.
        return:
            twists: (T, J-1, 4), except ROOT.
            swings: (T, J-1, 4)
        """
        
        rots = self.quats.copy()
        pos:  np.ndarray = self.lpos.copy()
        
        children = \
            -np.ones(len(self.parents), dtype=int)
        for i in range(1, len(self.parents)):
            children[self.parents[i]] = i
        
        swings = []
        twists = []
        
        for i in range(1, len(self.parents)):
            # ルートに最も近いJointはtwistのみ / only twist for nearest joint to root.
            if children[i] < 0:
                swings.append(quat.eye([len(self), 1]))
                twists.append(rots[:, i:i+1])
                continue
            
            u = pos[:, children[i]:children[i]+1]
            rot = rots[:, i:i+1]
            u = u / np.linalg.norm(u, axis=-1, keepdims=True)
            v = quat.mul_vec(rot, u)
            
            swing = quat.normalize(quat.between(u, v))
            swings.append(swing)
            twist = quat.inv_mul(swing, rot)
            twists.append(twist)
        
        swings_np = np.concatenate(swings, axis=1)
        twists_np = np.concatenate(twists, axis=1)

        return swings_np, twists_np

    # ==================
    #  Position from FK
    # ==================
    def set_positions_from_fk(self):
        self.positions = self.gpos
    
    @property
    def lpos(self):
        lpos = self.offsets[None].repeat(len(self), axis=0)
        lpos[:,0] = self.trans
        return lpos
    
    @property
    def gpos(self):
        """Global space positions."""
        
        _, gpos = quat.fk(
            lrot=self.quats, 
            lpos=self.lpos, 
            parents=self.parents
        )
        return gpos
    
    @property
    def rtpos(self):
        """Root-centric local positions."""
        
        lrots = self.quats.copy()
        lposs = self.lpos.copy()
        # ROOT to zero.
        lrots[:,0] = quat.eye([len(self)])
        lposs[:,0] = np.zeros([len(self), 3])
        
        _, rtpos = quat.fk(
            lrot=lrots, 
            lpos=lposs, 
            parents=self.parents
        )
        return rtpos
    
    @property
    def cpos(self):
        """Projected root(simulation bone) centric positions."""
        lrot = self.quats.copy()
        lpos = self.lpos
        c_root_rot, c_root_pos = self.croot()
        lrot[:, 0] = c_root_rot
        lpos[:, 0] = c_root_pos
        
        crot, cpos = quat.fk(lrot, lpos, self.parents)
        return cpos
    
    def croot(self, idx: int=None):
        """return character space info.
        return:
            crot: character space root rotations. [T, 4]
            cpos: character space root positions. [T, 3]
        """
        # Root rotations relative to the forward vector.
        root_rot = self.proj_root_rot
        # Root positions projected on the ground (y=0).
        root_pos = self.proj_root_pos()
        # Make relative to simulation bone.
        crot = quat.inv_mul(root_rot, self.quats[:, 0])
        cpos = quat.inv_mul_vec(root_rot, self.trans - root_pos)
        if idx is not None:
            return crot[idx], cpos[idx]
        else: return crot, cpos
    
    # ============
    #   Velocity
    # ============
    @property
    def gposvel(self):
        gpos  = self.gpos
        gpvel = np.zeros_like(gpos)
        gpvel[1:] = (gpos[1:] - gpos[:-1]) * self.fps # relative position from previous frame.
        gpvel[0] = gpvel[1] - (gpvel[3] - gpvel[2])
        return gpvel
    
    @property
    def cposvel(self):
        cpos  = self.cpos
        cpvel = np.zeros_like(cpos)
        cpvel[1:] = (cpos[1:] - cpos[:-1]) * self.fps # relative position from previous frame.
        cpvel[0] = cpvel[1] - (cpvel[3] - cpvel[2])
        return cpvel

    @property
    def lrotvel(self):
        """Calculate rotation velocities with 
           rotation vector style."""    
        
        lrot  = self.quats.copy()
        lrvel = np.zeros_like(self.lpos)
        lrvel[1:] = quat.to_axis_angle(quat.abs(quat.mul(lrot[1:], quat.inv(lrot[:-1])))) * self.fps
        lrvel[0] = lrvel[1] - (lrvel[3] - lrvel[2])
        return lrvel
    
    # ==============
    #   4x4 matrix
    # ==============
    @property
    def local_transform(self):
        xforms = self.xforms
        offsets = self.offsets
        transform = np.zeros(xforms.shape[:-2] + (4, 4,))
        transform[..., :3, :3] = xforms
        transform[..., :3,  3] = offsets
        transform[..., 3, 3] = 1
        return transform
    
    @property
    def global_transform(self):
        ltrans = self.local_transform.copy()
        parents = self.parents
        
        gtrans = [ltrans[...,:1,:,:]]
        for i in range(1, len(parents)):
            gtrans.append(np.matmul(gtrans[parents[i]],ltrans[...,i:i+1,:,:]))
        
        return np.concatenate(gtrans, axis=-3)
    
    # ==================
    #  dual quaternions
    # ==================
    @property
    def local_dualquat(self):
        return dualquat.from_rot_and_trans(self.quats, self.lpos)
    
    @property
    def global_dualquat(self):
        return dualquat.fk(self.local_dualquat, self.parents)
    
    # =============
    #  trajectory 
    # =============
    def proj_root_pos(self, remove_vertical: bool=False):
        """Root position projected on the ground (world space).
        return:
            Projected bone positons as ndarray of shape [len(self), 3] or [len(self), 2](remove_vertical).
        """
        vertical = self.skel.vertical
        if remove_vertical:
            settle_ax = []
            for i, ax in enumerate(vertical):
                if ax == 0:
                    settle_ax.append(i)
            return self.trans[..., settle_ax]
        else:
            return self.trans * np.array([abs(abs(ax) - 1) for ax in vertical])
    
    @property
    def proj_root_rot(self):
        # root rotations relative to the forward on the ground. [len(self), 4]
        forward = self.skel.forward
        return quat.normalize(
            quat.between(np.array(forward), self.root_direction())
        )
    
    def root_direction(self, remove_vertical: bool=False):
        """Forward orientation vectors on the ground (world space).
        return:
            Forward vectors as ndarray of shape [..., 3] or [..., 2](remove_vertical).
        """
        # Calculate forward vectors except vertical axis.
        vertical = self.skel.vertical
        rt_rots = self.quats[..., 0, :]
        forwards = np.zeros(shape=rt_rots.shape[:-1] + (3,))
        forwards[...,] = self.skel.rest_forward
        rt_fwd = quat.mul_vec(rt_rots, forwards) * np.array([abs(abs(ax) - 1) for ax in vertical]) # [T, 3]
        # Normalize vectors.
        norm_rt_fwd = rt_fwd / np.linalg.norm(rt_fwd, axis=-1, keepdims=True)
        if remove_vertical:
            settle_ax = []
            for i, ax in enumerate(vertical):
                if ax == 0:
                    settle_ax.append(i)
            norm_rt_fwd = norm_rt_fwd[..., settle_ax]
        return norm_rt_fwd
    
    # ===================
    #  Future trajectory
    # ===================
    def future_traj_poss(self, frame: int, remove_vertical: bool=True, cspace=True):
        """Calculate future trajectory positions on simulation bone.
        Args:
            frame (int): how many ahead frame to see.
            remove_vertical (bool, optional): remove vertical axis positions. Defaults to True.
            cspace (bool, optional): use local character space. Defaults to True.
        Returns:
            np.ndarray: future trajectories positions. shape=(len(self), 3) or (len(self), 2)
        """
        idxs = self.clamp_future_idxs(frame)
        proj_root_pos = self.proj_root_pos()
        if cspace:
            traj_pos = quat.inv_mul_vec(self.proj_root_rot, proj_root_pos[idxs] - proj_root_pos)
        else:
            traj_pos = proj_root_pos[idxs]

        if remove_vertical:
            vertical = self.skel.vertical
            settle_ax = []
            for i, ax in enumerate(vertical):
                if ax == 0:
                    settle_ax.append(i)
            return traj_pos[..., settle_ax]
        else: 
            return traj_pos

    def future_traj_dirs(self, frame: int, remove_vertical: bool=True, cspace=True):
        """Calculate future trajectory directions on simulation bone (local character space).
        Args:
            frame (int): how many ahead frame to see.
            remove_vertical (bool, optional): remove vertical axis. Defaults to True.
            cspace (bool, optional): use local character space. Defaults to True.
        Returns:
            np.ndarray: future trajectories directions. shape=(len(self), 3) or (len(self), 2)
        """
        idxs = self.clamp_future_idxs(frame)
        root_directions = self.root_direction()
        if cspace:
            traj_dir = quat.inv_mul_vec(self.proj_root_rot, root_directions[idxs])
        else:
            traj_dir = root_directions[idxs]
        
        if remove_vertical:
            vertical = self.skel.vertical
            settle_ax = []
            for i, ax in enumerate(vertical):
                if ax == 0:
                    settle_ax.append(i)
            return traj_dir[..., settle_ax]
        else: 
            return traj_dir
    
    def clamp_future_idxs(self, offset: int):
        """Function to calculate the frame array for `offset` frame ahead.
        If `offset` frame ahead does not exist, 
        return the last frame.
        """
        idxs = np.arange(len(self)) + offset
        idxs[-(offset + 1):] = idxs[-(offset + 1)]
        return idxs
    
    # =====================
    #    Other functions
    # =====================
    def calc_foot_contact(
        self, 
        method: str="velocity",
        threshold: float=0.15,
        left_foot_name: str="LeftToe",
        right_foot_name: str="RightToe",
    ):
        if method == "velocity":
            contact_vel = np.linalg.norm(
                self.gposvel[:, 
                    [self.joint_names.index(left_foot_name), 
                    self.joint_names.index(right_foot_name)]
                ], axis=-1
            )
            contacts = contact_vel < threshold
        elif method == "position":
            # vertical axis position for each frame.
            settle_idx = 0
            inverse = False # Is the negative direction vertical?
            for i, ax in enumerate(self.skel.vertical):
                if ax == 1:
                    settle_idx = i
                if ax == -1:
                    settle_idx = i
                    inverse = True 
            contact_pos = self.gpos[:,
                [self.joint_names.index(left_foot_name), 
                self.joint_names.index(right_foot_name)], settle_idx]
            if inverse:
                contact_pos *= -1
            contacts = contact_pos < threshold
        else:
            raise ValueError("unknown value selected on `method`.")
        for ci in range(contacts.shape[1]):
            contacts[:, ci] = ndimage.median_filter(
                contacts[:, ci],
                size=6,
                mode="nearest"
            )
        return contacts
    
    def mirror(self, dataset: str=None):
        vertical = self.skel.vertical
        forward = self.skel.forward
        mirror_axis = []
        for vert_ax, fwd_ax in zip(vertical, forward):
            if abs(vert_ax) == 1 or abs(fwd_ax) == 1:
                mirror_axis.append(1)
            else:
                mirror_axis.append(-1)
        if dataset == "lafan1":
            # quatM, lposM = animation_mirror(
            #     lrot=self.quats,
            #     lpos=self.lpos,
            #     names=self.joint_names,
            #     parents=self.parents
            # )
            # transM = lposM[:, 0]
            pass
        else:
            quatM, transM = mirror_rot_trans(
                lrot=self.quats,
                lpos=self.lpos,
                trans=self.trans,
                names=self.joint_names,
                parents=self.parents,
                mirror_axis=mirror_axis,
            )
        return Animation(
            skel=self.skel,
            quats=quatM,
            trans=transM,
            fps=self.fps,
            anim_name=self.name+"_M",
        )
    
    @staticmethod
    def no_animation(
        skel: Skel, 
        fps: int=30,
        num_frame: int=1,
        anim_name: str="animation",
    ):
        """Create a unit animation (no rotation, no transition) from Skel"""
        quats = quat.eye([num_frame, len(skel)])
        trans = np.zeros([num_frame, 3])
        return Animation(skel, quats, trans, None, fps, anim_name)


def mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

def inv(q):
    return np.array([1, -1, -1, -1], dtype=np.float32) * q

def fk_rot(lrot, parents):
    
    gr = [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gr.append(mul(gr[parents[i]], lrot[...,i:i+1,:]))
        
    return np.concatenate(gr, axis=-2)


def ik_rot(grot, parents):
    print(grot.shape)

    lr = grot.copy()
    for i in range(1, len(parents)):
        index= len(parents) - i
        lr[...,index,:] = mul(inv(grot[...,parents[index],:]), grot[...,index,:])
    return lr
    # return np.concatenate([grot[...,:1,:], mul(inv(grot[...,parents[1:],:]), grot[...,1:,:]),
    #     ], axis=-2)

def mirror_rot_trans(
    lrot,
    lpos,
    trans,
    names,
    parents,
    mirror_axis
):
    # print('mirror axis', mirror_axis)
    lower_names = [n.lower() for n in names]
    joints_mirror = np.array([(
        lower_names.index("left"+n[5:]) if n.startswith("right") else
        lower_names.index("right"+n[4:]) if n.startswith("left") else 
        lower_names.index(n)) for n in lower_names])

    mirror_pos = np.array(mirror_axis)
    mirror_rot = np.array([1,] + [-ax for ax in mirror_axis])
    # mirror_rot = np.array([1, -1, -1, 1])
    # print('mirror rot', mirror_rot)
    grot = fk_rot(lrot, parents)
    # print(lrot.shape)

    ########################################################################################
    # Reference: https://stackoverflow.com/questions/32438252/efficient-way-to-apply-mirror-effect-on-quaternion-rotation
    ########################################################################################
    # from scipy.spatial.transform import Rotation as R
    # mirror_matrix = np.diag([-1,1,1])
    # # print(mirror_matrix)

    # T = grot.shape[0]
    # J = grot.shape[1]

    # print(R.from_quat(grot[:,joints_mirror].reshape(-1,4), scalar_first=True).as_matrix().shape)
    # grot_matrix = quat.to_xform(grot[:,joints_mirror].reshape(-1,4))
    # grot_matrix_mirror = mirror_matrix @ grot_matrix @ mirror_matrix
    # grot_mirror = quat.from_xform(grot_matrix_mirror).reshape(T,J,4)
    # print(grot_matrix[0], grot_matrix_mirror[0])
    # print(grot_mirror.as_euler('ZYX', degrees=True)[0])
    # print(grot_mirror[0][0], grot[:,joints_mirror][0][0])
    # print(quat.to_euler(grot_mirror)[0][0]/3.14*180)
    # 0.943466 -0.030603 6.685755

    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:,joints_mirror]
    lrot_mirror = ik_rot(grot_mirror, parents)
    # lrot[:, :, :] = mirror_rot * lrot[:, :, :]
    # lrot = lrot[:,joints_mirror]
    # print('parents', parents)
    # lpos_mirror = lpos[:,joints_mirror]
    # lpos_mirror[...,0] *= -1
    # print('lpos', lpos, 'lpos_mirror', lpos_mirror)

    # pos = quat.fk(lrot, lpos, parents)[1]
    # pos_mirror = quat.fk(lrot_mirror, lpos, parents)[1]
    # print(pos-pos[:,:1,:], (pos_mirror-pos_mirror[:,:1,:])[:,joints_mirror])
    
    return lrot_mirror, trans_mirror
