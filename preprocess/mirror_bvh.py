import os
import utils.bvh as bvh


data_dir = '../data/motion/xsens-mocap'
mocap_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.bvh') and 'XYZ' not in f and 'mirror' not in f])
print(mocap_files)
# mocap_files = ["../data/xsens-mocap/walk8_fast.bvh"]

for f in mocap_files:
    # Get the mirror file name
    mirror_f = f.split('.bvh')[0] + '_mirror.bvh'
    print(mirror_f)

    # Load an animation from a BVH file
    anim = bvh.load(filepath=f)
    # Skel offsets must be symmetric!!!

    # Mirror the animation
    anim_mirror = anim.mirror()

    # Save the animation
    bvh.save(
        filepath=mirror_f,
        anim=anim_mirror
    )
