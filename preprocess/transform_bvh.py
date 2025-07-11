import bpy
import numpy as np
import os
from os import listdir

data_path = '../data/motion/xsens-mocap'

files = sorted([f for f in listdir(data_path) if f.endswith(".bvh") and '_XYZ' not in f])
print(files, len(files))

for f in files:
    source_path = os.path.join(data_path, f)
    dump_path = source_path[:-4] + '_XYZ.bvh'

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.import_anim.bvh(filepath=source_path, update_scene_fps=True)

    frame_start = 9999
    frame_end = -9999
    action = bpy.data.actions[-1]
    if action.frame_range[1] > frame_end:
        frame_end = action.frame_range[1]
    if action.frame_range[0] < frame_start:
        frame_start = action.frame_range[0]

    frame_end = np.max([1, frame_end])
    bpy.ops.export_anim.bvh(filepath=dump_path,
                            frame_start=int(frame_start),
                            frame_end=int(frame_end),
                            rotate_mode='XYZ',
                            root_transform_only=True)
    bpy.data.actions.remove(bpy.data.actions[-1])

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    print(source_path + " processed.")
