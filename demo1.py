"""
Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png``` and a corrected version in ```examples/im1010_shape_corrected.png```.
"""

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
from scipy.spatial.transform import Rotation as R

from models import hmr, SMPL
from utils.imutils import crop
# from utils.renderer import Renderer
import config
import constants
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import logging
import os

# Set up basic logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')
parser.add_argument('--img_list', nargs='+', default=None, help='List of input image paths')

def transform_vertices(vertices, camera_translation):
    return vertices + camera_translation

def show_mesh(vertices, faces, color=(0.8, 0.8, 1.0), elev=30, azim=45, alpha=1.0, show=True, outfile=None):
    """Simple 3D mesh viewer using matplotlib."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(vertices[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    # Auto scale to the mesh size
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Center the mesh
    mesh_center = vertices.mean(axis=0)
    ax.set_xlim(mesh_center[0] - 0.5, mesh_center[0] + 0.5)
    ax.set_ylim(mesh_center[1] - 0.5, mesh_center[1] + 0.5)
    ax.set_zlim(mesh_center[2] - 0.5, mesh_center[2] + 0.5)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    if outfile:
        plt.savefig(outfile, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)




def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    scale = bbox_size / 200.0 * rescale
    return center, scale

def bbox_from_json(bbox_file):
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    return center, scale

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy()
    if bbox_file is None and openpose_file is None:
        height, width = img.shape[:2]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        center, scale = bbox_from_json(bbox_file) if bbox_file else bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint, weights_only=False,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'], strict=False)

    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(device)
    model.eval()
    #renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

    img, norm_img = process_image(args.img, args.bbox, args.openpose, input_res=constants.IMG_RES)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
        print(pred_betas)
        print(pred_rotmat)
        print(pred_camera)
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        #pred_output = smpl(betas=pred_betas, global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

    camera_translation = torch.stack([
        pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:,0] + 1e-9)
    ], dim=-1)[0].cpu().numpy()

    pred_vertices = pred_vertices[0].cpu().numpy()
    print("predicted vertices",pred_vertices)
    img_np = img.permute(1,2,0).cpu().numpy()

    vertices_cam = transform_vertices(pred_vertices, camera_translation)


    # Visualize the predicted mesh in 3D
    show_mesh(vertices_cam, smpl.faces,outfile=args.outfile if args.outfile else "mesh_original.png")


    #print("predicted vertices",pred_vertices)

    #renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

    # Render original mesh
    #img_shape = renderer(pred_vertices, camera_translation, img_np)

    # Detect posture errors and correct
    corrected_vertices = pred_vertices.copy()

    # Shoulder imbalance detection and correction
    left_shoulder_y = pred_vertices[3651, 1]
    right_shoulder_y = pred_vertices[4532, 1]
    shoulder_diff = left_shoulder_y - right_shoulder_y
    if abs(shoulder_diff) > 0.02:
        mean_y = (left_shoulder_y + right_shoulder_y) / 2
        corrected_vertices[3651, 1] = mean_y
        corrected_vertices[4532, 1] = mean_y

    # Forward head posture correction
    head_z = np.mean(pred_vertices[4110:4140, 2])
    neck_z = pred_vertices[3500, 2]
    if head_z - neck_z > 0.04:
        corrected_vertices[4110:4140, 2] -= 0.03

    # Rounded/arched back correction
    lower_back_z = np.mean(pred_vertices[3500:3600, 2])
    hip_z = np.mean(pred_vertices[[2222, 3022], 2])
    if lower_back_z < hip_z - 0.03:
        corrected_vertices[3500:3600, 2] += 0.03

    # Uneven hips detection and correction
    left_hip_y = pred_vertices[2222, 1]
    right_hip_y = pred_vertices[3022, 1]
    hip_diff = left_hip_y - right_hip_y
    if abs(hip_diff) > 0.02:
        mean_hip_y = (left_hip_y + right_hip_y) / 2
        corrected_vertices[2222, 1] = mean_hip_y
        corrected_vertices[3022, 1] = mean_hip_y

    # Slight torso tilt correction (rotation neutralization)
    upper_body_indices = list(range(3000, 6890))
    center = pred_vertices.mean(axis=0)
    correction_rot = R.from_euler('z', -10, degrees=True).as_matrix()
    corrected_vertices[upper_body_indices] = np.dot(
        (corrected_vertices[upper_body_indices] - center), correction_rot.T) + center



    vertices_cam = transform_vertices(corrected_vertices, camera_translation)



    show_mesh(vertices_cam, smpl.faces, color=(1, 0.7, 0.7),outfile="mesh_corrected.png")  # Pinkish

    print("corrected vertices",corrected_vertices)

    #img_shape_corrected = renderer(corrected_vertices, camera_translation, img_np)

    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center
    print("corrected vertices",rot_vertices)
    show_mesh(rot_vertices, smpl.faces, color=(0.6, 1, 0.6),outfile="mesh_side.png") 


         # Greenish

    # img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img_np))

    # outfile = args.img.split('.')[0] if args.outfile is None else args.outfile
    # cv2.imwrite(outfile + '_shape.png', 255 * img_shape[:,:,::-1])
    # cv2.imwrite(outfile + '_shape_side.png', 255 * img_shape_side[:,:,::-1])
    # cv2.imwrite(outfile + '_shape_corrected.png', 255 * img_shape_corrected[:,:,::-1])
