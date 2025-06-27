# Add to top if calling from worker


def run_demo(img_path, openpose_path=None, bbox_path=None, out_path="mesh_original.png"):
    import torch
    from torchvision.transforms import Normalize
    import numpy as np
    import cv2
    import json
    from scipy.spatial.transform import Rotation as R
    from models import hmr, SMPL
    from utils.imutils import crop
    import config
    import constants
    from PIL import Image
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from ultralytics import YOLO
    import json
    import os
    import io


    def generate_openpose_json(image_path, output_json_path="000000_openpose.json"):
        print("image path",image_path)
        print("📂 Exists:", os.path.exists(image_path))
        img = cv2.imread(image_path)[:,:,::-1].copy()
        model = YOLO('yolov8s-pose.pt')  # download if not present
        results = model(image_path)
        print(results)


        if len(results[0].keypoints.xy) == 0:
            print("❌ No people detected.")
            return None

        keypoints_2d = results[0].keypoints.xy[0]
        pose_keypoints_2d = []
        for (x, y), conf in zip(keypoints_2d.tolist(), results[0].keypoints.conf[0].tolist()):
            pose_keypoints_2d.extend([x, y, float(conf)])

        pose_json = {
            "version": 1.2,
            "people": [{
                "pose_keypoints_2d": pose_keypoints_2d,
                "face_keypoints_2d": [0.0] * 210,
                "hand_left_keypoints_2d": [0.0] * 63,
                "hand_right_keypoints_2d": [0.0] * 63,
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }]
        }

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(pose_json, f)
        print(f"✅ Saved keypoints to {output_json_path}")
        return output_json_path


    def transform_vertices(vertices, camera_translation):
        return vertices + camera_translation



    def show_mesh3(vertices, faces, color=(0.8, 0.8, 1.0),
              elev=30, azim=45, alpha=1.0,
              show=False, outfile=None, return_bytes=True):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(vertices[faces], alpha=alpha)
        mesh.set_facecolor(color)
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        
        scale = vertices.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        mesh_center = vertices.mean(axis=0)
        ax.set_xlim(mesh_center[0] - 0.5, mesh_center[0] + 0.5)
        ax.set_ylim(mesh_center[1] - 0.5, mesh_center[1] + 0.5)
        ax.set_zlim(mesh_center[2] - 0.5, mesh_center[2] + 0.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        if outfile:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            with open(outfile, 'wb') as f:
                f.write(buf.getvalue())
            print(f"✅ Mesh image saved to: {outfile}")

        if show:
            plt.show(block=True)

        plt.close(fig)

        if return_bytes:
            return buf.getvalue()





    def show_mesh2(vertices, faces, color=(0.8, 0.8, 1.0), elev=30, azim=45, alpha=1.0, show=True, outfile=None):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(vertices[faces], alpha=alpha)
        mesh.set_facecolor(color)
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        
        scale = vertices.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        mesh_center = vertices.mean(axis=0)
        ax.set_xlim(mesh_center[0] - 0.5, mesh_center[0] + 0.5)
        ax.set_ylim(mesh_center[1] - 0.5, mesh_center[1] + 0.5)
        ax.set_zlim(mesh_center[2] - 0.5, mesh_center[2] + 0.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        if outfile:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            plt.show(block=True)
            plt.savefig(outfile, bbox_inches='tight')
            print(f"✅ Mesh image saved to: {outfile}")

        # if show:
        #     plt.show(block=True)
        
        plt.close(fig)


    def show_mesh(vertices, faces, color=(0.8, 0.8, 1.0), elev=30, azim=45, alpha=1.0, show=False, outfile=None):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(vertices[faces], alpha=alpha)
        mesh.set_facecolor(color)
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        scale = vertices.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        mesh_center = vertices.mean(axis=0)
        ax.set_xlim(mesh_center[0] - 0.5, mesh_center[0] + 0.5)
        ax.set_ylim(mesh_center[1] - 0.5, mesh_center[1] + 0.5)
        ax.set_zlim(mesh_center[2] - 0.5, mesh_center[2] + 0.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        if outfile: plt.savefig(outfile, bbox_inches='tight')
        if show: plt.show()
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
        print("in process image")
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

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load("data/model_checkpoint.pt" , weights_only=False,map_location=torch.device('cpu'))
    print("succeeded")
    #checkpoint = torch.load(config.HMR_CHECKPOINT, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    print("succeeded")
    model.eval()
    print("succeeded")
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(device)
    print("succeeded")
    # Derive the directory and JSON path
    img_dir = os.path.dirname(img_path)
    openpose_path = os.path.join(img_dir, "output_keypoints.json")  # or change name as needed
    print(openpose_path)
    # openpose_path = img_path + "/" + "openpose.json"
    openpose_path = generate_openpose_json(img_path, openpose_path)

    img, norm_img = process_image(img_path, bbox_path, openpose_path, input_res=constants.IMG_RES)
    print("succeeded")


    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

    camera_translation = torch.stack([
        pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:,0] + 1e-9)
    ], dim=-1)[0].cpu().numpy()

    pred_vertices = pred_vertices[0].cpu().numpy()
    img_np = img.permute(1,2,0).cpu().numpy()
    vertices_cam = transform_vertices(pred_vertices, camera_translation)

    # Corrections here (unchanged)...

    mesh_bytes = show_mesh3(vertices_cam, smpl.faces, outfile=out_path, show=False)


    print("mesh bytes",mesh_bytes)

    return  mesh_bytes


# CLI support (optional)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=False, help='Path to pretrained checkpoint (unused, pulled from config)')
    parser.add_argument('--img', type=str, required=True, help='Path to input image')
    parser.add_argument('--bbox', type=str, default=None)
    parser.add_argument('--openpose', type=str, default=None)
    parser.add_argument('--outfile', type=str, default="mesh_original.png")
    args = parser.parse_args()

    run_demo(args.img, args.openpose, args.bbox, args.outfile)
