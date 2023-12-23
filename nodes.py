import math
import torch
import numpy as np
import comfy.model_management as model_management
from comfyui_controlnet_aux.utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
from comfyui_controlnet_aux.src.controlnet_aux.util import HWC3, resize_image_with_pad
from comfyui_controlnet_aux.src.controlnet_aux import open_pose
from typing import List, Tuple
from .icp_utils import icp_2d

# Guess which pose in start_poses corresponds to which pose in end_poses by comparing distances between means 
# TODO: Consider face and hand data when matching
def match_poses( start_poses: List[open_pose.PoseResult], end_poses: List[open_pose.PoseResult] ) -> List[Tuple[int, int]]:
    pose_count : int = min(len(start_poses), len(end_poses))
    pose_matches_final : List[Tuple[int, int]] = []
    pose_match_candidates : List[Tuple[float, int, int]] = []
    for start_pose in start_poses:
        for end_pose in end_poses:
            start_mean_x = np.mean([keypoint.x for keypoint in start_pose.body.keypoints if keypoint is not None])
            start_mean_y = np.mean([keypoint.y for keypoint in start_pose.body.keypoints if keypoint is not None])
            end_mean_x = np.mean([keypoint.x for keypoint in end_pose.body.keypoints if keypoint is not None])
            end_mean_y = np.mean([keypoint.y for keypoint in end_pose.body.keypoints if keypoint is not None])
            distance = math.sqrt((start_mean_x - end_mean_x)**2 + (start_mean_y - end_mean_y)**2)
            pose_match_candidates.append((distance, start_poses.index(start_pose), end_poses.index(end_pose)))

    # Sort by distance, then add to pose_matches_final if neither pose is already matched
    pose_match_candidates.sort(key=lambda x: x[0])
    for i in range(min(pose_count, len(pose_match_candidates))):
        if pose_match_candidates[i][1] not in [x[0] for x in pose_matches_final] and pose_match_candidates[i][2] not in [x[1] for x in pose_matches_final]:
            pose_matches_final.append((pose_match_candidates[i][1], pose_match_candidates[i][2]))
    return pose_matches_final

def interpolate_poses( start_poses: List[open_pose.PoseResult], end_poses: List[open_pose.PoseResult], interp_factor: float, omit_missing_points: bool ) -> List[open_pose.PoseResult]:
    matches = match_poses(start_poses, end_poses)
    interpolated_poses = []
    for pose_match in matches:
        body_result = interpolate_body_pose(start_poses[pose_match[0]].body, end_poses[pose_match[1]].body, interp_factor, omit_missing_points)
        interp_pose = open_pose.PoseResult(body = body_result, left_hand = None, right_hand = None, face = None)
        interpolated_poses.append(interp_pose)
    
    return interpolated_poses

def interpolate_body_pose( start_pose: open_pose.BodyResult, end_pose: open_pose.BodyResult, interp_factor: float, omit_missing_points: bool ) -> open_pose.BodyResult: 
    body_keypoints = []
    score_sum = 0.0
    for start_keyp, end_keyp in zip(start_pose.keypoints, end_pose.keypoints):
        if start_keyp is None and end_keyp is None:
            body_keypoints.append(None)
        elif start_keyp is None and end_keyp is not None:
            if omit_missing_points:
                body_keypoints.append(None)
            else:
                body_keypoints.append(end_keyp)
        elif start_keyp is not None and end_keyp is None:
            if omit_missing_points:
                body_keypoints.append(None)
            else:
                body_keypoints.append(start_keyp)
        else:
            keyp_id = start_keyp.id
            keyp_x = (start_keyp.x * (1.0 - interp_factor)) + (end_keyp.x * interp_factor)
            keyp_y = (start_keyp.y * (1.0 - interp_factor)) + (end_keyp.y * interp_factor)
            keyp_score = 0.5 * (start_keyp.score + end_keyp.score)
            score_sum += keyp_score
            interp_keyp = open_pose.Keypoint(x = keyp_x, y = keyp_y, score = keyp_score, id = keyp_id)
            body_keypoints.append(interp_keyp)
    
    return open_pose.BodyResult(keypoints = body_keypoints, total_score = score_sum, total_parts=len(body_keypoints))


def get_face_transform(start_face: List[List[float]], end_face:List[List[float]]) -> List[float]:
    start_np = np.array(start_face)
    end_np = np.array(end_face)

    icp_2d(start_np, end_np)


def interpolate_face_poses( start_faces, end_faces, interp_factor ):
    # Faces don't have any identifying indexes, they're just lists of points.
    # They're not even guaranteed to have the same number of points. This kind of sucks for what we're trying to do here.
    # We'll get the estimated transform necessary to convert each start-end pair, which in addition to getting us our necessary interpolation can also tell us
    # which face corresponds to which.
    transform_factors = []
    for start_i in range(len(start_faces)):
        transform_factors.append([])
        for end_i in range(len(end_faces)):
            if len(start_faces[start_i]) < 10 or len(end_faces[end_i]) < 10: # not enough points to estimate transform
                transform_factors[start_i].append([-1.0, -1.0, -1.0])
            transform_factors[start_i].append(get_face_transform(start_faces[start_i], end_faces[end_i])) #translation, rotation, scale, all on [0.0, 1.0] range
    
    return


class OpenPose_Preprocessor_Interpolate:
    @classmethod
    def INPUT_TYPES(s):
        ret = {"required": { "start_image": ("IMAGE", ), 
                             "end_image": ("IMAGE", ), 
                             "interp_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "omit_any_missing_points": ("BOOLEAN", {"default": False }), }}
        return ret

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_interpolated_pose"
    CATEGORY = "ControlNet Preprocessors/Interpolation"

    # TODO: Implement hand and face interpolation
    def preprocess_interpolated_pose(self, start_image, end_image, interp_factor, omit_any_missing_points):
        model = open_pose.OpenposeDetector.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())

        # TODO: handle batched inputs
        np_start_image = np.asarray(start_image[0] * 255., dtype=np.uint8)

        # We don't actually want the output of the model() call - that gives us a final image.
        # We want the output of the detect_poses() call, which gives us a list of PoseResult objects.
        np_start_resized, start_remove_pad = resize_image_with_pad(np_start_image, 512, "INTER_CUBIC")
        start_poses = model.detect_poses(np_start_resized, include_hand=False, include_face=False)

        np_end_image = np.asarray(end_image[0] * 255., dtype=np.uint8)
        np_end_resized, end_remove_pad = resize_image_with_pad(np_end_image, 512, "INTER_CUBIC")
        end_poses = model.detect_poses(np_end_resized, include_hand=False, include_face=False)

        interp_poses = interpolate_poses(start_poses, end_poses, interp_factor, omit_any_missing_points)

        np_start_resized = start_remove_pad( np_start_resized)
        np_end_resized = end_remove_pad( np_end_resized)
        final_h = max(np_start_resized.shape[0], np_end_resized.shape[0])
        final_w = max(np_start_resized.shape[1], np_end_resized.shape[1])

        out_image = HWC3( open_pose.draw_poses( interp_poses, final_h, final_w, draw_body=True, draw_face=False, draw_hand=False) )

        out_imgs = torch.stack([ torch.from_numpy(out_image.astype(np.float32) / 255.0 ) ], dim=0)

        del model

        return (out_imgs,)
        
NODE_CLASS_MAPPINGS = {
    "OpenposePreprocessorInterpolate": OpenPose_Preprocessor_Interpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenposePreprocessorInterpolate": "Interpolate Poses",
}