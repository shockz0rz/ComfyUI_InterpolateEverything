import numpy as np
import cv2
from .util import common_annotator_call, img_np_to_tensor, img_tensor_to_np, HWC3, resize_image 
from .v11 import openpose_v11

def interpolate_body_poses( start_poses, end_poses, interp_factor ): 
    point_count = len (start_poses['subset'][0]) - 2 # probably 18

    # this rules out trippy merges of bodies for now, sadly
    pose_count = min(len(start_poses['subset']), len(end_poses['subset']))

    # Each row of subset is a pose, and the first 18 columns are the indices of the points for that pose in candidate
    # (the last two columns are the score and the number of detected points)
    # The index in subset is the ID of the point, so out_candidate[out_subset[0][5]] is the coordinate for the left shoulder of the first pose
    # Initialize output subset to no points found, score 0
    empty_subset = [-1.0 for i in range(point_count)] + [0.0, 0.0]
    out_subset = [empty_subset for i in range(pose_count)]

    # Initialize output candidate to empty
    out_candidate = []

    start_candidate = start_poses['candidate']
    end_candidate = end_poses['candidate']

    for pose_i in range(pose_count):
        start_subset = start_poses['subset'][pose_i]
        end_subset = end_poses['subset'][pose_i]
        for point_i in range(point_count):
            if start_subset[point_i] == -1.0 and end_subset[point_i] == -1.0:
                # point missing from both poses, out_subset already initialized to represent this, just keep going
                continue 
            # if here, we have at least one point found, increment point count in out_subset
            out_subset[pose_i][-1] += 1.0
            out_subset[pose_i][point_i] = float(len(out_candidate))
            if start_subset[point_i] == -1.0: #point missing from start pose, use end pose
                #possible enhancement: infer likely location of missing point from known points?
                out_candidate.append(end_candidate[int(end_subset[point_i])])
                continue
            if end_subset[point_i] == -1.0: #point missing from end pose, use start pose
                out_candidate.append(start_candidate[int(start_subset[point_i])])
                continue
                
            # otherwise, we have both start and end points, interpolate
            # x and y are a weighted average as you'd expect
            outX = (start_candidate[int(start_subset[point_i])][0] * (1.0 - interp_factor)) + (end_candidate[int(end_subset[point_i])][0] * interp_factor)
            outY = (start_candidate[int(start_subset[point_i])][1] * (1.0 - interp_factor)) + (end_candidate[int(end_subset[point_i])][1] * interp_factor)
            out_candidate.append([outX, outY])
        # set score to weighted average of start and end scores
        out_subset[pose_i][-2] = (start_subset[-2] * (1.0 - interp_factor)) + (end_subset[-2] * interp_factor)
    
    return {'candidate': out_candidate, 'subset': out_subset}

class OpenPose_Preprocessor_Interpolate:
    @classmethod
    def INPUT_TYPES(s):
        ret = {"required": { "start_image": ("IMAGE", ), 
                             "end_image": ("IMAGE", ), 
                             "interp_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}) }}
        return ret

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_interpolated_pose"
    CATEGORY = "preprocessors/pose"

    # TODO: Implement hand and face interpolation
    def preprocess_interpolated_pose(self, start_image, end_image, interp_factor):
        start_tensor_image = resize_image(HWC3(img_tensor_to_np(start_image)[0]), resolution=512)
        end_tensor_image = resize_image(HWC3(img_tensor_to_np(end_image)[0]), resolution=512)
        start_result = openpose_v11.OpenposeDetector()(start_tensor_image, hand=False, body=True, face=False, return_is_index=True)
        end_result = openpose_v11.OpenposeDetector()(end_tensor_image, hand=False, body=True, face=False, return_is_index=True)

        body_start = start_result['bodies']
        body_end = end_result['bodies']
        out_poses = {'bodies': interpolate_body_poses(body_start, body_end, interp_factor), 'faces': None, 'hands': None}
        out_width = max(start_tensor_image.shape[1], end_tensor_image.shape[1])
        out_height = max(start_tensor_image.shape[0], end_tensor_image.shape[0])

        result = openpose_v11.draw_pose(out_poses, out_height, out_width, draw_body=True, draw_face=False, draw_hand=False)

        return (img_np_to_tensor([cv2.resize(HWC3(result), (out_width, out_height), interpolation=cv2.INTER_AREA)]),)

NODE_CLASS_MAPPINGS = {
    "OpenposePreprocessorInterpolate": OpenPose_Preprocessor_Interpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenposePreprocessorInterpolate": "Interpolate Poses",
}