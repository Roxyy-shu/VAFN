import os
from .base_modality import Modality
import numpy as np
from scipy import stats
import glob

class EDaicWozAudioEgemaps(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'audio_egemaps'
        self.modality_mask_file = 'no_voice_idxs.npz'
        self.video_ref_modality = "video_pose_gaze_aus"
        super().__init__(args)

class EDaicWozAudioMfcc(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'audio_mfcc'
        self.modality_mask_file = 'no_voice_idxs.npz'
        self.video_ref_modality = "video_pose_gaze_aus"
        super().__init__(args)

class EDaicWozVideoResnet(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'video_cnn_resnet'
        self.modality_mask_file = 'no_face_idxs.npz'
        self.video_ref_modality = "video_pose_gaze_aus"
        super().__init__(args)

class EDaicWozPoseGazeAus(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'video_pose_gaze_aus'
        self.modality_mask_file = 'no_face_idxs.npz'
        self.video_ref_modality = "video_pose_gaze_aus"
        super().__init__(args)
