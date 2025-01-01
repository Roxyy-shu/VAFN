import models
MODELS = {
    'baseline': models.VAFN,
}

import datasets
DATASETS = {

    'e-daic-woz': datasets.EDaicWozDataset,
    'e-daic-woz-eval': datasets.EDaicWozEvaluationDataset,

}

MODALITY_ENCODERS = {

    'edaic_audio_egemaps': models.NoOpEncoder,
    'edaic_audio_mfcc': models.NoOpEncoder,
    'edaic_video_cnn_resnet': models.NoOpEncoder,
    'edaic_video_pose_gaze_aus': models.NoOpEncoder,

}

MODALITIES = {
    
    'edaic_audio_egemaps': datasets.modalities.EDaicWozAudioEgemaps,
    'edaic_audio_mfcc': datasets.modalities.EDaicWozAudioMfcc,
    'edaic_video_cnn_resnet': datasets.modalities.EDaicWozVideoResnet,
    'edaic_video_pose_gaze_aus': datasets.modalities.EDaicWozPoseGazeAus,

}

import trainers
TRAINERS = {
    'classification': trainers.Trainer,
}

import evaluators
EVALUATORS = {
    'temporal_evaluator': evaluators.TemporalEvaluator,
}

