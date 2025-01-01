import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np
import librosa
from scipy.spatial.distance import pdist, squareform
import constants

def fractional_positional_encoding(batch_size, d_model, length, downscale_factor):
    pe = torch.zeros(batch_size, length, d_model).to(downscale_factor.device)

    position = torch.arange(0, length).unsqueeze(1).tile((batch_size, )).to(downscale_factor.device)
    position = position * (1 / downscale_factor)
    position = position.T.unsqueeze(-1)

    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *-(math.log(10000.0) / d_model)))
    div_term = div_term.to(downscale_factor.device)

    sin_positions = torch.sin(position * div_term)
    cos_positions = torch.cos(position * div_term)

    pe[:, :, 0::2] = sin_positions
    pe[:, :, 1::2] = cos_positions

    return pe

class NoOpEncoder(torch.nn.Module):
    def __init__(self, args, modality_encoder_args):
        super(NoOpEncoder, self).__init__()
        self.args = args
        self.modality_encoder_args = modality_encoder_args

        self.batch_norm  = torch.nn.BatchNorm1d(
            self.modality_encoder_args.input_dim,
        )

        self.projection = torch.nn.Linear(
            self.modality_encoder_args.input_dim,
            self.modality_encoder_args.model_args.latent_dim,
            bias = False,
        )

        self.is_audio = "audio" in self.modality_encoder_args.name

        if 'original' in self.args.dataset:
            max_fps = 1
        else:
            max_fps = constants.MAX_AUDIO_FPS if self.is_audio else constants.MAX_VIDEO_FPS

        self.max_data_length = max_fps * self.args.seconds_per_window

    def forward(self, data, mask, framerate_ratio):
        data = data.view(data.shape[0], -1, self.modality_encoder_args.input_dim)
        data = self.batch_norm(data.permute(0, 2, 1)).permute(0, 2, 1)
        data = self.projection(data)

        downscale_factor = torch.ones((data.shape[0], )).float().to(data.device) if self.is_audio else framerate_ratio
        pe = fractional_positional_encoding(batch_size = data.shape[0], d_model = self.modality_encoder_args.model_args.latent_dim, length = self.max_data_length, downscale_factor = downscale_factor)
        pe = pe.to(data.device)
        data = data + pe

        return data