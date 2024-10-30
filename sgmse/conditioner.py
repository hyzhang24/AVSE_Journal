from sgmse.visual_module import VisualFrontend
import pytorch_lightning as pl
from .backbones.ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import torch
from torch.nn import functional as F


class ModalityFuse(pl.LightningModule): # compute conditioner

	"""
	A module that computes the fused audio and video features

	"""

	def __init__(self, feat_dim=256):
		super().__init__()
		self.visual_front_end = VisualFrontend(context_dim=feat_dim)
		decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=4, batch_first=True)
		self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers=4)


	def forward(self, audio_feat, video_feat):
		video_feat = self.visual_front_end(video_feat)
		fused_feat = self.decoder_layers(audio_feat, video_feat)
		return fused_feat