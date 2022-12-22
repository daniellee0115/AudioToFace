"""Modified wav2vec implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput

from typing import Tuple, Optional


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """Resample our features from a given input fps into a distinct output fps."""
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)

    if output_len is None:
            output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features,size=output_len,align_corners=True,mode='linear')

    return output_features.transpose(1, 2)

    
class Wav2Vec(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, 
                inputs, 
                dataset, 
                attention_mask=None, 
                output_attentions=None, 
                output_hidden_states=None, 
                return_dict=None,
                frame_num=None
    ):
        self.config.output_attentions = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        hidden_states = self.feature_extractor(inputs)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = linear_interpolation(hidden_states, 50, 30, frame_num)
        hidden_states = self.feature_projection(hidden_states)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )