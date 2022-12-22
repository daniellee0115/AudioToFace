import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec import Wav2Vec
from fast_transformers.builders import TransformerDecoderBuilder
import fast_transformers

# temporal mask, modeled off of ALiBi -- compiled from snippets in https://github.com/ofirpress/attention_with_linear_biases/issues/5
def alibi_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Alignment Bias
def alignment_bias(device, dataset, T, S):
    """Construct alignment bias mask"""

    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, i] = 0

    return (mask==1).to(device=device)


# Overwrite fast transformers mask so that it is compatible with our use case (float mask instead of boolen)
class Mask(fast_transformers.masking.FullMask):
    def __init__(self, mask=None, N=None, M=None, device="cuda"):
        # mask is a tensor so we ignore N and M
        if mask is not None and isinstance(mask, torch.Tensor):
            with torch.no_grad():
                self._mask = mask.clone()
            return

        # mask is an integer, N is an integer and M is None so assume they were
        # passed as N, M
        if mask is not None and M is None and isinstance(mask, int):
            M = N
            N = mask

        if N is not None:
            M = M or N
            with torch.no_grad():
                self._mask = torch.ones(N, M, dtype=torch.bool, device=device)
            self._all_ones = True
            return

        raise ValueError("Either mask or N should be provided")

    @property
    def additive_matrix(self):
        if not hasattr(self, "_additive_matrix"):
            with torch.no_grad():
                self._additive_matrix = self._mask
        return self._additive_matrix


class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        encoding = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)

        inside = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))     # t mod p / 10000^(2i/d)
        encoding[:, 0::2] = torch.sin(position * inside)
        encoding[:, 1::2] = torch.cos(position * inside)

        encoding = encoding.unsqueeze(0)    # encoding.shape = (1, period, d_model)
        encoding = encoding.repeat(1, max_seq_len // period + 1, 1)     # repeat period across sequence length

        self.register_buffer('encoding', encoding)


    def forward(self, x):
        x = x + self.encoding[:, :x.shape[1], :]

        return self.dropout(x)


class Faceformer(nn.Module):
    def __init__(self, args):
        super(Faceformer, self).__init__()
        """
        features:
            audio: (batch_size, r)
            template: (batch_size, 5023*3)
            vertice: (batch_size, seq_len, 5023*3)
        """
        self.dataset = args.dataset
        self.audio_encoder = Wav2Vec.from_pretrained("facebook/wav2vec2-base-960h")

        # audio encoder
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, args.feature_dim)

        # motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)

        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)

        # temporal bias
        self.biased_mask = alibi_mask(n_head = 4, max_seq_len = 600, period=args.period)
        
        # use fast transformers instead of normal transformers.
        decoder_layer = fast_transformers.transformers.TransformerDecoderLayer(fast_transformers.attention.attention_layer.AttentionLayer(fast_transformers.attention.full_attention.FullAttention(), args.feature_dim, 4), fast_transformers.attention.attention_layer.AttentionLayer(fast_transformers.attention.full_attention.FullAttention(), args.feature_dim, 4), d_model = args.feature_dim, d_ff = 2 * args.feature_dim)
        self.transformer_decoder = fast_transformers.transformers.TransformerDecoder([decoder_layer])

        # motion decoder
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
       
        # style embedding
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        self.device = args.device

        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, audio, template, vertice, one_hot, criterion,teacher_forcing=True):
        template = template.unsqueeze(1)
        obj_embedding = self.obj_vector(one_hot)
        frame_num = vertice.shape[1]

        # audio encoding
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        hidden_states = self.audio_feature_map(hidden_states)

        # motion encoding + cross-modal attention, applied per frame
        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            tgt_mask = Mask(tgt_mask)
            memory_mask = alignment_bias(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            memory_mask = fast_transformers.masking.FullMask(memory_mask)
            vertice_out = self.transformer_decoder.forward(x = vertice_input, memory = hidden_states, x_mask = tgt_mask, memory_mask = memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        return vertice_out

    def predict(self, audio, template, one_hot):
        template = template.unsqueeze(1)
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)

        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            tgt_mask = Mask(tgt_mask)
            memory_mask = alignment_bias(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            memory_mask = fast_transformers.masking.FullMask(memory_mask)
            vertice_out = self.transformer_decoder.forward(x = vertice_input, memory = hidden_states, x_mask = tgt_mask, memory_mask = memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        return vertice_out
