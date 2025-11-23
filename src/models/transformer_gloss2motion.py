import torch
import torch.nn as nn
import math

class Gloss2MotionTransformer(nn.Module):
    def __init__(
        self,
        gloss_vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        max_gloss_len=30,
        max_motion_len=60,
        num_joints=60,   # J
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_gloss_len = max_gloss_len
        self.max_motion_len = max_motion_len
        self.num_joints = num_joints
        self.out_dim = num_joints * 3  # (x,y,z)

        # 1) Gloss encoder
        self.gloss_embedding = nn.Embedding(gloss_vocab_size, d_model)
        self.gloss_pos_embedding = nn.Embedding(max_gloss_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 2) Motion decoder
        self.time_embedding = nn.Embedding(max_motion_len, d_model)
        self.motion_pos_embedding = nn.Embedding(max_motion_len, d_model)  # time과 같게 써도 됨

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 3) 출력 projection
        self.output_proj = nn.Linear(d_model, self.out_dim)

    def forward(self, gloss_ids, gloss_mask, motion_len=None):
        """
        gloss_ids: (B, N)
        gloss_mask: (B, N)  True=패딩 위치 or 반대로 할 수도 있음

        v1: motion_len는 모두 동일 T라고 가정, None이면 max_motion_len 사용
        return: motion_pred (B, T, J, 3)
        """
        B, N = gloss_ids.shape

        # ----- Encoder -----
        pos_idx = torch.arange(N, device=gloss_ids.device).unsqueeze(0)  # (1,N)
        gloss_emb = self.gloss_embedding(gloss_ids) + self.gloss_pos_embedding(pos_idx)  # (B,N,d)

        # Transformer mask 형식 맞게 바꿔야 함 (True=ignore 등)
        # 여기선 gloss_mask: (B,N)에서 (B,1,1,N) 형태로 바꾸는 식으로
        enc_key_padding_mask = gloss_mask  # (B,N)
        memory = self.encoder(gloss_emb, src_key_padding_mask=enc_key_padding_mask)  # (B,N,d)

        # ----- Decoder -----
        if motion_len is None:
            T = self.max_motion_len
        else:
            T = motion_len

        time_idx = torch.arange(T, device=gloss_ids.device).unsqueeze(0)  # (1,T)
        # 간단하게 time_embedding + pos_embedding 사용
        dec_input = self.time_embedding(time_idx) + self.motion_pos_embedding(time_idx)  # (1,T,d)
        dec_input = dec_input.repeat(B, 1, 1)  # (B,T,d)

        # cross-attention: query=dec_input, key/value=memory
        # padding mask: encoder 쪽 것만 있으면 됨
        tgt = dec_input
        # TransformerDecoder는 tgt_key_padding_mask(디코더 패딩)도 받을 수 있는데 v1에선 없음
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=enc_key_padding_mask
        )  # (B,T,d)

        motion_flat = self.output_proj(out)  # (B,T, J*3)
        motion = motion_flat.view(B, T, self.num_joints, 3)

        return motion
