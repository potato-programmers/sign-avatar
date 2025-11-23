# 역할: Gloss2MotionTransformer 모델 클래스 구현.

# 포함할 내용:

# class Gloss2MotionTransformer(nn.Module)

# __init__: vocab_size, d_model, etc. 받아서 encoder/decoder 구성

# forward(gloss_ids, gloss_mask, motion_len=None): (B,T,J,3) 리턴

# 여기에는 아까 내가 써준 Transformer 구조를 그대로 넣으면 돼.