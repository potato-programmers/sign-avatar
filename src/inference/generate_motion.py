def generate_motion(model, gloss_vocab, gloss_seq, device, target_T):
    gloss_ids, gloss_len = gloss_vocab.encode(gloss_seq, max_len=... )
    gloss_ids = gloss_ids.unsqueeze(0).to(device)
    gloss_mask = (gloss_ids == gloss_vocab.pad_id)

    with torch.no_grad():
        motion = model(gloss_ids, gloss_mask)  # (1,T,J,3)

    return motion.squeeze(0).cpu().numpy()
