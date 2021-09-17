import torch


def edit(latents, pca, edit_directions, factor_index):
    edit_latents = []
    for latent in latents:
        for pca_idx, start, end, strength_range in edit_directions:
            strenths = list(strength_range)
            strength = strenths[factor_index] if len(strenths) > factor_index else 0
            edit_latents.append(edit_latent(pca, latent, pca_idx, start, end, strength))
    return torch.stack(edit_latents)

def edit_latent(pca, latent, pca_idx, start, end, strength):
    delta = get_delta(pca, latent, pca_idx, strength)
    delta_padded = torch.zeros(latent.shape).to('cuda')
    delta_padded[start:end] += delta.repeat(end - start, 1)
    return latent + delta_padded

def get_delta(pca, latent, idx, strength):
    # pca: ganspace checkpoint. latent: (16, 512) w+
    w_centered = latent - pca['mean'].to('cuda')
    lat_comp = pca['comp'].to('cuda')
    lat_std = pca['std'].to('cuda')
    w_coord = torch.sum(w_centered[0].reshape(-1)*lat_comp[idx].reshape(-1)) / lat_std[idx]
    delta = (strength - w_coord)*lat_comp[idx]*lat_std[idx]
    return delta
