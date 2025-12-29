import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob 
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32, proxy_mode='only_real'):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        if proxy_mode == "only_real":
            self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
            print("Proxy Anchor Initialization: only real" )
        elif proxy_mode == "real_fake":
            self.proxies = torch.nn.Parameter(torch.randn(nb_classes*2, sz_embed).cuda())
            print("Proxy Anchor Initialization: real and fake")
        else:
            raise ValueError("proxy_mode must be either 'only_real' or 'real_fake'")
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.proxy_mode = proxy_mode
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        # Clamp cosine similarity to prevent overflow in exponentials
        # Cosine similarity is in [-1, 1], but we clamp more conservatively
        cos = torch.clamp(cos, -0.99, 0.99)
        
        # Map T to valid range [0, nb_classes-1] for binarize
        # label_binarize expects labels in [0, nb_classes), so we map to [0, nb_classes-1]
        T_mapped = T % self.nb_classes  # Map to [0, nb_classes-1]
        T_mapped = torch.clamp(T_mapped, 0, self.nb_classes - 1)  # Extra safety
        
        # Call binarize with nb_classes*2 - label_binarize will create one-hot vectors
        # For nb_classes=1, calling binarize(T=[0], nb_classes=2) creates [[1, 0]]
        # Then we take first nb_classes columns to get [[1]]
        P_one_hot = binarize(T = T_mapped, nb_classes = self.nb_classes*2)
        if self.proxy_mode == "only_real":
            P_one_hot = P_one_hot[:, :self.nb_classes]
        else:
            pass
        N_one_hot = 1 - P_one_hot
    
        # Clamp the arguments to exp to prevent overflow
        # For pos_exp: -alpha * (cos - mrg), we want this to be bounded
        # For neg_exp: alpha * (cos + mrg), we want this to be bounded
        pos_arg = -self.alpha * (cos - self.mrg)
        neg_arg = self.alpha * (cos + self.mrg)
        # Clamp to reasonable range to prevent overflow (exp(88) is near float32 max)
        pos_arg = torch.clamp(pos_arg, -88.0, 88.0)
        neg_arg = torch.clamp(neg_arg, -88.0, 88.0)
        
        pos_exp = torch.exp(pos_arg)
        neg_exp = torch.exp(neg_arg)

        # Find which proxies have positive samples
        P_one_hot_sum = P_one_hot.sum(dim = 0)
        with_pos_proxies = torch.nonzero(P_one_hot_sum != 0)
        if with_pos_proxies.dim() > 1:
            with_pos_proxies = with_pos_proxies.squeeze(dim = 1)
        elif with_pos_proxies.dim() == 0:
            with_pos_proxies = with_pos_proxies.unsqueeze(0)
        num_valid_proxies = with_pos_proxies.numel() if with_pos_proxies.numel() > 0 else 0
        
        # Handle case when no valid proxies (division by zero)
        if num_valid_proxies == 0:
            # Return zero loss if no valid proxies in batch
            # This can happen if writer IDs are not properly mapped to class indices
            return torch.tensor(0.0, device=X.device, requires_grad=True)
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        # Add small epsilon to prevent log(0)
        P_sim_sum = P_sim_sum + 1e-8
        N_sim_sum = N_sim_sum + 1e-8
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        if self.proxy_mode == "only_real":
            neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        elif self.proxy_mode == "real_fake":
            neg_term = torch.log(1 + N_sim_sum).sum() / (self.nb_classes*2)
        else:
            raise ValueError("proxy_mode must be either 'only_real' or 'real_fake'")
        loss = pos_term + neg_term
        
        # Check for NaN/Inf and replace with 0 if found
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=X.device, requires_grad=True)
        
        return loss


"""pen moving prediction and pen state classification losses"""
def get_pen_loss(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits, x1_data, x2_data,
                 pen_data):
    result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
    epsilon = 1e-10
    # result1 is the loss wrt pen offset
    result1 = torch.multiply(result0, z_pi)
    result1 = torch.sum(result1, 1, keepdims=True)
    result1 = - torch.log(result1 + epsilon)  # avoid log(0)

    fs = 1.0 - pen_data[:, 2]  # use training data for this
    fs = fs.reshape(-1, 1)
    # Zero out loss terms beyond N_s, the last actual stroke
    result1 = torch.multiply(result1, fs)
    loss_fn = torch.nn.CrossEntropyLoss()
    result2 = loss_fn(z_pen_logits, torch.argmax(pen_data, -1))
    return result1, result2 # result1: pen offset loss, result2: category loss

"""Normal distribution"""
def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    s1 = torch.clip(s1, 1e-6, 500.0)
    s2 = torch.clip(s2, 1e-6, 500.0)

    norm1 = torch.subtract(x1, mu1)  # Returns x1-mu1 element-wise
    norm2 = torch.subtract(x2, mu2)
    s1s2 = torch.multiply(s1, s2)

    z = (torch.square(torch.div(norm1, s1)) + torch.square(torch.div(norm2, s2)) -
         2 * torch.div(torch.multiply(rho, torch.multiply(norm1, norm2)), s1s2))
    neg_rho = torch.clip(1 - torch.square(rho), 1e-6, 1.0)
    result = torch.exp(torch.div(-z, 2 * neg_rho))
    denom = 2 * np.pi * torch.multiply(s1s2, torch.sqrt(neg_rho))
    result = torch.div(result, denom)
    return result