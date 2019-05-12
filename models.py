import torch
import torch.nn as nn
import torch.nn.functional as F

from glow import Glow


class BeautyGlow(nn.Module):

    def __init__(self):
        self.w = nn.Linear(128, 128, bias=False)
        self.glow = Glow(3, 32, 4, affine=True, conv_lu=True)

    def forward(self, reference, source, l_x, l_y):
        l_ref = self.glow.reverse(reference)
        l_source = self.glow.reverse(source)
        f_ref = self.w(l_ref)
        f_source = self.w(l_souece)
        m_ref = F.linear(l_ref, torch.eye(128) - self.w.weight)
        m_source =F.linear(l_source, torch.eye(128) - self.w.weight)
        l_source_y = m_ref + f_source
        print(l_source_y)
        result = self.glow(l_source)

        perceptual_loss = F.mse_loss(f_ref, l_source)

        makeup_loss = F.mse_loss(m_ref, l_y - l_x)

        intra_domain_loss = F.mse_loss(f_ref, l_x) + F.mse_loss(l_source, l_y)

        l2_norm_f = F.mse_loss(f_ref, torch.zeros(f_ref.size())) * \
            F.mse_loss(l_y, torch.zeros(l_y.size()))
        sim_f = torch.sum(f_ref * l_y) / l2_norm_f
        l2_norm_l = F.mse_loss(l_source, torch.zeros(l_source.size())) * \
            F.mse_loss(l_x, torch.zeros(l_x.size()))
        sim_l = torch.sum(l_source * l_x) / l2_norm_l
        inter_domain_loss = 1 + sim_f + 1 + sim_l

        cycle_f = F.mse_loss(self.w(l_source_y), f_source)
        cycle_m = F.mse_loss(F.linear(l_source_y, torch.eye(128) - self.w.weight, m_ref))
        cycle_consistency_loss = cycle_f + cycle_m

        perceptual = 0.01
        cycle = 0.001
        makeup = 0.1
        intra = 0.1
        inter = 1000

        loss = perceptual_loss + cycle * cycle_consistency_loss + makeup * makeup_loss\
            + intra * intra_domain_loss + inter * inter_domain_loss

        return result, loss
