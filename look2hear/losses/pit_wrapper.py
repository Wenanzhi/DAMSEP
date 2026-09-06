from itertools import permutations
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
import torchaudio
from .feature import transforms

class RIMag_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        ret = (
            (output.real - target.real).abs()
            + (output.imag - target.imag).abs()
            + (output.abs() - target.abs()).abs()
        )
        return ret.mean()

class PITLossWrapper(nn.Module):
    def __init__(
        self, loss_func, pit_from="pw_mtx", perm_reduce=None, threshold_byloss=True, w_rev=0.0, w_recon=0.0, w_rir=0.0,
        stft_n_fft=512, stft_hop_length=128, stft_win_length=None, stft_center=True
    ):
        super().__init__()
        self.loss_func = loss_func
        self.pit_from = pit_from
        self.perm_reduce = perm_reduce
        self.threshold_byloss = threshold_byloss
        self.w_rev = w_rev
        self.w_recon = w_recon
        self.w_rir = w_rir
        if self.pit_from not in ["pw_mtx", "pw_pt", "perm_avg", "no_pit"]:
            raise ValueError(
                "Unsupported loss function type {} for now. Expected"
                "one of [`pw_mtx`, `pw_pt`, `perm_avg`]".format(self.pit_from)
            )
        self.spec_loss = RIMag_loss()

        # STFT params
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length if stft_win_length is not None else stft_n_fft
        self.stft_center = stft_center
        self.register_buffer("stft_window", torch.hann_window(self.stft_win_length), persistent=False)
        self.transforms=transforms(sr = 8000, n_fft = 512, win_len = 256, hop_len = 128, win_type = "sqrthann")

    # def forward(self, ests, est_derev, est_rir, derev_targets, targets, return_ests=False, reduce_kwargs=None, **kwargs):
    def forward(self, est_rev, ests, est_rir, targets, rev_targets, rir_target, return_ests=False, reduce_kwargs=None, **kwargs):
        B = targets.shape[0]
        n_src = targets.shape[1]
        device = targets.device

        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()

        if self.pit_from == "no_pit":
            # identity permutation: [0, 1, ..., n_src-1]
            batch_indices = torch.arange(n_src, device=device).unsqueeze(0).repeat(B, 1)

            # Evaluate the pairwise loss and keep the fixed-order diagonal.
            pw_loss = self.loss_func(ests, targets, **kwargs)

            # Pairwise loss: pw_loss = [B, N, N].
            if pw_loss.ndim == 3:
                aligned = pw_loss.diagonal(dim1=-2, dim2=-1)   # [B, N]
                min_loss = aligned.mean(dim=-1)                # [B]
            # Also accept a source-aligned [B, N] loss.
            elif pw_loss.ndim == 2:
                min_loss = pw_loss.mean(dim=-1)                # [B]
            # Also accept a scalar or per-batch [B] loss.
            else:
                min_loss = pw_loss if pw_loss.ndim == 1 else pw_loss.expand(B)

        elif self.pit_from == "pw_mtx":
            pw_loss = self.loss_func(ests, targets, **kwargs)
            min_loss, batch_indices = self.find_best_perm(
                pw_loss, perm_reduce=self.perm_reduce, **reduce_kwargs
            )

        elif self.pit_from == "pw_pt":
            pw_loss = self.get_pw_losses(self.loss_func, ests, targets, **kwargs)
            min_loss, batch_indices = self.find_best_perm(
                pw_loss, perm_reduce=self.perm_reduce, **reduce_kwargs
            )

        elif self.pit_from == "perm_avg":
            min_loss, batch_indices = self.best_perm_from_perm_avg_loss(
                self.loss_func, ests, targets, **kwargs
            )
            mean_loss = torch.mean(min_loss)
            if not return_ests:
                return mean_loss
            reordered = self.reordered_sources(ests, batch_indices)
            return mean_loss, reordered

        else:
            raise RuntimeError("unreachable")

        # 后面保持你原逻辑：用 batch_indices 去对齐辅助项
        reordered = self.reordered_sources(ests, batch_indices)

        if self.threshold_byloss:
            if min_loss[min_loss > -30].nelement() > 0:
                min_loss = min_loss[min_loss > -30]
        mean_loss = torch.mean(min_loss)
        loss_total = mean_loss

        # rev 频谱 RIMag_loss（先对齐 est_derev，再 STFT）
        if self.w_rev != 0.0:
            est_rev_aligned = self.reordered_sources(est_rev, batch_indices)
            # STFT -> complex
            est_reverb = self.transforms.stft(est_rev_aligned, output_type="complex").to(dtype=torch.complex64)
            Td = self.transforms.stft(rev_targets, output_type="complex").to(dtype=torch.complex64)        # [B, n, F, frames]
            loss_rev = self.spec_loss(est_reverb, Td)
            loss_total = loss_total + self.w_rev * loss_rev

        # # recon 频谱 RIMag_loss（recon vs reordered）
        # if self.w_recon != 0.0:
        #     rir_target_per = rir_target.permute(0,2,1).contiguous()
        #     rir_target_stft = self.transforms.stft(rir_target_per, output_type="complex").to(dtype=torch.complex64)
        #     ests_stft = self.transforms.stft(reordered, output_type="complex").to(dtype=torch.complex64)
        #     recon = torchaudio.functional.convolve(ests_stft, rir_target_stft, mode="full")
        #     recon = recon[..., : ests_stft.shape[-1]]  # crop to T
        #     Td = self.transforms.stft(rev_targets, output_type="complex").to(dtype=torch.complex64)
        #     loss_recon = self.spec_loss(recon, Td)
        #     loss_total = loss_total + self.w_recon * loss_recon

        if self.w_recon != 0.0:
            clean_target_stft = self.transforms.stft(targets, output_type="complex").to(dtype=torch.complex64)
            rir = est_rir[:, 0, ...].unsqueeze(1) + 1j * est_rir[:, 1, ...].unsqueeze(1)
            est_rir_aligned = rir.permute(1,0,2,3).contiguous()
            est_rir_aligned = self.reordered_sources(est_rir_aligned, batch_indices).contiguous()
            recon = torchaudio.functional.convolve(clean_target_stft, est_rir_aligned, mode="full")
            recon = recon[..., : clean_target_stft.shape[-1]]
            Td = self.transforms.stft(rev_targets, output_type="complex").to(dtype=torch.complex64)
            loss_recon_clean = self.spec_loss(recon, Td)
            loss_total = loss_total + self.w_recon * loss_recon_clean

        # # rir RIMag_loss
        # if self.w_rir != 0.0:
        #     rir = est_rir[:, 0, ...].unsqueeze(1) + 1j * est_rir[:, 1, ...].unsqueeze(1)
        #     est_rir_aligned = rir.permute(1,0,2,3).contiguous()
        #     est_rir_aligned = self.reordered_sources(est_rir_aligned, batch_indices).contiguous()
        #     min_len = min(est_rir_aligned.shape[-1], rir_target_stft.shape[-1])
        #     loss_rir = self.spec_loss(est_rir_aligned[...,:min_len], rir_target_stft[...,:min_len])
        #     loss_total = loss_total + self.w_rir * loss_rir
        # print(mean_loss, loss_rev, loss_recon, loss_rir)
        if not return_ests:
            return loss_total

        return loss_total, reordered

    def get_pw_losses(self, loss_func, ests, targets, **kwargs):
        B, n_src, _ = targets.shape
        pair_wise_losses = targets.new_empty(B, n_src, n_src)
        for est_idx, est_src in enumerate(ests.transpose(0, 1)):
            for target_idx, target_src in enumerate(targets.transpose(0, 1)):
                pair_wise_losses[:, est_idx, target_idx] = loss_func(
                    est_src, target_src, **kwargs
                )
        return pair_wise_losses

    def best_perm_from_perm_avg_loss(self, loss_func, ests, targets, **kwargs):
        n_src = targets.shape[1]
        perms = torch.tensor(list(permutations(range(n_src))), dtype=torch.long)
        # import pdb; pdb.set_trace()
        loss_set = torch.stack(
            [loss_func(ests[:, perm], targets) for perm in perms], dim=1
        )
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)
        batch_indices = torch.stack([perms[m] for m in min_loss_idx], dim=0)
        return min_loss, batch_indices

    def reordered_sources(self, sources, batch_indices):
        reordered_sources = torch.stack(
            [torch.index_select(s, 0, b) for s, b in zip(sources, batch_indices)]
        )
        return reordered_sources

    def find_best_perm(self, pair_wise_losses, perm_reduce=None, **kwargs):
        n_src = pair_wise_losses.shape[-1]
        if perm_reduce is not None or n_src <= 3:
            min_loss, batch_indices = self.find_best_perm_factorial(
                pair_wise_losses, perm_reduce=perm_reduce, **kwargs
            )
        else:
            min_loss, batch_indices = self.find_best_perm_hungarian(pair_wise_losses)
        return min_loss, batch_indices

    def find_best_perm_factorial(self, pair_wise_losses, perm_reduce=None, **kwargs):
        n_src = pair_wise_losses.shape[-1]
        # After transposition, dim 1 corresp. to sources and dim 2 to estimates
        pwl = pair_wise_losses.transpose(-1, -2)
        perms = pwl.new_tensor(list(permutations(range(n_src))), dtype=torch.long)
        # Column permutation indices
        idx = torch.unsqueeze(perms, 2)
        # Loss mean of each permutation
        if perm_reduce is None:
            # one-hot, [n_src!, n_src, n_src]
            # import pdb; pdb.set_trace()
            perms_one_hot = pwl.new_zeros((*perms.size(), n_src)).scatter_(2, idx, 1)
            loss_set = torch.einsum("bij,pij->bp", [pwl, perms_one_hot])
            loss_set /= n_src
        else:
            # batch = pwl.shape[0]; n_perm = idx.shape[0]
            # [batch, n_src!, n_src] : Pairwise losses for each permutation.
            pwl_set = pwl[:, torch.arange(n_src), idx.squeeze(-1)]
            # Apply reduce [batch, n_src!, n_src] --> [batch, n_src!]
            loss_set = perm_reduce(pwl_set, **kwargs)
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)

        # Permutation indices for each batch.
        batch_indices = torch.stack([perms[m] for m in min_loss_idx], dim=0)
        return min_loss, batch_indices

    def find_best_perm_hungarian(self, pair_wise_losses: torch.Tensor):
        pwl = pair_wise_losses.transpose(-1, -2)
        # Just bring the numbers to cpu(), not the graph
        pwl_copy = pwl.detach().cpu()
        # Loop over batch + row indices are always ordered for square matrices.
        batch_indices = torch.tensor(
            [linear_sum_assignment(pwl)[1] for pwl in pwl_copy]
        ).to(pwl.device)
        min_loss = torch.gather(pwl, 2, batch_indices[..., None]).mean([-1, -2])
        return min_loss, batch_indices


if __name__ == "__main__":
    import torch
    from matrix import pairwise_neg_sisdr, pairwise_neg_sisdr

    ests = torch.randn(10, 2, 32000)
    targets = torch.randn(10, 2, 32000)

    pit_wrapper_1 = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    pit_wrapper_2 = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    print(pit_wrapper_1(ests, targets))
    print(pit_wrapper_2(ests, targets))
