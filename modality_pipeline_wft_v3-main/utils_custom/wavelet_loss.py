import math
import torch
import torch.nn as nn
import ptwt
import pywt

HF_KEYS = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")

# modes where zero/constant padding fabricates a hard edge that long filters smear inward
_SHORT_FILTER_MODES = {"zero", "constant"}


class WaveletL1Loss(nn.Module):
    """Multi-level 3D DWT L1 loss, high-frequency bands weighted heavier.

    Works with any discrete wavelet supported by PyWavelets. Per-level
    weights are derived from the wavelet's own filter bank so the effective
    weighting is comparable across families.
    """

    def __init__(self, levels=2, w_lf=1.0, w_hf=4.0, wavelet="db4",
                 mode="auto", level_decay=2.0, normalize_scale=True):
        super().__init__()

        self.wavelet = self._resolve_wavelet(wavelet)
        self.levels = int(levels)
        self.w_lf = float(w_lf)
        self.w_hf = float(w_hf)
        self.dec_len = self.wavelet.dec_len

        if self.levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")

        # long filters + zero padding = manufactured boundary edge
        if mode == "auto":
            mode = "zero" if self.dec_len <= 2 else "reflect"
        elif mode in _SHORT_FILTER_MODES and self.dec_len > 2:
            import warnings
            warnings.warn(
                f"mode={mode!r} with {self.wavelet.name} (dec_len={self.dec_len}) "
                f"creates boundary artifacts; prefer 'reflect' or 'periodization'.",
                stacklevel=2,
            )
        self.mode = mode

        # A smooth (locally constant) signal gets multiplied by sum(dec_lo) per
        # 1D pass -> cubed for separable 3D. That's sqrt(2)**3 = sqrt(8) for any
        # orthonormal family, but biorthogonal banks differ, so read it off the
        # actual filters rather than assuming.
        if normalize_scale:
            lf_gain = abs(sum(self.wavelet.dec_lo)) ** 3
            lf_gain = max(lf_gain, 1e-8)
        else:
            lf_gain = 1.0
        self.lf_gain = lf_gain

        decay = level_decay * lf_gain
        self.level_w = [1.0 / decay ** j for j in range(self.levels)]

        # authoritative minimum size, per PyWavelets' own rule
        self.min_size = self._min_spatial_size()

    @staticmethod
    def _resolve_wavelet(wavelet):
        """Accept a name, a Wavelet object, or a raw filter bank."""
        if isinstance(wavelet, pywt.Wavelet):
            return wavelet
        if isinstance(wavelet, (list, tuple)) and len(wavelet) == 4:
            return pywt.Wavelet("custom", filter_bank=wavelet)
        if isinstance(wavelet, str):
            try:
                return pywt.Wavelet(wavelet)
            except ValueError as e:
                if wavelet in pywt.wavelist(kind="continuous"):
                    raise ValueError(
                        f"{wavelet!r} is a continuous wavelet and has no DWT "
                        f"filter bank. Use a discrete family: "
                        f"{pywt.families(short=True)}"
                    ) from e
                raise ValueError(
                    f"Unknown wavelet {wavelet!r}. Available: "
                    f"{pywt.wavelist(kind='discrete')}"
                ) from e
        raise TypeError(f"Cannot interpret {type(wavelet)} as a wavelet")

    def _min_spatial_size(self):
        """Smallest n with pywt.dwt_max_level(n, dec_len) >= levels."""
        n = 2
        while pywt.dwt_max_level(n, self.dec_len) < self.levels:
            n *= 2
            if n > 1 << 20:
                raise ValueError(
                    f"levels={self.levels} is unreachable with "
                    f"{self.wavelet.name} (dec_len={self.dec_len})"
                )
        return n

    def extra_repr(self):
        return (f"wavelet={self.wavelet.name}, dec_len={self.dec_len}, "
                f"levels={self.levels}, mode={self.mode}, "
                f"w_lf={self.w_lf}, w_hf={self.w_hf}, "
                f"lf_gain={self.lf_gain:.3f}, min_size={self.min_size}")

    def max_levels(self, shape):
        """How many levels this wavelet supports for a given (D, H, W)."""
        return pywt.dwt_max_level(min(shape[-3:]), self.dec_len)

    def forward(self, pred, target):
        # (B, C, D, H, W)
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        if pred.ndim != 5:
            raise ValueError(f"expected 5D (B,C,D,H,W), got {pred.ndim}D")

        B, C, D, H, W = pred.shape
        if min(D, H, W) < self.min_size:
            raise ValueError(
                f"volume {(D, H, W)} too small for {self.levels} levels of "
                f"{self.wavelet.name}; need min spatial dim >= {self.min_size} "
                f"(this volume supports {self.max_levels(pred.shape)} levels)"
            )

        N = B * C
        with torch.autocast(device_type=pred.device.type, enabled=False):
            # one transform over both, then split — halves the conv work
            a = torch.cat([pred.float(), target.float()], dim=0).reshape(2 * N, D, H, W)
            loss = torch.zeros((), device=pred.device, dtype=torch.float32)

            for j in range(self.levels):
                a, details = ptwt.wavedec3(a, self.wavelet, level=1, mode=self.mode)

                lf = (a[:N] - a[N:]).abs().mean()

                hf_bands = torch.stack([details[k] for k in HF_KEYS], dim=0)
                # mean over stacked bands * 7 == sum of per-band means
                hf = (hf_bands[:, :N] - hf_bands[:, N:]).abs().mean() * len(HF_KEYS)

                loss = loss + self.level_w[j] * (self.w_lf * lf + self.w_hf * hf)

        return loss



# import torch
# import torch.nn as nn
# import ptwt
# import pywt

# HF_KEYS = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")


# class WaveletL1Loss(nn.Module):
#     """Multi-level 3D Haar DWT L1 loss, high-frequency bands weighted heavier."""

#     def __init__(self, levels=2, w_lf=1.0, w_hf=4.0, wavelet="haar", mode="zero"):
#         super().__init__()
#         self.levels = levels
#         self.w_lf = w_lf
#         self.w_hf = w_hf
#         self.wavelet = pywt.Wavelet(wavelet)
#         self.mode = mode
#         # precompute level weights once
#         self.register_buffer(
#             "level_w",
#             torch.tensor([1.0 / (2 ** (j - 1)) for j in range(1, levels + 1)]),
#         )
#         print("Wavelet Loss Initialied with: ",
#             "levels ", self.levels,
#             "w_lf ", self.w_lf,
#             "w_hf ", self.w_hf, 
#             "wavelet ", self.wavelet,
#             "mode ", self.mode
#               )

#     def forward(self, pred, target):
#         # pred, target: (B, C, D, H, W)
#         B = pred.shape[0]
#         with torch.autocast(device_type=pred.device.type, enabled=False):
#             x = torch.cat([pred, target], dim=0)      # one transform, not two

#             loss = pred.new_zeros(())
#             a = x

#             for j in range(self.levels):
#                 a, details = ptwt.wavedec3(a, self.wavelet, level=1, mode=self.mode)

#                 # LF: single abs + reduction over the pred/target difference
#                 ap, at = a[:B], a[B:]
#                 lf = (ap - at).abs().mean()

#                 # HF: stack the 7 bands -> one abs + one reduction, not seven
#                 hf_bands = torch.stack([details[k] for k in HF_KEYS], dim=0)
#                 hp, ht = hf_bands[:, :B], hf_bands[:, B:]
#                 hf = (hp - ht).abs().mean() * len(HF_KEYS)   # keep sum-over-bands semantics

#                 loss = loss + self.level_w[j] * (self.w_lf * lf + self.w_hf * hf)
#         return loss

HF_KEYS_2D = ("lh", "hl", "hh")


class WaveletL1Loss2D(nn.Module):
    """Multi-level 2D DWT L1 loss, high-frequency bands weighted heavier.

    The slice-pipeline counterpart of `WaveletL1Loss`. Same knobs, same
    normalisation logic, but the low-frequency gain is squared (two separable
    1D passes) instead of cubed.
    """

    def __init__(self, levels=2, w_lf=1.0, w_hf=4.0, wavelet="db4",
                 mode="auto", level_decay=2.0, normalize_scale=True):
        super().__init__()

        self.wavelet = WaveletL1Loss._resolve_wavelet(wavelet)
        self.levels = int(levels)
        self.w_lf = float(w_lf)
        self.w_hf = float(w_hf)
        self.dec_len = self.wavelet.dec_len

        if self.levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")

        # long filters + zero padding = manufactured boundary edge
        if mode == "auto":
            mode = "zero" if self.dec_len <= 2 else "reflect"
        elif mode in _SHORT_FILTER_MODES and self.dec_len > 2:
            import warnings
            warnings.warn(
                f"mode={mode!r} with {self.wavelet.name} (dec_len={self.dec_len}) "
                f"creates boundary artifacts; prefer 'reflect' or 'periodization'.",
                stacklevel=2,
            )
        self.mode = mode

        if normalize_scale:
            lf_gain = max(abs(sum(self.wavelet.dec_lo)) ** 2, 1e-8)
        else:
            lf_gain = 1.0
        self.lf_gain = lf_gain

        decay = level_decay * lf_gain
        self.level_w = [1.0 / decay ** j for j in range(self.levels)]

        self.min_size = WaveletL1Loss._min_spatial_size(self)

    def extra_repr(self):
        return (f"wavelet={self.wavelet.name}, dec_len={self.dec_len}, "
                f"levels={self.levels}, mode={self.mode}, "
                f"w_lf={self.w_lf}, w_hf={self.w_hf}, "
                f"lf_gain={self.lf_gain:.3f}, min_size={self.min_size}")

    def max_levels(self, shape):
        """How many levels this wavelet supports for a given (H, W)."""
        return pywt.dwt_max_level(min(shape[-2:]), self.dec_len)

    def forward(self, pred, target):
        # (B, C, H, W)
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        if pred.ndim != 4:
            raise ValueError(f"expected 4D (B,C,H,W), got {pred.ndim}D")

        B, C, H, W = pred.shape
        if min(H, W) < self.min_size:
            raise ValueError(
                f"slice {(H, W)} too small for {self.levels} levels of "
                f"{self.wavelet.name}; need min spatial dim >= {self.min_size} "
                f"(this slice supports {self.max_levels(pred.shape)} levels)"
            )

        N = B * C
        with torch.autocast(device_type=pred.device.type, enabled=False):
            # one transform over both, then split -- halves the conv work
            a = torch.cat([pred.float(), target.float()], dim=0).reshape(2 * N, H, W)
            loss = torch.zeros((), device=pred.device, dtype=torch.float32)

            for j in range(self.levels):
                a, details = ptwt.wavedec2(a, self.wavelet, level=1, mode=self.mode)

                lf = (a[:N] - a[N:]).abs().mean()

                hf_bands = torch.stack(list(details), dim=0)
                # mean over stacked bands * 3 == sum of per-band means
                hf = (hf_bands[:, :N] - hf_bands[:, N:]).abs().mean() * len(HF_KEYS_2D)

                loss = loss + self.level_w[j] * (self.w_lf * lf + self.w_hf * hf)

        return loss
