import warnings

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None


class MLPTcnn(nn.Module):
    """tiny-cuda-nn backed MLP decoder with INR-compatible interface."""

    def __init__(self, input_dim, hidden_dim, depth=4, output_dim=3, dtype="fp16"):
        super().__init__()
        if tcnn is None:
            raise ImportError(
                "tinycudann is required for MLPTcnn. Install with: "
                "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
            )
        if depth < 2:
            raise ValueError("MLPTcnn requires depth >= 2.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.output_dim = int(output_dim)
        self.dtype = str(dtype).lower()
        if self.dtype not in {"fp16", "fp32"}:
            raise ValueError("MLPTcnn dtype must be 'fp16' or 'fp32'.")

        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": self.hidden_dim,
            "n_hidden_layers": self.depth - 1,
        }
        try:
            self._network = tcnn.Network(self.input_dim, self.output_dim, network_config)
        except Exception:
            network_config["otype"] = "CutlassMLP"
            self._network = tcnn.Network(self.input_dim, self.output_dim, network_config)

        native_fp16 = self._tcnn_native_fp16()
        if self.dtype == "fp32" and native_fp16:
            warnings.warn(
                "MLPTcnn requested fp32, but this tinycudann build computes in fp16. "
                "Reinstall with: TCNN_HALF_PRECISION=0 pip install --no-build-isolation "
                "\"git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch\"",
                UserWarning,
                stacklevel=2,
            )

    @staticmethod
    def _tcnn_native_fp16() -> bool:
        """True when the installed tinycudann kernels use fp16 internally."""
        if not torch.cuda.is_available():
            return True
        try:
            probe = tcnn.Network(
                4,
                1,
                {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 16,
                    "n_hidden_layers": 1,
                },
            )
            y = probe(torch.zeros(1, 4, device="cuda"))
            return y.dtype == torch.float16
        except Exception:
            return True

    def forward(self, x):
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_dim).contiguous()
        y = self._network(x)
        if self.dtype == "fp32" and y.dtype != torch.float32:
            y = y.float()
        return y.reshape(*orig_shape, self.output_dim)

