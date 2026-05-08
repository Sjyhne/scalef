import warnings

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None


class MLPTcnn(nn.Module):
    """tiny-cuda-nn backed MLP decoder with INR-compatible interface."""

    def __init__(self, input_dim, hidden_dim, depth=4, output_dim=3, dtype="fp16", device=None):
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

        target_device = torch.device(device) if device is not None else None
        use_cuda_device = (
            target_device is not None
            and target_device.type == "cuda"
            and torch.cuda.is_available()
        )
        prev_cuda_device = None
        if use_cuda_device:
            prev_cuda_device = torch.cuda.current_device()
            torch.cuda.set_device(target_device)

        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": self.hidden_dim,
            "n_hidden_layers": self.depth - 1,
        }
        try:
            try:
                self._network = tcnn.Network(self.input_dim, self.output_dim, network_config)
            except Exception:
                network_config["otype"] = "CutlassMLP"
                self._network = tcnn.Network(self.input_dim, self.output_dim, network_config)
        finally:
            if use_cuda_device and prev_cuda_device is not None and prev_cuda_device != target_device.index:
                try:
                    torch.cuda.set_device(prev_cuda_device)
                except RuntimeError:
                    # Keep chosen target device if default/current device is unavailable.
                    pass

        if self.dtype == "fp32":
            warnings.warn(
                "MLPTcnn requested fp32, but this tinycudann build computes in fp16. "
                "To get true fp32 kernels, reinstall tinycudann with TCNN_HALF_PRECISION=0.",
                UserWarning,
                stacklevel=2,
            )

        if target_device is not None:
            self.to(target_device)

    def forward(self, x):
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_dim).contiguous()
        y = self._network(x)
        if self.dtype == "fp32" and y.dtype != torch.float32:
            y = y.float()
        return y.reshape(*orig_shape, self.output_dim)

