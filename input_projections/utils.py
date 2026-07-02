import torch.nn.functional as F

from input_projections.fourier_projection import FourierProjection
from input_projections.hashgrid_projection import HashGridProjection


def _make_hashgrid_kwargs(
    input_dim,
    hash_n_levels,
    hash_n_features_per_level,
    hash_log2_hashmap_size,
    hash_base_resolution,
    hash_max_resolution,
    device,
    hash_interpolation="smoothstep",
    hash_level_sigma=0.0,
):
    return dict(
        input_dim=input_dim,
        n_levels=hash_n_levels,
        n_features_per_level=hash_n_features_per_level,
        log2_hashmap_size=hash_log2_hashmap_size,
        base_resolution=hash_base_resolution,
        max_resolution=hash_max_resolution,
        interpolation=hash_interpolation,
        level_sigma=hash_level_sigma,
        device=device,
    )


def get_input_projection(
    input_projection_name,
    input_dim,
    output_dim,
    device,
    fourier_scale=10.0,
    legendre_max_degree=10,
    activation=F.relu,
    hash_n_levels=16,
    hash_n_features_per_level=2,
    hash_log2_hashmap_size=19,
    hash_base_resolution=16,
    hash_max_resolution=2048,
    hash_tcnn_output_dtype="fp32",
    hash_interpolation="smoothstep",
    hash_level_sigma=0.0,
):
    if input_projection_name is None:
        return None
    name = str(input_projection_name).lower()
    if name == "none":
        return None
    if name == "fourier":
        return FourierProjection(input_dim=input_dim, output_dim=output_dim, scale=fourier_scale, device=device)
    if name in {"hashgrid", "hash", "ngp_hash"}:
        return HashGridProjection(
            **_make_hashgrid_kwargs(
                input_dim,
                hash_n_levels,
                hash_n_features_per_level,
                hash_log2_hashmap_size,
                hash_base_resolution,
                hash_max_resolution,
                device,
                hash_interpolation,
                hash_level_sigma,
            )
        )
    if name in {"hashgrid_tcnn", "hash_tcnn", "ngp_hash_tcnn"}:
        from input_projections import hashgrid_tcnn as _tcnn_mod

        if _tcnn_mod.tcnn is None:
            import warnings

            warnings.warn(
                "tinycudann not found; using PyTorch hashgrid instead. "
                "Install with: pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch",
                UserWarning,
                stacklevel=2,
            )
            return HashGridProjection(
                **_make_hashgrid_kwargs(
                    input_dim,
                    hash_n_levels,
                    hash_n_features_per_level,
                    hash_log2_hashmap_size,
                    hash_base_resolution,
                    hash_max_resolution,
                    device,
                    hash_interpolation,
                    hash_level_sigma,
                )
            )

        from input_projections.hashgrid_tcnn import HashGridTcnn

        tcnn_kwargs = _make_hashgrid_kwargs(
            input_dim,
            hash_n_levels,
            hash_n_features_per_level,
            hash_log2_hashmap_size,
            hash_base_resolution,
            hash_max_resolution,
            device,
            hash_interpolation,
            hash_level_sigma,
        )
        tcnn_kwargs["output_dtype"] = hash_tcnn_output_dtype
        return HashGridTcnn(**tcnn_kwargs)

    raise ValueError(
        f"Unknown projection: {input_projection_name}. Use 'fourier', 'hashgrid', or 'hashgrid_tcnn'."
    )
