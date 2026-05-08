import torch.nn.functional as F

from input_projections.fourier_projection import FourierProjection


def normalize_input_projection_name(input_projection_name, fourier_scale):
    if input_projection_name is None:
        return None, fourier_scale

    name = str(input_projection_name).lower()
    if name.startswith("fourier_"):
        suffix = name.split("_", 1)[1]
        try:
            return "fourier", float(suffix)
        except ValueError:
            pass
    if name == "fourier":
        return "fourier", fourier_scale
    if name in {"hashgrid", "hash", "ngp_hash", "hashgrid_tcnn", "hash_tcnn", "ngp_hash_tcnn"}:
        return "hashgrid_tcnn", fourier_scale
    if name == "none":
        return "none", fourier_scale
    raise ValueError(
        f"Unknown projection: {input_projection_name}. "
        "Use 'fourier' (e.g. fourier_10), 'hashgrid_tcnn' (or alias hashgrid / hash), or 'none'."
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
    hash_encoding_output_dtype="fp32",
    coord_margin: float = 0.0,
    hash_encoding_preset: str = "hashgrid",
    hash_grid_type: str = "Hash",
):
    name, fourier_scale = normalize_input_projection_name(input_projection_name, fourier_scale)
    if name is None:
        return None
    if name == "none":
        return None
    if name == "fourier":
        return FourierProjection(
            input_dim=input_dim, output_dim=output_dim, scale=fourier_scale, device=device
        )
    if name == "hashgrid_tcnn":
        from input_projections.hashgrid_tcnn import HashGridTcnn

        _ = coord_margin  # Deprecated/ignored: hashgrid is fixed to the base-frame [0,1] domain.
        return HashGridTcnn(
            input_dim=input_dim,
            n_levels=hash_n_levels,
            n_features_per_level=hash_n_features_per_level,
            log2_hashmap_size=hash_log2_hashmap_size,
            base_resolution=hash_base_resolution,
            max_resolution=hash_max_resolution,
            output_dtype=hash_encoding_output_dtype,
            device=device,
            encoding_preset=str(hash_encoding_preset),
            grid_type=str(hash_grid_type),
        )
    raise ValueError(
        f"Unknown projection: {input_projection_name}. Use 'fourier', 'hashgrid_tcnn', or 'none'."
    )
