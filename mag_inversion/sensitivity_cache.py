"""Hash-based sensitivity matrix caching utilities.

The sensitivity matrix G (shape nD x nC) is expensive to compute but
depends only on geometry and physics parameters, not on regularization.
We cache it on disk keyed by a hash of the inputs that affect G.

Usage in a system class::

    sens_path = os.path.join(base_path, sensitivity_hash(...))
    os.makedirs(sens_path, exist_ok=True)
    sim = Simulation3DIntegral(..., sensitivity_path=sens_path + os.sep)
    # SimPEG checks for sens_path/sensitivity.npy and reuses if shape matches.

For blob storage integration (Nagelfluh process layer):

    remote_key = f"sensitivity_cache/{h}/sensitivity.npy"
    # Download before inversion, upload after.
"""

import hashlib
import json

import numpy as np


def sensitivity_hash(
    receiver_locations: np.ndarray,
    field_params: list,
    mesh_params: dict,
    model_type: str = "scalar",
) -> str:
    """Compute a 16-character hex hash of all G-affecting parameters.

    Parameters
    ----------
    receiver_locations:
        Nx3 float array of (x, y, z) observation positions.
    field_params:
        [intensity_nT, inclination_deg, declination_deg]
    mesh_params:
        Dict of mesh construction parameters (cell sizes, depths, etc.).
        Must be JSON-serialisable. Key names are arbitrary — include
        everything that changes the mesh geometry.
    model_type:
        "scalar" or "vector".  Affects the column count of G.

    Returns
    -------
    str
        16-character hex digest (truncated SHA-256).
    """
    loc_hash = hashlib.sha256(
        np.asarray(receiver_locations, dtype=np.float64).tobytes()
    ).hexdigest()

    payload = {
        "receiver_loc_hash": loc_hash,
        "field_params": [round(float(v), 6) for v in field_params],
        "mesh_params": mesh_params,
        "model_type": model_type,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
