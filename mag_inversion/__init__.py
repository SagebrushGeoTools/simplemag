"""Airborne magnetics inversion systems.

Two systems are provided:

    MagEquivalentSourceSystem
        Inverts flight-line TMI data for effective dipoles on a flat
        horizontal layer (TensorMesh).  Produces gridded Bx/By/Bz/TMI
        fields at a uniform altitude (xarray Dataset, webxtile-compatible).

    MagInversion3DSystem
        Inverts gridded Bx/By/Bz or TMI (from equivalent source) for a
        3D susceptibility (scalar) or magnetization vector (MVI) model
        using an OcTree (TreeMesh).  Produces a 3D xarray Dataset.

Both classes follow the same attribute-override pattern as the AEM
XYZSystem — every tunable parameter is a class-level attribute that can
be overridden at instantiation:

    system = MagEquivalentSourceSystem(mag_data, layer__cell_size=100)
"""

from .equivalent_source import MagEquivalentSourceSystem
from .full_3d import MagInversion3DSystem

__all__ = ["MagEquivalentSourceSystem", "MagInversion3DSystem"]
