"""Setup script for magnetics inversion system library."""

from setuptools import setup, find_packages

setup(
    name="mag-inversion",
    version="0.1.0",
    description=(
        "Airborne magnetics equivalent source and 3D inversion systems "
        "built on SimPEG. Standalone library; can be used independently "
        "of Nagelfluh."
    ),
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "discretize",
        "simpeg",
    ],
    entry_points={
        "ymerflow.mag_equiv_source_systems": [
            "Equivalent Source=mag_inversion.equivalent_source:MagEquivalentSourceSystem",
        ],
        "ymerflow.mag_inversion_3d_systems": [
            "3D Inversion=mag_inversion.full_3d:MagInversion3DSystem",
        ],
    },
)
