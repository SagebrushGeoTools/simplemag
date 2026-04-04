"""Full 3D magnetic inversion using an OcTree (TreeMesh).

Takes gridded Bx/By/Bz or TMI fields (typically produced by
MagEquivalentSourceSystem) and inverts for a 3D susceptibility (scalar)
or magnetization vector (MVI) model.

Basic usage::

    import xarray as xr
    from mag_inversion import MagInversion3DSystem

    ds = xr.open_dataset("equiv_out/", engine="webxtile")
    system = MagInversion3DSystem(ds)
    ds_3d = system.run()
    ds_3d.webxtile.to_webxtile("mag3d_out/")

All class-level attributes can be overridden at instantiation::

    system = MagInversion3DSystem(
        ds,
        model_type="amplitude",
        mesh__core_cell_size=100,
        directives__sensitivity_weights__enable=True,
    )
"""

import os
import tempfile
import typing

import numpy as np
import xarray as xr
from scipy.interpolate import NearestNDInterpolator

from discretize import TreeMesh
from discretize.utils import mesh_builder_xyz, refine_tree_xyz, active_from_xyz
from SimPEG import (
    data,
    data_misfit,
    directives,
    inversion,
    inverse_problem,
    maps,
    optimization,
    regularization,
)
from SimPEG.potential_fields import magnetics

from ._crs import _crs_to_wkt, _crs_variable_attrs
from .directives import ReportingDirective
from .sensitivity_cache import sensitivity_hash


class MagInversion3DSystem:
    """Full 3D magnetic inversion system (OcTree mesh).

    Inverts gridded magnetic field data — produced by
    :class:`~mag_inversion.MagEquivalentSourceSystem` or any other source
    that provides an ``xr.Dataset`` with dimensions ``(y, x)`` and an
    ``output_altitude_m`` attribute — for a 3D susceptibility or
    magnetisation vector model.

    Three inversion modes are supported via ``model_type``:

    ``"scalar"``
        Susceptibility χ (SI) per cell assuming induced magnetisation.
        Works with any single component (typically ``"tmi"``).

    ``"vector"``
        Three-component magnetisation per cell (MVI).  Input must contain
        ``"bx"``, ``"by"``, ``"bz"`` components.  Handles remanence.

    ``"amplitude"``
        Scalar susceptibility model, but data are converted to |B|
        amplitude before inversion.  Requires ``"bx"``, ``"by"``,
        ``"bz"`` in input.  Remanence-insensitive without vector model.

    Parameters
    ----------
    grid_data:
        xarray Dataset produced by the equivalent source step (or any
        dataset with ``(y, x)`` dimensions, an ``output_altitude_m`` attr
        and the expected field variables).
    **kw:
        Any class-level attribute can be overridden here.
    """

    def __init__(self, grid_data: xr.Dataset, **kw):
        self._grid_data = grid_data
        self.options = kw

    def __getattribute__(self, name):
        options = object.__getattribute__(self, "options")
        if name in options:
            return options[name]
        return object.__getattribute__(self, name)

    # ── Input field selection ─────────────────────────────────────────────────
    input__components: typing.Optional[list] = None
    """Components to use from grid_data.

    If None, auto-detected based on ``model_type`` and available
    variables.  Override to use a specific subset, e.g. ``["tmi"]``.
    """

    # ── Earth field ───────────────────────────────────────────────────────────
    field_intensity: typing.Optional[float] = None
    "Earth field intensity (nT). If None, read from grid_data.attrs."
    field_inclination: typing.Optional[float] = None
    "Earth field inclination (deg). If None, read from grid_data.attrs."
    field_declination: typing.Optional[float] = None
    "Earth field declination (deg). If None, read from grid_data.attrs."

    # ── Model type ────────────────────────────────────────────────────────────
    model_type: typing.Literal["scalar", "vector", "amplitude"] = "scalar"
    """Inversion mode: ``"scalar"``, ``"vector"`` (MVI), or ``"amplitude"``."""

    # ── OcTree mesh ───────────────────────────────────────────────────────────
    mesh__core_cell_size: float = 50.0
    "Core OcTree cell size (m).  Controls finest resolution."
    mesh__octree_levels_obs: list = [2, 4]
    "OcTree refinement levels around observation points (fine → coarse)."
    mesh__octree_levels_surf: list = [2, 4]
    "OcTree refinement levels near surface / topography."
    mesh__depth_core: float = 500.0
    "Depth of the core modelling region below observation level (m)."
    mesh__max_distance: float = 5000.0
    "Horizontal padding extent beyond data bounds (m)."

    # ── Topography ────────────────────────────────────────────────────────────
    topography: typing.Optional[np.ndarray] = None
    """Optional topography surface as Nx3 array ``(x, y, z)`` or an
    ``xr.DataArray`` with ``(y, x)`` dimensions.

    If None, a flat surface at the observation altitude is assumed
    (acceptable when the equivalent source step has already removed
    terrain effects).
    """

    # ── Starting / reference model ────────────────────────────────────────────
    startmodel__susceptibility: float = 1e-4
    "Starting susceptibility value (SI) applied to all active cells."
    startmodel__reference: float = 0.0
    "Reference model susceptibility (SI) used in the regularisation."

    # ── Uncertainties ─────────────────────────────────────────────────────────
    uncertainties__std_data: float = 0.02
    "Fractional uncertainty: std = std_data × |data| + floor_nT."
    uncertainties__floor_nT: float = 1.0
    "Absolute noise floor (nT)."

    # ── Regularization ────────────────────────────────────────────────────────
    regularization__alpha_s: float = 1e-4
    "Smallness weight."
    regularization__alpha_x: float = 1.0
    "X-direction smoothness weight."
    regularization__alpha_y: float = 1.0
    "Y-direction smoothness weight."
    regularization__alpha_z: float = 1.0
    "Z-direction smoothness weight."

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer__max_iter: int = 40
    "Maximum Gauss-Newton iterations."
    optimizer__max_iter_cg: int = 20
    "Maximum CG iterations per GN step."

    # ── Beta schedule ─────────────────────────────────────────────────────────
    directives__beta__seed: typing.Optional[int] = None
    "Random seed for beta estimator."
    directives__beta__beta0_ratio: float = 10.0
    "Ratio of initial beta to eigenvalue estimate."
    directives__beta__cooling_factor: float = 2.0
    "Factor by which beta is divided at each cooling step."
    directives__beta__cooling_rate: int = 1
    "GN iterations between cooling steps."

    # ── Sensitivity weights ────────────────────────────────────────────────────
    directives__sensitivity_weights__enable: bool = True
    """Apply depth-dependent sensitivity weighting.

    Corrects the natural decay of resolution with depth (sensitivity
    falls off as ~1/r^3 for magnetics).  Almost always beneficial for
    3D inversions.
    """

    # ── IRLS ──────────────────────────────────────────────────────────────────
    directives__irls__enable: bool = False
    "Run IRLS after L2 convergence to produce a compact/sparse model."
    directives__irls__max_iterations: int = 20
    directives__irls__minGNiter: int = 1
    directives__irls__fix_Jmatrix: bool = True
    directives__irls__f_min_change: float = 1e-3
    directives__irls__coolingRate: int = 1

    # ── Sensitivity matrix ─────────────────────────────────────────────────────
    store_sensitivities: typing.Literal["disk", "ram", "forward_only"] = "disk"
    """How to store/access the sensitivity matrix G."""
    sensitivity_path: typing.Optional[str] = None
    """Root directory for sensitivity caching (hash sub-dir appended automatically)."""

    # ── Output grid ───────────────────────────────────────────────────────────
    output__xy_spacing: typing.Optional[float] = None
    "Horizontal spacing of regular output grid (m). If None, uses mesh__core_cell_size."
    output__z_spacing: typing.Optional[float] = None
    "Vertical spacing of regular output grid (m). If None, uses mesh__core_cell_size."

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def _field_params(self) -> list:
        attrs = self._grid_data.attrs
        intensity = (
            self.field_intensity
            if self.field_intensity is not None
            else float(attrs.get("field_intensity_nT", 50000.0))
        )
        inclination = (
            self.field_inclination
            if self.field_inclination is not None
            else float(attrs.get("field_inclination_deg", 70.0))
        )
        declination = (
            self.field_declination
            if self.field_declination is not None
            else float(attrs.get("field_declination_deg", 0.0))
        )
        return [intensity, inclination, declination]

    @property
    def _survey_components(self) -> list:
        """Components actually present in the dataset that we will use."""
        if self.input__components is not None:
            return list(self.input__components)
        available = set(self._grid_data.data_vars.keys())
        if self.model_type in ("vector", "amplitude"):
            needed = ["bx", "by", "bz"]
            if all(c in available for c in needed):
                return needed
        if "tmi" in available:
            return ["tmi"]
        raise ValueError(
            f"Cannot auto-detect components for model_type={self.model_type!r}. "
            f"Available variables: {sorted(available)}. "
            "Set input__components explicitly."
        )

    @property
    def _receiver_locations(self) -> np.ndarray:
        ds = self._grid_data
        alt = float(ds.attrs.get("output_altitude_m", 0.0))
        X, Y = np.meshgrid(ds.x.values, ds.y.values)
        return np.c_[X.ravel(), Y.ravel(), np.full(X.size, alt)]

    @property
    def _data_array(self) -> np.ndarray:
        ds = self._grid_data
        comps = self._survey_components
        if self.model_type == "amplitude":
            bx = ds["bx"].values.ravel().astype(float)
            by = ds["by"].values.ravel().astype(float)
            bz = ds["bz"].values.ravel().astype(float)
            return np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        return np.concatenate([ds[c].values.ravel().astype(float) for c in comps])

    @property
    def _uncertainty(self) -> np.ndarray:
        d = self._data_array
        return self.uncertainties__std_data * np.abs(d) + self.uncertainties__floor_nT

    def _resolved_sensitivity_path(self) -> str:
        base = self.sensitivity_path or os.path.join(
            tempfile.gettempdir(), "mag_3d_sensitivity"
        )
        h = sensitivity_hash(
            receiver_locations=self._receiver_locations,
            field_params=self._field_params,
            mesh_params={
                "core_cell_size": self.mesh__core_cell_size,
                "octree_levels_obs": self.mesh__octree_levels_obs,
                "octree_levels_surf": self.mesh__octree_levels_surf,
                "depth_core": self.mesh__depth_core,
                "max_distance": self.mesh__max_distance,
            },
            model_type=self.model_type,
        )
        path = os.path.join(base, h)
        os.makedirs(path, exist_ok=True)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # SimPEG component factories
    # ─────────────────────────────────────────────────────────────────────────

    def make_topography(self) -> np.ndarray:
        """Return Nx3 topography array (x, y, z).

        If ``topography`` is not set, uses a flat surface at the
        observation altitude (safe assumption after equivalent source
        gridding).
        """
        topo = self.topography
        if topo is None:
            locs = self._receiver_locations
            return locs.copy()
        if hasattr(topo, "values"):
            # xr.DataArray with (y, x) dims
            X, Y = np.meshgrid(topo.x.values, topo.y.values)
            return np.c_[X.ravel(), Y.ravel(), topo.values.ravel()]
        return np.asarray(topo, dtype=float)

    def make_mesh(self) -> TreeMesh:
        """Build the OcTree (TreeMesh) with adaptive refinement."""
        locs = self._receiver_locations
        topo = self.make_topography()
        cs = self.mesh__core_cell_size

        mesh = mesh_builder_xyz(
            locs,
            [cs, cs, cs],
            padding_distance=[
                [self.mesh__max_distance, self.mesh__max_distance],
                [self.mesh__max_distance, self.mesh__max_distance],
                [self.mesh__max_distance, self.mesh__depth_core],
            ],
            mesh_type="tree",
        )

        mesh = refine_tree_xyz(
            mesh,
            topo,
            method="surface",
            octree_levels=self.mesh__octree_levels_surf,
            finalize=False,
        )
        mesh = refine_tree_xyz(
            mesh,
            locs,
            method="radial",
            octree_levels=self.mesh__octree_levels_obs,
            finalize=True,
        )
        return mesh

    def make_active_cells(self, mesh: TreeMesh) -> np.ndarray:
        """Return boolean array of cells below (or at) topography."""
        topo = self.make_topography()
        return active_from_xyz(mesh, topo)

    def make_survey(self) -> magnetics.survey.Survey:
        """Build the survey at gridded observation locations."""
        locs = self._receiver_locations
        comps = self._survey_components

        # For amplitude inversion we predict all 3 components then compute |B|
        sim_components = comps if self.model_type != "amplitude" else ["bx", "by", "bz"]

        receiver = magnetics.receivers.Point(locs, components=sim_components)
        source = magnetics.sources.SourceField(
            receiver_list=[receiver], parameters=self._field_params
        )
        return magnetics.survey.Survey(source)

    def make_simulation(
        self,
        mesh: TreeMesh,
        survey: magnetics.survey.Survey,
        active_cells: np.ndarray,
    ) -> magnetics.simulation.Simulation3DIntegral:
        """Build the 3D integral forward simulation."""
        n_active = int(active_cells.sum())
        n_params = n_active if self.model_type != "vector" else 3 * n_active
        chi_map = maps.IdentityMap(nP=n_params)
        sens_path = self._resolved_sensitivity_path()

        is_amplitude = self.model_type == "amplitude"
        sim_model_type = "vector" if self.model_type == "vector" else "scalar"

        return magnetics.simulation.Simulation3DIntegral(
            mesh=mesh,
            survey=survey,
            chiMap=chi_map,
            actInd=active_cells,
            store_sensitivities=self.store_sensitivities,
            sensitivity_path=sens_path + os.sep,
            model_type=sim_model_type,
            is_amplitude_data=is_amplitude,
        )

    def make_data_object(self, survey: magnetics.survey.Survey) -> data.Data:
        d_obs = self._data_array
        d_std = self._uncertainty
        return data.Data(survey, dobs=d_obs, standard_deviation=d_std)

    def make_misfit(
        self,
        sim: magnetics.simulation.Simulation3DIntegral,
        data_obj: data.Data,
    ) -> data_misfit.L2DataMisfit:
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data_obj)
        dmis.W = 1.0 / self._uncertainty
        return dmis

    def make_startmodel(self, active_cells: np.ndarray) -> np.ndarray:
        n_active = int(active_cells.sum())
        n_params = n_active if self.model_type != "vector" else 3 * n_active
        return np.full(n_params, self.startmodel__susceptibility)

    def make_regularization(
        self, mesh: TreeMesh, active_cells: np.ndarray
    ) -> regularization.Tikhonov:
        n_active = int(active_cells.sum())
        n_params = n_active if self.model_type != "vector" else 3 * n_active

        mref = np.full(n_params, self.startmodel__reference)

        reg = regularization.Tikhonov(
            mesh,
            indActive=active_cells,
            mapping=maps.IdentityMap(nP=n_params),
            alpha_s=self.regularization__alpha_s,
            alpha_x=self.regularization__alpha_x,
            alpha_y=self.regularization__alpha_y,
            alpha_z=self.regularization__alpha_z,
            mref=mref,
        )
        return reg

    def make_optimizer(self) -> optimization.InexactGaussNewton:
        return optimization.InexactGaussNewton(
            maxIter=self.optimizer__max_iter,
            maxIterCG=self.optimizer__max_iter_cg,
        )

    def make_directives(self, sim, reg) -> list:
        """Build the list of inversion directives.

        Parameters
        ----------
        sim, reg:
            Passed through to directives that need a reference to the
            simulation or regularisation (e.g. UpdateSensitivityWeights).
        """
        if self.directives__beta__seed is not None:
            beta_est = directives.BetaEstimate_ByEig(
                beta0_ratio=self.directives__beta__beta0_ratio,
                seed=self.directives__beta__seed,
            )
        else:
            beta_est = directives.BetaEstimate_ByEig(
                beta0_ratio=self.directives__beta__beta0_ratio,
            )

        dirs = [
            beta_est,
            directives.BetaSchedule(
                coolingFactor=self.directives__beta__cooling_factor,
                coolingRate=self.directives__beta__cooling_rate,
            ),
            directives.TargetMisfit(),
            ReportingDirective(),
        ]

        if self.directives__sensitivity_weights__enable:
            dirs.append(directives.UpdateSensitivityWeights())

        if self.directives__irls__enable:
            dirs += [
                directives.Update_IRLS(
                    max_irls_iterations=self.directives__irls__max_iterations,
                    minGNiter=self.directives__irls__minGNiter,
                    fix_Jmatrix=self.directives__irls__fix_Jmatrix,
                    f_min_change=self.directives__irls__f_min_change,
                    coolingRate=self.directives__irls__coolingRate,
                ),
                directives.UpdatePreconditioner(),
            ]

        return dirs

    def make_inversion(self):
        """Assemble the full SimPEG Inversion.

        Exposed for advanced use (e.g. inspecting the mesh or active cells
        before running).  Normal callers should use :meth:`invert` instead.

        Returns
        -------
        inv : SimPEG.inversion.BaseInversion
        mesh : TreeMesh
        sim : Simulation3DIntegral
        active_cells : np.ndarray  (bool, shape mesh.nC)
        """
        mesh = self.make_mesh()
        active_cells = self.make_active_cells(mesh)
        survey = self.make_survey()
        sim = self.make_simulation(mesh, survey, active_cells)
        data_obj = self.make_data_object(survey)
        misfit = self.make_misfit(sim, data_obj)
        reg = self.make_regularization(mesh, active_cells)
        opt = self.make_optimizer()
        dirs = self.make_directives(sim, reg)

        inv_prob = inverse_problem.BaseInvProblem(misfit, reg, opt)
        return inversion.BaseInversion(inv_prob, dirs), mesh, sim, active_cells

    # ─────────────────────────────────────────────────────────────────────────
    # High-level API
    # ─────────────────────────────────────────────────────────────────────────

    def invert(self) -> xr.Dataset:
        """Run the 3D inversion.

        Returns
        -------
        xr.Dataset
            Dimensions ``(z, y, x)``.  Variables: ``susceptibility``
            (scalar / amplitude) or ``mx`` / ``my`` / ``mz`` (vector).
            Attributes carry CRS and Earth field parameters.
        """
        inv, mesh, sim, active_cells = self.make_inversion()
        m0 = self.make_startmodel(active_cells)
        model = inv.run(m0)
        return self._to_xarray(model, mesh, active_cells)

    def _to_xarray(
        self,
        model: np.ndarray,
        mesh: TreeMesh,
        active_cells: np.ndarray,
    ) -> xr.Dataset:
        """Convert 3D inversion result to an xarray Dataset.

        OcTree cell centres are interpolated (nearest neighbour) onto a
        regular 3D grid suitable for webxtile output.  The grid spacing is
        controlled by ``output__xy_spacing`` / ``output__z_spacing``
        (defaulting to ``mesh__core_cell_size``).

        Returns
        -------
        xr.Dataset
            Dimensions: ``(z, y, x)``.  Variables: ``susceptibility``
            (scalar model) or ``mx`` / ``my`` / ``mz`` (vector model).
        """
        cs = self.mesh__core_cell_size
        xy_step = self.output__xy_spacing or cs
        z_step = self.output__z_spacing or cs

        # Active cell centres and their values
        cc = mesh.cell_centers[active_cells]

        if self.model_type in ("scalar", "amplitude"):
            values_dict = {"susceptibility": model}
            units = "SI"
            long_names = {"susceptibility": "Magnetic susceptibility"}
        else:  # vector
            n = len(model) // 3
            values_dict = {
                "mx": model[:n],
                "my": model[n : 2 * n],
                "mz": model[2 * n :],
            }
            units = "A/m"
            long_names = {
                "mx": "Magnetisation X",
                "my": "Magnetisation Y",
                "mz": "Magnetisation Z",
            }

        # Build regular output grid spanning active cell extent
        xi = np.arange(cc[:, 0].min(), cc[:, 0].max() + xy_step, xy_step)
        yi = np.arange(cc[:, 1].min(), cc[:, 1].max() + xy_step, xy_step)
        zi = np.arange(cc[:, 2].min(), cc[:, 2].max() + z_step, z_step)

        Xi, Yi, Zi = np.meshgrid(xi, yi, zi, indexing="ij")
        grid_pts = np.c_[Xi.ravel(), Yi.ravel(), Zi.ravel()]

        in_attrs = self._grid_data.attrs
        crs_wkt, epsg_code = _crs_to_wkt(in_attrs.get("crs_wkt") or in_attrs.get("epsg_code") or in_attrs.get("crs", ""))

        data_vars = {}
        for vname, vals in values_dict.items():
            interp = NearestNDInterpolator(cc, vals)
            gridded = interp(grid_pts).reshape(Xi.shape)

            # (x, y, z) → reorder to (z, y, x) for CF convention
            data_vars[vname] = xr.DataArray(
                gridded.transpose(2, 1, 0),  # z, y, x
                dims=["z", "y", "x"],
                attrs={
                    "units": units,
                    "long_name": long_names[vname],
                    "grid_mapping": "spatial_ref",
                },
            )
        data_vars["spatial_ref"] = xr.DataArray(
            0,
            attrs=_crs_variable_attrs(crs_wkt, epsg_code),
        )

        return xr.Dataset(
            data_vars,
            coords={
                "z": xr.DataArray(
                    zi,
                    dims=["z"],
                    attrs={
                        "standard_name": "depth",
                        "long_name": "Depth (positive up)",
                        "units": "m",
                        "axis": "Z",
                        "positive": "up",
                    },
                ),
                "y": xr.DataArray(
                    yi,
                    dims=["y"],
                    attrs={
                        "standard_name": "projection_y_coordinate",
                        "long_name": "Northing",
                        "units": "m",
                        "axis": "Y",
                    },
                ),
                "x": xr.DataArray(
                    xi,
                    dims=["x"],
                    attrs={
                        "standard_name": "projection_x_coordinate",
                        "long_name": "Easting",
                        "units": "m",
                        "axis": "X",
                    },
                ),
            },
            attrs={
                "Conventions": "CF-1.8",
                "title": "3D magnetic inversion",
                "model_type": self.model_type,
                "field_intensity_nT": self._field_params[0],
                "field_inclination_deg": self._field_params[1],
                "field_declination_deg": self._field_params[2],
            },
        )

