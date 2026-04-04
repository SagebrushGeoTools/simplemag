"""Equivalent source inversion for airborne magnetics.

Inverts TMI (total magnetic intensity) flight-line data for effective
dipoles on a flat horizontal TensorMesh layer.  Produces gridded Bx, By,
Bz and TMI fields at a uniform output altitude.

Basic usage::

    from mag_inversion import MagEquivalentSourceSystem

    # mag_data is an AirMagTools.MagData instance
    system = MagEquivalentSourceSystem(mag_data)
    ds = system.run()                    # xr.Dataset: tmi/bx/by/bz on regular grid
    ds.webxtile.to_webxtile("equiv_out/")

All class-level attributes can be overridden at instantiation::

    system = MagEquivalentSourceSystem(
        mag_data,
        layer__cell_size=100,
        model_type="vector",
        directives__irls__enable=True,
    )
"""

import os
import tempfile
import typing

import numpy as np
import xarray as xr

from discretize import TensorMesh
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

from .directives import ReportingDirective
from .sensitivity_cache import sensitivity_hash


class MagEquivalentSourceSystem:
    """Equivalent source inversion for airborne TMI data.

    Inverts observed TMI for magnetic dipoles on a single flat horizontal
    layer (TensorMesh, 1 cell thick) placed ``layer__depth_below_flight``
    metres below the minimum observed flight altitude.

    After inversion the forward operator predicts Bx, By, Bz and/or TMI
    on a regular output grid at ``output__altitude``.

    Parameters
    ----------
    mag_data:
        AirMagTools.MagData instance.  Must have a ``data`` DataFrame with
        columns for easting, northing, altitude and TMI (column names are
        configurable via ``columns__*`` attributes), and a ``meta`` dict.
    **kw:
        Any class-level attribute can be overridden here.
    """

    def __init__(self, mag_data, **kw):
        self._mag_data = mag_data
        self.options = kw

    def __getattribute__(self, name):
        options = object.__getattribute__(self, "options")
        if name in options:
            return options[name]
        return object.__getattribute__(self, name)

    # ── Column names ──────────────────────────────────────────────────────────
    columns__x: str = "UTMX"
    "Easting column in mag_data.data (projected, metres)."
    columns__y: str = "UTMY"
    "Northing column in mag_data.data (projected, metres)."
    columns__altitude: str = "alt"
    "Flight altitude column in mag_data.data (metres, same vertical datum as z)."
    columns__tmi: str = "tmi"
    "Total magnetic intensity column (nT)."
    columns__tmi_std: typing.Optional[str] = None
    "Uncertainty column (nT). If None, computed from uncertainties__std_data + floor."

    # ── Earth field ───────────────────────────────────────────────────────────
    field_intensity: typing.Optional[float] = None
    "Earth field intensity (nT). If None, read from mag_data.meta['field_intensity']."
    field_inclination: typing.Optional[float] = None
    "Earth field inclination (deg). If None, read from mag_data.meta['field_inclination']."
    field_declination: typing.Optional[float] = None
    "Earth field declination (deg). If None, read from mag_data.meta['field_declination']."

    # ── Mesh ──────────────────────────────────────────────────────────────────
    layer__depth_below_flight: float = 30.0
    "Depth of equivalent source layer below minimum flight altitude (m)."
    layer__cell_size: float = 50.0
    "Horizontal cell size of the layer mesh (m)."
    layer__cell_thickness: float = 5.0
    "Thickness of the single equivalent source layer cell (m)."
    layer__padding_cells: int = 8
    "Number of padding cells added around the data extent on each side."

    # ── Model ─────────────────────────────────────────────────────────────────
    model_type: typing.Literal["scalar", "vector"] = "scalar"
    """Model type.

    ``"scalar"`` — susceptibility model, assumes induced magnetisation
    aligned with the Earth's field (one parameter per cell).

    ``"vector"`` — MVI (Magnetic Vector Inversion), three-component
    magnetisation per cell, handles remanence.
    """

    # ── Uncertainties ─────────────────────────────────────────────────────────
    uncertainties__std_data: float = 0.02
    "Fractional uncertainty: std = std_data × |tmi| + floor_nT."
    uncertainties__floor_nT: float = 1.0
    "Absolute noise floor added to all uncertainties (nT)."

    # ── Starting model ────────────────────────────────────────────────────────
    startmodel__susceptibility: float = 0.0
    "Starting model susceptibility (SI). Zero gives a neutral start."

    # ── Regularization ────────────────────────────────────────────────────────
    regularization__alpha_s: float = 1e-4
    "Smallness weight (penalises departure from reference model)."
    regularization__alpha_x: float = 1.0
    "X-direction smoothness weight."
    regularization__alpha_y: float = 1.0
    "Y-direction smoothness weight."
    # alpha_z is always 0 for a single-layer equivalent source.

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer__max_iter: int = 40
    "Maximum Gauss-Newton iterations."
    optimizer__max_iter_cg: int = 20
    "Maximum conjugate-gradient iterations per GN step."

    # ── Beta schedule ─────────────────────────────────────────────────────────
    directives__beta__seed: typing.Optional[int] = None
    "Random seed for beta estimator. Set for reproducibility."
    directives__beta__beta0_ratio: float = 10.0
    "Ratio of initial beta to eigenvalue estimate."
    directives__beta__cooling_factor: float = 2.0
    "Factor by which beta is divided at each cooling step."
    directives__beta__cooling_rate: int = 1
    "Number of GN iterations between cooling steps."

    # ── IRLS (sparse regularization) ──────────────────────────────────────────
    directives__irls__enable: bool = False
    "Run IRLS after L2 convergence to produce a sparse model."
    directives__irls__max_iterations: int = 20
    "Maximum IRLS iterations."
    directives__irls__minGNiter: int = 1
    directives__irls__fix_Jmatrix: bool = True
    directives__irls__f_min_change: float = 1e-3
    directives__irls__coolingRate: int = 1

    # ── Sensitivity matrix ─────────────────────────────────────────────────────
    store_sensitivities: typing.Literal["disk", "ram", "forward_only"] = "disk"
    """How to store the sensitivity matrix G.

    ``"disk"``         — write to .npy on disk; memmap'd during inversion.
                         Required for large surveys (> a few GB).
    ``"ram"``          — keep entirely in RAM; fastest but memory-limited.
    ``"forward_only"`` — never store; recompute each Jvec (slowest).
    """
    sensitivity_path: typing.Optional[str] = None
    """Directory for on-disk sensitivity storage.

    If None a temporary directory is used.  Set to a persistent location
    (e.g. a local mount of blob storage) to cache G across runs.
    A hash-based subdirectory is appended automatically so that cached
    matrices are reused only when the mesh and survey match exactly.
    """

    # ── Output grid ───────────────────────────────────────────────────────────
    output__altitude: typing.Optional[float] = None
    "Output grid altitude (m). If None, uses mean flight altitude."
    output__xy_spacing: typing.Optional[float] = None
    "Output grid spacing (m). If None, uses layer__cell_size."
    output__components: list = ["tmi", "bx", "by", "bz"]
    "Field components to predict on the output grid."

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def _field_params(self) -> list:
        """Resolved [intensity_nT, inclination_deg, declination_deg]."""
        meta = self._mag_data.meta
        intensity = (
            self.field_intensity
            if self.field_intensity is not None
            else float(meta.get("field_intensity", meta.get("B0", 50000.0)))
        )
        inclination = (
            self.field_inclination
            if self.field_inclination is not None
            else float(meta.get("field_inclination", meta.get("I", 70.0)))
        )
        declination = (
            self.field_declination
            if self.field_declination is not None
            else float(meta.get("field_declination", meta.get("D", 0.0)))
        )
        return [intensity, inclination, declination]

    @property
    def _receiver_locations(self) -> np.ndarray:
        """Nx3 float array of (x, y, altitude) for all observations."""
        df = self._mag_data.data.reset_index()
        x = df[self.columns__x].values.astype(float)
        y = df[self.columns__y].values.astype(float)
        z = df[self.columns__altitude].values.astype(float)
        return np.c_[x, y, z]

    @property
    def _tmi_data(self) -> np.ndarray:
        df = self._mag_data.data.reset_index()
        return df[self.columns__tmi].values.astype(float)

    @property
    def _uncertainty(self) -> np.ndarray:
        df = self._mag_data.data.reset_index()
        tmi = self._tmi_data
        if self.columns__tmi_std and self.columns__tmi_std in df.columns:
            std = df[self.columns__tmi_std].values.astype(float)
            std = np.where(np.isnan(std), self.uncertainties__floor_nT, std)
        else:
            std = self.uncertainties__std_data * np.abs(tmi) + self.uncertainties__floor_nT
        return std

    @property
    def _layer_altitude(self) -> float:
        """Z coordinate of the equivalent source layer (m)."""
        return float(np.nanmin(self._receiver_locations[:, 2])) - self.layer__depth_below_flight

    def _resolved_sensitivity_path(self) -> str:
        """Return hash-keyed sensitivity directory, creating it if needed."""
        base = self.sensitivity_path or os.path.join(
            tempfile.gettempdir(), "mag_equiv_sensitivity"
        )
        locs = self._receiver_locations
        h = sensitivity_hash(
            receiver_locations=locs,
            field_params=self._field_params,
            mesh_params={
                "depth_below_flight": self.layer__depth_below_flight,
                "cell_size": self.layer__cell_size,
                "cell_thickness": self.layer__cell_thickness,
                "padding_cells": self.layer__padding_cells,
            },
            model_type=self.model_type,
        )
        path = os.path.join(base, h)
        os.makedirs(path, exist_ok=True)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # SimPEG component factories
    # ─────────────────────────────────────────────────────────────────────────

    def make_survey(self) -> magnetics.survey.Survey:
        """Build the TMI survey at flight-line receiver locations."""
        receiver = magnetics.receivers.Point(
            self._receiver_locations, components=["tmi"]
        )
        source = magnetics.sources.SourceField(
            receiver_list=[receiver], parameters=self._field_params
        )
        return magnetics.survey.Survey(source)

    def make_mesh(self) -> TensorMesh:
        """Build the flat TensorMesh for the equivalent source layer."""
        locs = self._receiver_locations
        cs = self.layer__cell_size
        pad = self.layer__padding_cells * cs

        x_min, x_max = locs[:, 0].min(), locs[:, 0].max()
        y_min, y_max = locs[:, 1].min(), locs[:, 1].max()

        nx = max(1, int(np.ceil((x_max - x_min + 2 * pad) / cs)))
        ny = max(1, int(np.ceil((y_max - y_min + 2 * pad) / cs)))

        hx = np.ones(nx) * cs
        hy = np.ones(ny) * cs
        hz = np.array([self.layer__cell_thickness])

        x0 = x_min - pad - (nx * cs - (x_max - x_min + 2 * pad)) / 2
        y0 = y_min - pad - (ny * cs - (y_max - y_min + 2 * pad)) / 2
        z0 = self._layer_altitude

        return TensorMesh([hx, hy, hz], origin=[x0, y0, z0])

    def make_simulation(
        self, mesh: TensorMesh, survey: magnetics.survey.Survey
    ) -> magnetics.simulation.Simulation3DIntegral:
        """Build the integral forward simulation."""
        n_active = mesh.nC
        chi_map = maps.IdentityMap(nP=n_active)
        sens_path = self._resolved_sensitivity_path()

        return magnetics.simulation.Simulation3DIntegral(
            mesh=mesh,
            survey=survey,
            chiMap=chi_map,
            actInd=np.ones(mesh.nC, dtype=bool),
            store_sensitivities=self.store_sensitivities,
            sensitivity_path=sens_path + os.sep,
            model_type=self.model_type,
        )

    def make_data_object(self, survey: magnetics.survey.Survey) -> data.Data:
        """Build the SimPEG Data container."""
        return data.Data(
            survey,
            dobs=self._tmi_data,
            standard_deviation=self._uncertainty,
        )

    def make_misfit(
        self,
        simulation: magnetics.simulation.Simulation3DIntegral,
        data_obj: data.Data,
    ) -> data_misfit.L2DataMisfit:
        """Build the L2 data misfit with per-datum weighting."""
        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_obj)
        dmis.W = 1.0 / self._uncertainty
        return dmis

    def make_startmodel(self, mesh: TensorMesh) -> np.ndarray:
        """Build starting model vector."""
        n_params = mesh.nC if self.model_type == "scalar" else 3 * mesh.nC
        return np.full(n_params, self.startmodel__susceptibility)

    def make_regularization(
        self, mesh: TensorMesh
    ) -> regularization.Tikhonov:
        """Build Tikhonov regularization.

        alpha_z is forced to zero because the equivalent source has only
        one cell in the vertical direction.
        """
        reg = regularization.Tikhonov(
            mesh,
            mapping=maps.IdentityMap(nP=mesh.nC),
            alpha_s=self.regularization__alpha_s,
            alpha_x=self.regularization__alpha_x,
            alpha_y=self.regularization__alpha_y,
            alpha_z=0.0,
        )
        return reg

    def make_optimizer(self) -> optimization.InexactGaussNewton:
        """Build the Gauss-Newton optimizer."""
        return optimization.InexactGaussNewton(
            maxIter=self.optimizer__max_iter,
            maxIterCG=self.optimizer__max_iter_cg,
        )

    def make_directives(self) -> list:
        """Build the list of inversion directives."""
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
        """Assemble and return the full SimPEG Inversion object.

        Returns
        -------
        inv : SimPEG.inversion.BaseInversion
        mesh : TensorMesh
        sim : Simulation3DIntegral
        """
        survey = self.make_survey()
        mesh = self.make_mesh()
        sim = self.make_simulation(mesh, survey)
        data_obj = self.make_data_object(survey)
        misfit = self.make_misfit(sim, data_obj)
        reg = self.make_regularization(mesh)
        opt = self.make_optimizer()
        dirs = self.make_directives()

        inv_prob = inverse_problem.BaseInvProblem(misfit, reg, opt)
        return inversion.BaseInversion(inv_prob, dirs), mesh, sim

    # ─────────────────────────────────────────────────────────────────────────
    # High-level API
    # ─────────────────────────────────────────────────────────────────────────

    def invert(self):
        """Run the equivalent source inversion.

        Returns
        -------
        model : np.ndarray
            Recovered model (susceptibility or magnetization vector).
        mesh : TensorMesh
        sim : Simulation3DIntegral
        """
        inv, mesh, sim = self.make_inversion()
        m0 = self.make_startmodel(mesh)
        model = inv.run(m0)
        return model, mesh, sim

    def predict_on_grid(self, model, mesh, sim):
        """Forward-predict fields on a regular horizontal grid.

        Parameters
        ----------
        model, mesh, sim:
            Output of :meth:`invert`.

        Returns
        -------
        xi : np.ndarray  shape (nx,)
        yi : np.ndarray  shape (ny,)
        fields : dict[str, np.ndarray shape (ny, nx)]
            Predicted fields keyed by component name.
        alt_out : float
            Altitude used for the output grid (m).
        """
        locs = self._receiver_locations
        spacing = (
            self.output__xy_spacing
            if self.output__xy_spacing is not None
            else self.layer__cell_size
        )
        alt_out = (
            self.output__altitude
            if self.output__altitude is not None
            else float(np.nanmean(locs[:, 2]))
        )

        xi = np.arange(locs[:, 0].min(), locs[:, 0].max() + spacing, spacing)
        yi = np.arange(locs[:, 1].min(), locs[:, 1].max() + spacing, spacing)
        Xi, Yi = np.meshgrid(xi, yi)
        grid_locs = np.c_[Xi.ravel(), Yi.ravel(), np.full(Xi.size, alt_out)]

        components = list(self.output__components)

        # Build a prediction survey at the output grid locations
        receiver_pred = magnetics.receivers.Point(grid_locs, components=components)
        source_pred = magnetics.sources.SourceField(
            receiver_list=[receiver_pred], parameters=self._field_params
        )
        survey_pred = magnetics.survey.Survey(source_pred)

        # Use forward_only for prediction (G not reused here)
        sim_pred = magnetics.simulation.Simulation3DIntegral(
            mesh=mesh,
            survey=survey_pred,
            chiMap=maps.IdentityMap(nP=mesh.nC),
            actInd=np.ones(mesh.nC, dtype=bool),
            store_sensitivities="forward_only",
            model_type=self.model_type,
        )

        dpred = sim_pred.dpred(model)

        nD_per_comp = len(grid_locs)
        fields = {
            comp: dpred[i * nD_per_comp : (i + 1) * nD_per_comp].reshape(Xi.shape)
            for i, comp in enumerate(components)
        }
        return xi, yi, fields, alt_out

    def to_xarray(self, model, mesh, sim) -> xr.Dataset:
        """Convert inversion result to an xarray Dataset.

        The dataset has dimensions ``(y, x)`` and one variable per
        predicted component. Attributes carry the CRS, altitude, and
        Earth field parameters so that downstream processes (e.g.
        MagInversion3DSystem) can read them without extra arguments.
        """
        xi, yi, fields, alt_out = self.predict_on_grid(model, mesh, sim)

        meta = self._mag_data.meta
        crs = str(meta.get("crs", ""))

        data_vars = {
            comp: xr.DataArray(
                values,
                dims=["y", "x"],
                attrs={"units": "nT", "long_name": comp.upper()},
            )
            for comp, values in fields.items()
        }

        return xr.Dataset(
            data_vars,
            coords={"y": yi, "x": xi},
            attrs={
                "crs": crs,
                "output_altitude_m": float(alt_out),
                "field_intensity_nT": self._field_params[0],
                "field_inclination_deg": self._field_params[1],
                "field_declination_deg": self._field_params[2],
                "Conventions": "CF-1.8",
                "title": "Magnetic equivalent source — gridded fields",
            },
        )

    def run(self) -> xr.Dataset:
        """Full pipeline: invert then predict on grid.

        Returns
        -------
        xr.Dataset
            Gridded magnetic fields ready for webxtile output.
        """
        model, mesh, sim = self.invert()
        return self.to_xarray(model, mesh, sim)
