"""SimPEG directives for magnetics inversion monitoring."""

import numpy as np
import SimPEG.directives


class ReportingDirective(SimPEG.directives.InversionDirective):
    """Directive that logs inversion progress after every iteration.

    Computes and prints phi_d, phi_m, beta, and RMSE at each step.
    Stores logs in self.logs for programmatic access after inversion.
    """

    def __init__(self):
        self.logs = []

    def log(self, data):
        self.logs.append(data)
        print(f"Mag inversion step: {data}")

    def _calc_rmse(self, status):
        W = self.invProb.dmisfit.W
        if hasattr(W, "diagonal"):
            n_data = int(np.sum(W.diagonal() > 0))
        else:
            n_data = int(self.survey.nD)
        if n_data == 0:
            return
        status["rmse_d"] = float(np.sqrt((status["phi_d"] * 2) / n_data))
        status["rmse_m"] = float(np.sqrt((status["phi_m"] * 2) / n_data))

    def endIter(self):
        status = {
            "step": int(self.opt.iter + 2),
            "iter": int(self.opt.iter),
            "beta": float(self.invProb.beta),
            "phi_d": float(self.opt.parent.phi_d * self.opt.parent.opt.factor),
            "phi_m": float(self.opt.parent.phi_m * self.opt.parent.opt.factor),
            "f": float(self.opt.f),
            "status": "update",
        }
        self._calc_rmse(status)
        self.log(status)

    def initialize(self):
        self.log({"step": 1, "status": "initialize"})

    def finish(self):
        self.log({"step": int(self.opt.iter + 2), "status": "end"})
