"""PINN-SR-style hybrid identification (claude_try2), after Chen, Liu & Sun (Nat. Commun. 2021) adapted to the
inverse-dynamics gap of a manipulator.

    surrogate    q_hat_k(t) = NN_k(Fourier features of t)          one network per trajectory
    derivatives  qd_hat, qdd_hat by autograd
    library      fr3_twin.build_library at (q_hat, qd_hat, qdd_hat) on the 200 Hz sample grid
    sparse fit   identify.stlsq_group with physical bounds (same regression as SINDy-group)
    physics loss || tau_meas - ID_cal(q_hat, qd_hat, qdd_hat; xi) ||^2 / sigma_tau^2
                 ID_cal = inverse dynamics of the calibrated MuJoCo model (+ non-native friction terms);
                 its Jacobians w.r.t. (q, qd, qdd) come from mujoco.mjd_inverseFD (finite differences) and enter
                 autograd through a custom Function.
    ADO          pretrain (data) -> sparse fit -> refine NN (data + physics) -> sparse fit -> ...
"""
import numpy as np
import torch
import torch.nn as nn
import mujoco
import fr3_twin as ft
import identify as idf

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class FourierNet(nn.Module):
    """q(t) = q0 + W phi(t) + MLP(phi(t)); phi = Fourier features up to n_harm harmonics of 1/T plus t/T.
    The linear part is initialised by least squares (global smoother); the MLP (zero-initialised output) adds the
    non-band-limited detail (stick-slip kinks) that the derivatives need."""

    def __init__(self, T, n_harm=100, hidden=128, layers=2, q0=None):
        super().__init__()
        self.T = T
        n_harm = int(n_harm)
        self.register_buffer("w", 2 * np.pi * torch.arange(1, n_harm + 1, dtype=torch.float64) / T)
        nf = 2 * n_harm + 1
        self.lin = nn.Linear(nf, 7).double()
        dims = [nf] + [hidden] * layers + [7]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.SiLU())
        self.mlp = nn.Sequential(*mods).double()
        with torch.no_grad():
            self.mlp[-1].weight.zero_(); self.mlp[-1].bias.zero_()
        self.register_buffer("q0", torch.zeros(7, dtype=torch.float64) if q0 is None else torch.tensor(q0, dtype=torch.float64))

    def features(self, t):
        t = t.reshape(-1, 1)
        ph = t * self.w
        return torch.cat([torch.sin(ph), torch.cos(ph), t / self.T], dim=1)

    def forward(self, t):
        x = self.features(t)
        return self.q0 + self.lin(x) + self.mlp(x)

    def init_linear(self, t, q, ridge=1e-6):
        with torch.no_grad():
            X = self.features(t)
            Y = q - self.q0
            Xa = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], dim=1)
            G = Xa.T @ Xa + ridge * torch.eye(Xa.shape[1], dtype=X.dtype, device=X.device)
            W = torch.linalg.solve(G, Xa.T @ Y)
            self.lin.weight.copy_(W[:-1].T); self.lin.bias.copy_(W[-1])

    def derivs(self, t, create_graph=False):
        t = t.reshape(-1, 1).requires_grad_(True)
        q = self.forward(t)
        qd = torch.stack([torch.autograd.grad(q[:, j].sum(), t, create_graph=True, retain_graph=True)[0][:, 0] for j in range(7)], dim=1)
        qdd = torch.stack([torch.autograd.grad(qd[:, j].sum(), t, create_graph=create_graph, retain_graph=True)[0][:, 0] for j in range(7)], dim=1)
        return q, qd, qdd


class CalibratedID:
    """tau = ID_cal(q, qd, qdd) with finite-difference Jacobians (MuJoCo mjd_inverseFD + numeric for extras)."""

    def __init__(self, model, extra=None, eps=1e-6):
        self.model, self.extra, self.eps = model, extra, eps
        self.data = mujoco.MjData(model)
        nv = model.nv
        self.Jq, self.Jv, self.Ja = (np.zeros((nv, nv)) for _ in range(3))

    def __call__(self, q, v, a, jac=False):
        m, d = self.model, self.data
        n = q.shape[0]
        tau = np.empty((n, 7))
        J = np.empty((n, 3, 7, 7)) if jac else None
        for i in range(n):
            d.qpos[:] = q[i]; d.qvel[:] = v[i]; d.qacc[:] = a[i]
            mujoco.mj_inverse(m, d)
            tau[i] = d.qfrc_inverse
            if self.extra is not None:
                tau[i] += self.extra(q[i], v[i], a[i])
            if jac:
                mujoco.mjd_inverseFD(m, d, self.eps, False, self.Jq, self.Jv, self.Ja, None, None, None, None)
                J[i, 0], J[i, 1], J[i, 2] = self.Jq, self.Jv, self.Ja          # transposed: J[k, l] = d tau_l / d x_k
                if self.extra is not None:                                    # extras depend on v only
                    for k in range(7):
                        vp = v[i].copy(); vp[k] += self.eps
                        J[i, 1, k] += (self.extra(q[i], vp, a[i]) - self.extra(q[i], v[i], a[i])) / self.eps
        return (tau, J) if jac else tau


class IDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, v, a, idcal):
        tau, J = idcal(q.detach().cpu().numpy(), v.detach().cpu().numpy(), a.detach().cpu().numpy(), jac=True)
        ctx.J = torch.tensor(J, dtype=q.dtype, device=q.device)
        return torch.tensor(tau, dtype=q.dtype, device=q.device)

    @staticmethod
    def backward(ctx, g):
        J = ctx.J                                                             # (n, 3, 7, 7), J[:, s, k, l] = d tau_l / d x_k
        gq = torch.einsum("nkl,nl->nk", J[:, 0], g)
        gv = torch.einsum("nkl,nl->nk", J[:, 1], g)
        ga = torch.einsum("nkl,nl->nk", J[:, 2], g)
        return gq, gv, ga, None


class PINNSR:
    def __init__(self, model_nom, sigma_q=1e-4, sigma_tau=0.05, f_max=2.5, hidden=64, pre_steps=1500, ref_steps=200,
                 rounds=2, batch_phys=192, lr=1e-3, dec=5, trim=0.2, seed=0, fit_method="SINDy-group", verbose=None,
                 lam_grid=(0.1, 1.0, 10.0, 100.0, 1000.0), chi2_max=1.3, a0=10.0):
        self.model_nom = model_nom
        self.sigma_q, self.sigma_tau = sigma_q, sigma_tau
        self.f_max, self.hidden, self.pre_steps, self.ref_steps, self.rounds = f_max, hidden, pre_steps, ref_steps, rounds
        self.lam_grid, self.chi2_max, self.a0 = list(lam_grid), chi2_max, a0
        self.batch_phys, self.lr, self.dec, self.trim, self.seed, self.fit_method = batch_phys, lr, dec, trim, seed, fit_method
        self.say = verbose if verbose is not None else (lambda *a: None)

    # ----- surrogate
    def _pretrain(self, net, t, q, steps, lam_s):
        """data chi2 + lam_s * mean(qdd^2)/a0^2: a spline-like smoother in the surrogate."""
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        n = t.shape[0]
        for s in range(steps):
            idx = torch.randint(0, n, (4096,), device=DEV)
            l_data = ((net(t[idx]) - q[idx]) ** 2).mean() / self.sigma_q ** 2
            jdx = torch.randint(0, n, (1024,), device=DEV)
            _, _, qdd = net.derivs(t[jdx], create_graph=True)
            loss = l_data + lam_s * (qdd ** 2).mean() / self.a0 ** 2
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            chi2 = ((net(t) - q) ** 2).mean().item() / self.sigma_q ** 2
        return chi2

    def _pretrain_auto(self, net, t, q, steps):
        """Discrepancy principle: the largest smoothing weight whose data chi2 per sample stays below chi2_max."""
        state0 = {k: v.clone() for k, v in net.state_dict().items()}
        best = None
        for lam in sorted(self.lam_grid, reverse=True):     # smoothest first; stop at the first admissible fit
            net.load_state_dict(state0)
            chi2 = self._pretrain(net, t, q, steps, lam)
            self.say(f"    smoothing lam {lam:g}: data chi2 {chi2:.2f}")
            best = (lam, chi2, {k: v.clone() for k, v in net.state_dict().items()})
            if chi2 <= self.chi2_max:
                break
        net.load_state_dict(best[2])
        return best[0], best[1]

    def _refine(self, net, t, q, t_lib, tau_lib, idcal, steps, lam_s):
        opt = torch.optim.Adam(net.parameters(), lr=self.lr * 0.3)
        n, nl = t.shape[0], t_lib.shape[0]
        for s in range(steps):
            idx = torch.randint(0, n, (4096,), device=DEV)
            l_data = ((net(t[idx]) - q[idx]) ** 2).mean() / self.sigma_q ** 2
            jdx = torch.randint(0, nl, (self.batch_phys,), device=DEV)
            qh, vh, ah = net.derivs(t_lib[jdx], create_graph=True)
            tau_hat = IDFunction.apply(qh, vh, ah, idcal)
            l_phys = ((tau_hat - tau_lib[jdx]) ** 2).mean() / self.sigma_tau ** 2
            loss = l_data + l_phys + lam_s * (ah ** 2).mean() / self.a0 ** 2
            opt.zero_grad(); loss.backward(); opt.step()
        return l_data.item(), l_phys.item()

    def _dataset(self):
        """identify-compatible dataset from the surrogates on the decimated grid."""
        data = []
        for k, log in enumerate(self.logs):
            net, tl = self.nets[k], self.t_lib[k]
            with torch.enable_grad():
                qh, vh, ah = net.derivs(tl)
            q, qd, qdd = (x.detach().cpu().numpy() for x in (qh, vh, ah))
            tau = self.tau_lib[k].cpu().numpy()
            tau_nom, A = ft.build_library(self.model_nom, q, qd, qdd)
            data.append(dict(q=q, qd=qd, qdd=qdd, tau=tau, tau_nom=tau_nom, A=A, r=tau - tau_nom, seed=log["seed"], t=tl.cpu().numpy()))
        return data

    def fit(self, logs):
        torch.manual_seed(self.seed)
        self.logs = logs
        self.nets, self.t_all, self.q_all, self.t_lib, self.tau_lib = [], [], [], [], []
        for log in logs:
            dt = log["t"][1] - log["t"][0]
            T = log["t"][-1] + dt
            n_trim = int(round(self.trim / dt))
            sl = slice(n_trim, len(log["t"]) - n_trim, self.dec)
            net = FourierNet(T, max(4, int(self.f_max * T)), self.hidden, q0=log["q_meas"][0]).to(DEV)
            t = torch.tensor(log["t"], dtype=torch.float64, device=DEV)
            q = torch.tensor(log["q_meas"], dtype=torch.float64, device=DEV)
            self.nets.append(net); self.t_all.append(t); self.q_all.append(q)
            self.t_lib.append(torch.tensor(log["t"][sl], dtype=torch.float64, device=DEV))
            self.tau_lib.append(torch.tensor(log["tau_meas"][sl], dtype=torch.float64, device=DEV))
        self.lam_s = []
        for k, net in enumerate(self.nets):
            net.init_linear(self.t_all[k], self.q_all[k])
            lam, chi2 = self._pretrain_auto(net, self.t_all[k], self.q_all[k], self.pre_steps)
            self.lam_s.append(lam)
            self.say(f"  pinnsr pretrain traj {k}: smoothing {lam:g}, data chi2 {chi2:.2f}")
        self.history = []
        for r in range(self.rounds + 1):
            data = self._dataset()
            self.xi, info = idf.identify(data, self.fit_method, model_nom=self.model_nom)
            tm = idf.torque_metrics(data, self.xi)
            self.history.append(dict(round=r, nrmse_train=tm["nrmse"], nnz=int((self.xi != 0).sum()), thr=info.get("thr")))
            self.say(f"  pinnsr round {r}: train NRMSE {tm['nrmse']:.3f} nnz {int((self.xi != 0).sum())}")
            if r == self.rounds:
                break
            model_cal, extra, _ = idf.instantiate(self.model_nom, self.xi)
            idcal = CalibratedID(model_cal, extra)
            for k, net in enumerate(self.nets):
                ld, lp = self._refine(net, self.t_all[k], self.q_all[k], self.t_lib[k], self.tau_lib[k], idcal, self.ref_steps, self.lam_s[k])
                self.say(f"    refine traj {k}: data {ld:.2f} phys {lp:.2f}")
        self.data = data
        return self

    def derivative_error(self, logs):
        """RMS error of the surrogate derivatives against the true (simulator) ones on the library grid."""
        out = []
        for k, log in enumerate(logs):
            dt = log["t"][1] - log["t"][0]
            n_trim = int(round(self.trim / dt)); sl = slice(n_trim, len(log["t"]) - n_trim, self.dec)
            d = self.data[k]
            out.append(dict(qd=float(np.sqrt(((d["qd"] - log["qd"][sl]) ** 2).mean())), qdd=float(np.sqrt(((d["qdd"] - log["qdd"][sl]) ** 2).mean()))))
        return out
