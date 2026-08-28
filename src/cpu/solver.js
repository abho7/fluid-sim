/**
 * The CPU reference solver, in f64.
 *
 * This is the correctness oracle for the whole project. Its job is to be right,
 * not fast: it uses conjugate gradient for the projection and double precision
 * throughout, so that the analytic validation measures the SCHEME's error and
 * not the solver's sloppiness. The GPU backend implements the same algorithm in
 * f32 with iterative solvers and is checked field-by-field against this.
 *
 * ONE STEP, in order:
 *   1. advect velocity through itself
 *   2. diffuse (viscosity)
 *   3. add body forces, including vorticity confinement
 *   4. project onto the divergence-free subspace
 *   5. advect the dye through the (now divergence-free) velocity
 *
 * The ordering is not arbitrary. Projection is LAST among the velocity steps
 * because advection, diffusion and forces each introduce divergence, and
 * whatever runs after the projection leaves the field non-solenoidal for the
 * rest of the step. Dye is advected after the projection because advecting a
 * scalar through a divergent field does not conserve it -- dye would visibly
 * pool and thin out, which reads as "the fluid is compressing" and is purely an
 * ordering artifact.
 */

import {
  divergence, subtractGradient, vorticityCorner, wrap,
  kineticEnergy, enstrophy, l2Divergence, maxDivergence, cflNumber,
} from "../core/grid.js";
import { Grid } from "../core/grid.js";
import { ADVECTION } from "./advect.js";
import { solvePoissonCG, solvePoissonFFT, solvePoissonJacobi, laplacian } from "./projection.js";
import { fft2, ifft2, isPow2 } from "../core/fft.js";
import { applyCoupling } from "./solid.js";

/**
 * Implicit diffusion, solved with conjugate gradient.
 *
 *   (I − ν·dt·∇²) u_new = u_old
 *
 * Implicit rather than explicit because explicit diffusion is stable only for
 * ν·dt/h² ≤ 1/4 in 2D, and at the resolutions this project targets that bound
 * forces a timestep far smaller than advection needs. Stam's method exists to
 * decouple the timestep from stability, and an explicit diffusion step would
 * hand that back.
 *
 * `explicit: true` deliberately switches to the unstable version -- not as a
 * fallback but as an instrument. The stability study sweeps ν·dt/h² across 1/4
 * and needs a scheme that genuinely blows up in order to locate the boundary
 * with data instead of asserting the textbook value.
 */
export function diffuse(g, field, out, nu, dt, { explicit = false, tol = 1e-12 } = {}) {
  if (nu <= 0) { out.set(field); return { iterations: 0, residual: 0 }; }
  const { nx, ny } = g;

  if (explicit) {
    const alpha = nu * dt / (g.dx * g.dx);
    const beta = nu * dt / (g.dy * g.dy);
    for (let j = 0; j < ny; j++) {
      const jm = wrap(j - 1, ny) * nx, jp = wrap(j + 1, ny) * nx, j0 = j * nx;
      for (let i = 0; i < nx; i++) {
        const im = wrap(i - 1, nx), ip = wrap(i + 1, nx);
        out[j0 + i] = field[j0 + i]
          + alpha * (field[j0 + im] - 2 * field[j0 + i] + field[j0 + ip])
          + beta * (field[jm + i] - 2 * field[j0 + i] + field[jp + i]);
      }
    }
    return { iterations: 1, residual: 0, explicit: true };
  }

  // CG on (I − ν·dt·∇²). Symmetric positive definite for ν·dt > 0, and unlike
  // the pressure Poisson system it has no null space, so no mean removal.
  const n = field.length;
  const k = nu * dt;
  const applyA = (x, y) => {
    laplacian(g, x, y);
    for (let i = 0; i < n; i++) y[i] = x[i] - k * y[i];
    return y;
  };

  out.set(field);                       // warm start from the current field
  const r = new Float64Array(n);
  const Ap = new Float64Array(n);
  applyA(out, Ap);
  for (let i = 0; i < n; i++) r[i] = field[i] - Ap[i];

  const d = Float64Array.from(r);
  let rr = 0;
  for (let i = 0; i < n; i++) rr += r[i] * r[i];
  const bNorm = Math.sqrt(field.reduce((s, x) => s + x * x, 0)) || 1;

  let iterations = 0;
  for (let it = 0; it < 400; it++) {
    if (Math.sqrt(rr) / bNorm <= tol) break;
    applyA(d, Ap);
    let dAd = 0;
    for (let i = 0; i < n; i++) dAd += d[i] * Ap[i];
    if (Math.abs(dAd) < 1e-300) break;
    const alpha = rr / dAd;
    let rrNew = 0;
    for (let i = 0; i < n; i++) {
      out[i] += alpha * d[i];
      r[i] -= alpha * Ap[i];
      rrNew += r[i] * r[i];
    }
    const beta = rrNew / rr;
    for (let i = 0; i < n; i++) d[i] = r[i] + beta * d[i];
    rr = rrNew;
    iterations = it + 1;
  }
  return { iterations, residual: Math.sqrt(rr) / bNorm };
}

/**
 * Vorticity confinement (Fedkiw et al. 2001).
 *
 * READ THIS BEFORE USING IT IN ANYTHING MEASURED. This force is not physical.
 * It computes the gradient of |ω|, normalises it to get a unit vector pointing
 * from low to high vorticity, and applies f = ε·h·(N × ω) -- a force that
 * pushes energy INTO existing vortices. It is a fabricated energy source whose
 * only justification is that it happens to counteract the energy the advection
 * scheme is losing.
 *
 * It does not restore the lost information; it manufactures plausible-looking
 * replacement detail. So a kinetic-energy decay rate or an energy spectrum
 * measured with this enabled is measuring the confinement parameter ε, not the
 * fluid. Every validation run in this project has it off, and the solver
 * refuses to enable it in validation mode rather than relying on the caller to
 * remember.
 */
export function vorticityConfinement(g, u, v, eps, dt) {
  if (eps <= 0) return;
  const { nx, ny } = g;
  const w = vorticityCorner(g, u, v);
  const absw = new Float64Array(nx * ny);
  for (let i = 0; i < w.length; i++) absw[i] = Math.abs(w[i]);

  // N = ∇|ω| / |∇|ω||, evaluated at corners alongside ω itself.
  const Nx = new Float64Array(nx * ny), Ny = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const gx = (absw[wrap(j, ny) * nx + wrap(i + 1, nx)] - absw[wrap(j, ny) * nx + wrap(i - 1, nx)]) / (2 * g.dx);
      const gy = (absw[wrap(j + 1, ny) * nx + wrap(i, nx)] - absw[wrap(j - 1, ny) * nx + wrap(i, nx)]) / (2 * g.dy);
      const m = Math.hypot(gx, gy) + 1e-20;   // guard: |∇|ω|| is exactly 0 in uniform flow
      Nx[j * nx + i] = gx / m;
      Ny[j * nx + i] = gy / m;
    }
  }

  // f = ε·h·(N × ω ẑ) = ε·h·(N_y·ω, −N_x·ω), interpolated to the faces.
  const s = eps * g.h * dt;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const c00 = j * nx + i;
      const c01 = wrap(j + 1, ny) * nx + i;
      const fu = 0.5 * (Ny[c00] * w[c00] + Ny[c01] * w[c01]);
      u[g.idxU(i, j)] += s * fu;

      const c10 = j * nx + wrap(i + 1, nx);
      const fv = -0.5 * (Nx[c00] * w[c00] + Nx[c10] * w[c10]);
      v[g.idxV(i, j)] += s * fv;
    }
  }
}

/**
 * Band-limited random forcing in an annulus of wavenumber space.
 *
 * Required for the 2D turbulence study, and it must be injected at an
 * INTERMEDIATE scale. 2D turbulence has a dual cascade: energy travels to
 * scales larger than the forcing, enstrophy to scales smaller. Forcing at the
 * largest scale would leave no room above it and there would be no inverse
 * cascade to measure; forcing at the grid scale would leave no enstrophy range
 * below. Only forcing in the middle produces both ranges the report compares
 * against Kraichnan.
 *
 * Deterministic given a seed, so a spectrum run reproduces exactly.
 */
export function bandForcing(g, u, v, { kf = 28, width = 2, amp = 0.4, seed = 1, dt = 0.01 }) {
  const { nx, ny } = g;
  let s = seed >>> 0;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  // Sum of a handful of random modes in the shell. Building the force in
  // physical space from an explicit mode list keeps it exactly band-limited;
  // adding spatial noise and filtering would leak energy into the ranges whose
  // slopes are being measured.
  const modes = [];
  for (let n = 0; n < 24; n++) {
    const theta = 2 * Math.PI * rnd();
    const kmag = kf + (rnd() - 0.5) * 2 * width;
    modes.push({
      kx: Math.round(kmag * Math.cos(theta)),
      ky: Math.round(kmag * Math.sin(theta)),
      phase: 2 * Math.PI * rnd(),
    });
  }
  const scale = amp * dt / Math.sqrt(modes.length);
  for (const m of modes) {
    if (m.kx === 0 && m.ky === 0) continue;
    const kk = m.kx * m.kx + m.ky * m.ky;
    // Curl of a scalar streamfunction, so the forcing is divergence-free by
    // construction and the projection has nothing to undo.
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const xu = g.uX(i), yu = g.uY(j);
        u[g.idxU(i, j)] += scale * (m.ky / Math.sqrt(kk)) *
          Math.cos(m.kx * xu + m.ky * yu + m.phase);
        const xv = g.vX(i), yv = g.vY(j);
        v[g.idxV(i, j)] += scale * (-m.kx / Math.sqrt(kk)) *
          Math.cos(m.kx * xv + m.ky * yv + m.phase);
      }
    }
  }
}

/**
 * Large-scale linear drag, −α·u applied only to low wavenumbers.
 *
 * Without this the 2D inverse cascade has nowhere to deposit energy: it piles
 * up in the largest mode the box allows and forms a "condensate", a pair of
 * box-sized vortices that dominate everything and prevent a statistical steady
 * state from ever existing. The spectrum then drifts forever and any slope
 * fitted to it depends on when you stopped. Drag removes energy at large scales
 * at the rate forcing injects it, which is what makes a steady state possible.
 *
 * Implemented as a simple uniform linear drag, which is scale-selective enough
 * in practice because the large scales hold most of the energy.
 */
export function linearDrag(u, v, alpha, dt) {
  if (alpha <= 0) return;
  const f = Math.exp(-alpha * dt);   // exact integration of du/dt = -αu
  for (let i = 0; i < u.length; i++) u[i] *= f;
  for (let i = 0; i < v.length; i++) v[i] *= f;
}

// ==================================================================== solver

export class FluidSolver {
  /**
   * @param {object} opts
   * @param {number} opts.n grid resolution (square)
   * @param {number} opts.nu kinematic viscosity
   * @param {string} opts.advection "semi-lagrangian" | "maccormack"
   * @param {boolean} opts.validation when true, physically refuses vorticity
   *   confinement -- see below
   */
  constructor({
    n = 64, nx = n, ny = n, lx = 2 * Math.PI, ly = 2 * Math.PI,
    nu = 0.01, advection = "semi-lagrangian",
    confinement = 0, explicitDiffusion = false,
    projTol = 1e-10, projMaxIter = 2000,
    projection = "cg",
    solids = [],
    maskWidth = 1.5,
    couplingPasses = 2,
    validation = false,
  } = {}) {
    this.g = new Grid(nx, ny, lx, ly);
    this.nu = nu;
    this.validation = validation;

    if (!ADVECTION[advection]) {
      throw new Error(`unknown advection scheme ${advection}; ` +
        `have ${Object.keys(ADVECTION).join(", ")}`);
    }
    this.advection = advection;
    this.scheme = ADVECTION[advection];

    // The honesty constraint, enforced in code rather than by convention.
    // A validation run measuring numerical viscosity or an energy spectrum
    // with confinement on would be measuring ε. Making that a thrown error
    // rather than a comment means it cannot happen by forgetting.
    if (validation && confinement > 0) {
      throw new Error(
        "vorticity confinement is an artificial energy source and cannot be " +
        "enabled in validation mode -- it would corrupt exactly the " +
        "quantities being measured (energy decay, spectra)."
      );
    }
    this.confinement = confinement;
    this.explicitDiffusion = explicitDiffusion;
    this.projTol = projTol;
    this.projMaxIter = projMaxIter;

    if (!["cg", "fft", "jacobi"].includes(projection)) {
      throw new Error(`unknown projection ${projection}; use "cg", "fft" or "jacobi"`);
    }
    // The FFT solver is exact but only valid on a periodic power-of-two grid.
    // Failing loudly here beats silently falling back, which would make a run
    // that was supposed to be exact quietly iterative and slower.
    if (projection === "fft" && !(isPow2(nx) && isPow2(ny))) {
      throw new Error(
        `fft projection needs power-of-two dimensions, got ${nx}x${ny}`);
    }
    this.projection = projection;

    // Immersed rigid bodies. Empty by default; the solver is unchanged when
    // there are none, so the coupling cannot perturb any existing result.
    this.solids = solids;
    this.maskWidth = maskWidth;
    this.couplingPasses = couplingPasses;

    const g = this.g;
    this.u = g.u();
    this.v = g.v();
    this.dye = g.p();
    this.p = g.p();

    // Scratch, allocated once. Per-step allocation of a few MB at 512² makes
    // the GC the dominant cost and would corrupt the CPU-vs-GPU timing.
    this._tu = g.u(); this._tv = g.v(); this._tq = g.p();
    this._scratch = {
      fwd: g.p(), back: g.p(),
      fu: g.u(), fv: g.v(), bu: g.u(), bv: g.v(),
    };

    this.t = 0;
    this.steps = 0;
    this.lastProjection = null;
  }

  /** Initialise from callbacks over physical coordinates. */
  init(fu, fv, fdye) {
    this.g.fillU(this.u, fu);
    this.g.fillV(this.v, fv);
    if (fdye) this.g.fillP(this.dye, fdye);
    this.t = 0;
    this.steps = 0;
    return this;
  }

  /** Advance one timestep. */
  step(dt, opts = {}) {
    const g = this.g;

    // 1. advect velocity through itself
    this.scheme.velocity(g, this.u, this.v, this._tu, this._tv, dt, this._scratch);
    [this.u, this._tu] = [this._tu, this.u];
    [this.v, this._tv] = [this._tv, this.v];

    // 2. viscous diffusion
    if (this.nu > 0) {
      const o = { explicit: this.explicitDiffusion };
      diffuse(g, this.u, this._tu, this.nu, dt, o);
      diffuse(g, this.v, this._tv, this.nu, dt, o);
      [this.u, this._tu] = [this._tu, this.u];
      [this.v, this._tv] = [this._tv, this.v];
    }

    // 3. body forces
    if (opts.force) opts.force(g, this.u, this.v, dt, this.t);
    if (this.confinement > 0) {
      vorticityConfinement(g, this.u, this.v, this.confinement, dt);
    }
    if (opts.drag > 0) linearDrag(this.u, this.v, opts.drag, dt);

    // 3b. fluid-structure coupling, BEFORE the projection.
    //
    // Direct forcing sets the velocity inside each solid without regard to
    // incompressibility, so it introduces divergence that the projection then
    // removes. Doing it the other way round would leave the field divergent for
    // the rest of the step, and a passive tracer would visibly pool against the
    // obstacle -- an artifact of ordering that reads as the fluid compressing.
    if (this.solids.length) {
      applyCoupling(g, this.u, this.v, this.solids, dt,
        { maskWidth: this.maskWidth, passes: this.couplingPasses });
      for (const sd of this.solids) sd.integrate(dt, { gravity: opts.gravity ?? 0 });
    }

    // 4. projection -- last, so nothing reintroduces divergence afterwards
    const div = divergence(g, this.u, this.v);
    this.p.fill(0);
    // "jacobi" exists so the GPU benchmark can compare the SAME algorithm on
    // both devices. Comparing CPU conjugate gradient against GPU Jacobi would
    // be comparing two different methods and calling the difference hardware.
    this.lastProjection =
      this.projection === "fft"
        ? solvePoissonFFT(g, div, this.p, { fft2, ifft2 })
        : this.projection === "jacobi"
          ? solvePoissonJacobi(g, div, this.p, { iterations: this.projMaxIter })
          : solvePoissonCG(g, div, this.p, {
            tol: this.projTol, maxIter: this.projMaxIter,
          });
    subtractGradient(g, this.u, this.v, this.p, 1);

    // 5. dye, through the divergence-free field
    if (opts.advectDye !== false) {
      this.scheme.scalar(g, this.u, this.v, this.dye, this._tq, dt, this._scratch);
      [this.dye, this._tq] = [this._tq, this.dye];
    }

    this.t += dt;
    this.steps++;
    return this;
  }

  // --------------------------------------------------------------- diagnostics

  kineticEnergy() { return kineticEnergy(this.g, this.u, this.v); }
  enstrophy() { return enstrophy(this.g, this.u, this.v); }
  maxDivergence() { return maxDivergence(this.g, this.u, this.v); }
  l2Divergence() { return l2Divergence(this.g, this.u, this.v); }
  cfl(dt) { return cflNumber(this.g, this.u, this.v, dt); }

  /** True once the run has gone non-finite -- the stability sweeps test this. */
  isBlownUp() {
    for (let i = 0; i < this.u.length; i++) {
      if (!Number.isFinite(this.u[i])) return true;
    }
    for (let i = 0; i < this.v.length; i++) {
      if (!Number.isFinite(this.v[i])) return true;
    }
    return false;
  }
}
