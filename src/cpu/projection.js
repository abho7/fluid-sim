/**
 * Pressure projection: the step that makes the flow incompressible.
 *
 * THE MATHEMATICS. Helmholtz-Hodge says any vector field splits uniquely into
 * a divergence-free part and a gradient:  w = u + ∇p, with ∇·u = 0. Taking the
 * divergence of that gives a Poisson equation,
 *
 *     ∇²p = ∇·w
 *
 * and then u = w − ∇p is the divergence-free field we want. So "enforce
 * incompressibility" means "solve a Poisson equation", and the quality of the
 * whole simulation is the quality of that solve. This is the step the brief
 * calls out as needing a real iterative solver rather than a toy, and it is
 * where a fluid sim is usually cheated.
 *
 * WHY CONJUGATE GRADIENT HERE. The discrete Laplacian on a periodic MAC grid is
 * symmetric negative-definite (up to the constant null space below), which is
 * exactly the class CG is built for. CG converges in at most N iterations in
 * exact arithmetic and in practice reaches machine precision in tens, whereas
 * Jacobi needs thousands. This file is the reference implementation whose job
 * is to be RIGHT, so it uses the method that converges hardest; the GPU side
 * implements Jacobi, red-black Gauss-Seidel and multigrid, and is measured
 * against what this produces.
 *
 * THE NULL SPACE, which must be handled or CG will not converge. On a fully
 * periodic domain, p and p+c give identical gradients, so the Laplacian has a
 * one-dimensional null space (the constants) and is only positive semi-definite.
 * Two consequences:
 *   1. the right-hand side must have zero mean, or the equation is unsolvable
 *      (a compatibility condition, not a numerical detail)
 *   2. the solution must be pinned, here by projecting out the mean each
 *      iteration, or the iterate drifts along the null space
 * Both are done explicitly below. Skipping either produces a solver that
 * "fails to converge" for reasons that look like a bug in the discretisation.
 */

import { divergence, subtractGradient, wrap } from "../core/grid.js";

/**
 * Apply the 5-point Laplacian for cell-centred pressure on a periodic grid.
 *
 * Sign convention: this returns ∇²p with the SAME sign as the continuous
 * operator, so the system being solved is ∇²p = div. The operator is negative
 * definite, which CG handles fine as long as the sign is consistent between the
 * operator and the residual (an inconsistency here makes CG diverge
 * immediately, which is at least a loud failure).
 */
export function laplacian(g, p, out) {
  const { nx, ny, dx, dy } = g;
  const ix = 1 / (dx * dx), iy = 1 / (dy * dy);
  out = out || g.p();
  for (let j = 0; j < ny; j++) {
    const jm = wrap(j - 1, ny) * nx, jp = wrap(j + 1, ny) * nx, j0 = j * nx;
    for (let i = 0; i < nx; i++) {
      const im = wrap(i - 1, nx), ip = wrap(i + 1, nx);
      out[j0 + i] =
        (p[j0 + im] - 2 * p[j0 + i] + p[j0 + ip]) * ix +
        (p[jm + i] - 2 * p[j0 + i] + p[jp + i]) * iy;
    }
  }
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function removeMean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  const m = s / a.length;
  for (let i = 0; i < a.length; i++) a[i] -= m;
  return m;
}

/**
 * Solve ∇²p = rhs by conjugate gradient on a periodic domain.
 *
 * @returns {{iterations:number, residual:number, converged:boolean, history:number[]}}
 *
 * `history` is the residual after each iteration and is what the report's
 * "not a toy approximation" claim is drawn from -- it is kept because a
 * convergence curve is evidence and a final number alone is not.
 */
export function solvePoissonCG(g, rhs, p, { tol = 1e-10, maxIter = 2000 } = {}) {
  const n = rhs.length;
  p = p || g.p();

  // Compatibility: the periodic Poisson equation is only solvable when the RHS
  // integrates to zero. Physically ∮∇·w = 0 already, so a nonzero mean here is
  // pure round-off -- but left in, it is a component with no solution and CG
  // stalls against it forever.
  const work = Float64Array.from(rhs);
  removeMean(work);

  const r = Float64Array.from(work);
  const Ap = g.p();
  laplacian(g, p, Ap);
  for (let i = 0; i < n; i++) r[i] -= Ap[i];
  removeMean(r);

  const d = Float64Array.from(r);
  let rr = dot(r, r);
  const rhsNorm = Math.sqrt(dot(work, work)) || 1;

  const history = [];
  let iter = 0;
  let residual = Math.sqrt(rr) / rhsNorm;

  if (residual <= tol) return { iterations: 0, residual, converged: true, history: [residual] };

  for (; iter < maxIter; iter++) {
    laplacian(g, d, Ap);
    removeMean(Ap);              // keep the search direction out of the null space

    const dAd = dot(d, Ap);
    if (Math.abs(dAd) < 1e-300) break;   // exhausted the Krylov space

    const alpha = rr / dAd;
    for (let i = 0; i < n; i++) {
      p[i] += alpha * d[i];
      r[i] -= alpha * Ap[i];
    }

    const rrNew = dot(r, r);
    residual = Math.sqrt(rrNew) / rhsNorm;
    history.push(residual);
    if (residual <= tol) { iter++; break; }

    const beta = rrNew / rr;
    for (let i = 0; i < n; i++) d[i] = r[i] + beta * d[i];
    rr = rrNew;
  }

  removeMean(p);   // pin the solution; pressure is only defined up to a constant
  return { iterations: iter, residual, converged: residual <= tol, history };
}

/**
 * Damped Jacobi, kept on the CPU as the reference the GPU Jacobi is checked
 * against. Far too slow to use for real work here -- that is the point: the
 * report compares its convergence against CG and multigrid, and a method has to
 * exist to be compared.
 *
 * The iteration for ∇²p = rhs on a uniform grid solves each cell for its own
 * value given its neighbours:
 *     p_new = (Σ neighbours − h²·rhs) / 4
 */
export function solvePoissonJacobi(g, rhs, p, { iterations = 80, omega = 1.0 } = {}) {
  const { nx, ny } = g;
  if (!g.isUniform) throw new Error("Jacobi implementation assumes square cells");
  const h2 = g.dx * g.dx;
  p = p || g.p();
  let cur = p;
  let next = g.p();

  const work = Float64Array.from(rhs);
  removeMean(work);

  const history = [];
  for (let it = 0; it < iterations; it++) {
    for (let j = 0; j < ny; j++) {
      const jm = wrap(j - 1, ny) * nx, jp = wrap(j + 1, ny) * nx, j0 = j * nx;
      for (let i = 0; i < nx; i++) {
        const im = wrap(i - 1, nx), ip = wrap(i + 1, nx);
        const sum = cur[j0 + im] + cur[j0 + ip] + cur[jm + i] + cur[jp + i];
        const jac = (sum - h2 * work[j0 + i]) * 0.25;
        next[j0 + i] = cur[j0 + i] + omega * (jac - cur[j0 + i]);
      }
    }
    const t = cur; cur = next; next = t;
    history.push(poissonResidual(g, cur, work));
  }

  if (cur !== p) p.set(cur);
  removeMean(p);
  return { iterations, residual: history[history.length - 1] ?? NaN, history };
}

/**
 * Solve ∇²p = rhs exactly, by FFT.
 *
 * On a periodic grid the discrete Laplacian is DIAGONAL in Fourier space, so
 * the Poisson equation is not an iterative problem at all -- it is a division.
 * Transform, divide by the eigenvalue, transform back. O(N log N), no
 * iterations, no tolerance, exact to round-off.
 *
 * THE EIGENVALUES MUST BE THE DISCRETE ONES. The continuous Laplacian has
 * eigenvalue −|k|²; the 5-point stencil has
 *
 *     λ = −(4/dx²)sin²(π·kx/nx) − (4/dy²)sin²(π·ky/ny)
 *
 * which agrees with −|k|² only for small k and is off by 20% near the grid
 * scale. Dividing by −|k|² instead would leave a residual divergence that is
 * worst exactly at the high wavenumbers the turbulence study is trying to
 * measure -- a subtle error that would show up as a spurious bend in the tail
 * of the energy spectrum and be blamed on the physics.
 *
 * WHY THIS EXISTS ALONGSIDE CG. The turbulence runs need thousands of steps at
 * 512², where CG's per-step cost made the study impractical (171 s for 1400
 * steps at 128² alone). This is both faster and more accurate. CG is retained
 * because it is the method that generalises to non-periodic walls, where the
 * FFT approach does not apply -- and the two agreeing is an independent check
 * on both.
 */
export function solvePoissonFFT(g, rhs, p, fftMod) {
  const { nx, ny, dx, dy } = g;
  const { fft2, ifft2 } = fftMod;
  p = p || g.p();

  const [re, im] = fft2(rhs, nx, ny);

  for (let j = 0; j < ny; j++) {
    const sy = Math.sin(Math.PI * j / ny);
    const ly = -4 * sy * sy / (dy * dy);
    for (let i = 0; i < nx; i++) {
      const sx = Math.sin(Math.PI * i / nx);
      const lx = -4 * sx * sx / (dx * dx);
      const lam = lx + ly;
      const k = j * nx + i;
      if (i === 0 && j === 0) {
        // The constant mode: the null space. Pressure is defined only up to a
        // constant, so this is set to zero rather than divided by zero.
        re[k] = 0; im[k] = 0;
      } else {
        re[k] /= lam; im[k] /= lam;
      }
    }
  }

  const [pr] = ifft2(re, im, nx, ny);
  p.set(pr);
  return { iterations: 1, residual: 0, converged: true, exact: true, history: [] };
}

/** Relative L2 residual ‖∇²p − rhs‖ / ‖rhs‖. */
export function poissonResidual(g, p, rhs) {
  const Ap = laplacian(g, p);
  let num = 0, den = 0;
  for (let i = 0; i < Ap.length; i++) {
    const d = Ap[i] - rhs[i];
    num += d * d;
    den += rhs[i] * rhs[i];
  }
  return Math.sqrt(num) / (Math.sqrt(den) || 1);
}

/**
 * Make a MAC velocity field divergence-free, in place.
 *
 * The scale factor: the Poisson equation solved is ∇²p = ∇·w, so p here carries
 * an implicit factor of dt relative to physical pressure. Since the same p is
 * immediately subtracted as ∇p, the dt cancels and never needs to appear. Folding
 * dt in on one side but not the other is a classic way to get a projection that
 * works at one timestep and mysteriously fails when dt changes.
 */
export function project(g, u, v, opts = {}) {
  const div = divergence(g, u, v);
  const p = opts.p || g.p();
  if (!opts.p) p.fill(0);         // warm-starting is opt-in, see the solver loop
  const info = solvePoissonCG(g, div, p, opts);
  subtractGradient(g, u, v, p, 1);
  return { ...info, p };
}
