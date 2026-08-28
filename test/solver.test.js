/**
 * Phase 1 gate: the CPU reference solver.
 *
 * The tests here fall into two groups.
 *
 * PROPERTIES the solver must satisfy exactly or to solver tolerance:
 * incompressibility after projection, exactness of the projection on fields
 * that are already divergence-free, and preservation of flows that are
 * stationary solutions. These can be asserted tightly because they are not
 * discretisation-limited.
 *
 * CONVERGENCE ORDERS, which are the honest way to test a discretisation. A
 * fixed error threshold on a scheme is a guess; the order of accuracy is the
 * property actually being claimed, and asserting it catches a scheme that has
 * silently dropped an order (the usual symptom of a subtle indexing bug that
 * still produces plausible-looking fluid).
 */

import test from "node:test";
import assert from "node:assert/strict";

import { Grid, divergence, maxDivergence, kineticEnergy, velocityAtCenters } from "../src/core/grid.js";
import { taylorGreen, translatingGaussian, relL2, velocityErrorTG } from "../src/core/analytic.js";
import { convergenceOrder } from "../src/core/fft.js";
import { laplacian, solvePoissonCG, solvePoissonJacobi, poissonResidual, project } from "../src/cpu/projection.js";
import { advectScalarSL, advectScalarMacCormack } from "../src/cpu/advect.js";
import { FluidSolver, diffuse } from "../src/cpu/solver.js";

// ============================================================== the Laplacian

test("the discrete Laplacian is symmetric", () => {
  // <Ax, y> == <x, Ay>. Symmetry is a precondition for conjugate gradient; an
  // asymmetric operator makes CG converge to the wrong answer or not at all,
  // and the failure looks like "the pressure solve is unstable".
  const g = new Grid(16, 16);
  let s = 4242;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  const x = g.p(), y = g.p();
  for (let i = 0; i < x.length; i++) { x[i] = rnd(); y[i] = rnd(); }

  const Ax = laplacian(g, x), Ay = laplacian(g, y);
  let lhs = 0, rhs = 0;
  for (let i = 0; i < x.length; i++) { lhs += Ax[i] * y[i]; rhs += x[i] * Ay[i]; }
  assert.ok(Math.abs(lhs - rhs) / Math.max(Math.abs(lhs), 1e-30) < 1e-12,
    `Laplacian not symmetric: ${lhs} vs ${rhs}`);
});

test("the Laplacian annihilates constants", () => {
  // The null space that solvePoissonCG has to handle. Confirming it exists is
  // what justifies the mean-removal code; if this failed, that code would be
  // silently corrupting the solution instead of pinning it.
  const g = new Grid(16, 16);
  const p = g.p().fill(3.7);
  const Ap = laplacian(g, p);
  let m = 0;
  for (let i = 0; i < Ap.length; i++) m = Math.max(m, Math.abs(Ap[i]));
  assert.ok(m < 1e-12, `constant field had nonzero Laplacian ${m}`);
});

test("the Laplacian converges at 2nd order on a known function", () => {
  // ∇²sin(x)sin(y) = -2 sin(x)sin(y).
  const hs = [], errs = [];
  for (const n of [16, 32, 64, 128]) {
    const g = new Grid(n, n);
    const p = g.fillP(g.p(), (x, y) => Math.sin(x) * Math.sin(y));
    const exact = g.fillP(g.p(), (x, y) => -2 * Math.sin(x) * Math.sin(y));
    hs.push(g.h);
    errs.push(relL2(laplacian(g, p), exact));
  }
  const { order } = convergenceOrder(hs, errs);
  assert.ok(order > 1.9 && order < 2.1, `Laplacian order ${order}`);
});

// ============================================================ Poisson solvers

test("conjugate gradient solves a Poisson problem to tolerance", () => {
  const g = new Grid(32, 32);
  // ∇²p = -2 sin(x)sin(y) has the exact solution p = sin(x)sin(y).
  const rhs = g.fillP(g.p(), (x, y) => -2 * Math.sin(x) * Math.sin(y));
  const p = g.p();
  const info = solvePoissonCG(g, rhs, p, { tol: 1e-12, maxIter: 500 });

  assert.ok(info.converged, `CG did not converge: residual ${info.residual}`);
  const exact = g.fillP(g.p(), (x, y) => Math.sin(x) * Math.sin(y));
  // Discretisation error dominates the solver error here; the point is that
  // the SOLVER converged, verified by the residual, and that it landed on the
  // right function to within the 2nd-order truncation error of the operator.
  assert.ok(relL2(p, exact) < 5e-3, `CG solution relL2 ${relL2(p, exact)}`);
  assert.ok(info.residual < 1e-12);
});

test("CG converges far faster than Jacobi on the same problem", () => {
  // Substantiates the choice of CG for the reference solver, with numbers
  // rather than the assertion that it is better.
  const g = new Grid(32, 32);
  const rhs = g.fillP(g.p(), (x, y) => Math.sin(3 * x) * Math.cos(2 * y));

  const cg = solvePoissonCG(g, rhs, g.p(), { tol: 1e-10, maxIter: 2000 });
  const jac = solvePoissonJacobi(g, rhs, g.p(), { iterations: 200 });

  assert.ok(cg.converged, "CG failed to converge");
  assert.ok(cg.iterations < 60, `CG took ${cg.iterations} iterations`);
  // 200 Jacobi sweeps must still be worse than CG's tolerance, which is the
  // quantitative form of "Jacobi is not good enough on its own".
  assert.ok(jac.residual > cg.residual,
    `Jacobi (${jac.residual}) beat CG (${cg.residual}), unexpected`);
});

test("Jacobi's residual decreases monotonically", () => {
  // Damped Jacobi on this operator is a contraction; a rising residual would
  // mean the iteration or the residual measure is wrong.
  const g = new Grid(32, 32);
  const rhs = g.fillP(g.p(), (x, y) => Math.sin(2 * x) * Math.sin(y));
  const { history } = solvePoissonJacobi(g, rhs, g.p(), { iterations: 40 });
  for (let i = 1; i < history.length; i++) {
    assert.ok(history[i] <= history[i - 1] + 1e-14,
      `Jacobi residual rose at iteration ${i}: ${history[i - 1]} -> ${history[i]}`);
  }
});

test("CG handles a right-hand side with a nonzero mean", () => {
  // The periodic Poisson equation is unsolvable unless ∮rhs = 0. Rather than
  // producing garbage, the solver removes the mean and solves the solvable
  // part. This test pins that, because the alternative is a silent stall.
  const g = new Grid(16, 16);
  const rhs = g.fillP(g.p(), (x, y) => Math.sin(x) * Math.sin(y) + 5.0);
  const info = solvePoissonCG(g, rhs, g.p(), { tol: 1e-11, maxIter: 500 });
  assert.ok(info.converged, `did not converge, residual ${info.residual}`);
});

// ================================================================= projection

test("projection makes an arbitrary field divergence-free", () => {
  // THE core claim of the whole solver: ∇·u = 0.
  const g = new Grid(32, 32);
  let s = 777;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  const u = g.u(), v = g.v();
  for (let i = 0; i < u.length; i++) { u[i] = rnd(); v[i] = rnd(); }

  assert.ok(maxDivergence(g, u, v) > 0.1, "test field was not divergent to begin with");
  project(g, u, v, { tol: 1e-12, maxIter: 2000 });
  const md = maxDivergence(g, u, v);
  assert.ok(md < 1e-10, `max|div| after projection = ${md}`);
});

test("projection leaves an already divergence-free field alone", () => {
  // Idempotence. If projection perturbed a solenoidal field it would be
  // removing energy every step, and that loss would be indistinguishable from
  // physical viscosity -- it would be absorbed into the numerical-viscosity
  // measurement and misattributed to the advection scheme.
  const g = new Grid(32, 32);
  const [u, v] = taylorGreen.init(g, g.u(), g.v(), 0, 0.01);
  const u0 = Float64Array.from(u), v0 = Float64Array.from(v);

  project(g, u, v, { tol: 1e-13, maxIter: 2000 });

  let m = 0;
  for (let i = 0; i < u.length; i++) {
    m = Math.max(m, Math.abs(u[i] - u0[i]), Math.abs(v[i] - v0[i]));
  }
  assert.ok(m < 1e-11, `projection moved a divergence-free field by ${m}`);
});

test("projection is idempotent", () => {
  const g = new Grid(16, 16);
  let s = 31337;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  const u = g.u(), v = g.v();
  for (let i = 0; i < u.length; i++) { u[i] = rnd(); v[i] = rnd(); }

  project(g, u, v, { tol: 1e-13, maxIter: 2000 });
  const u1 = Float64Array.from(u), v1 = Float64Array.from(v);
  project(g, u, v, { tol: 1e-13, maxIter: 2000 });

  let m = 0;
  for (let i = 0; i < u.length; i++) {
    m = Math.max(m, Math.abs(u[i] - u1[i]), Math.abs(v[i] - v1[i]));
  }
  assert.ok(m < 1e-11, `second projection changed the field by ${m}`);
});

// ================================================================== diffusion

test("implicit diffusion decays a sine mode at the analytic rate", () => {
  // For ∂q/∂t = ν∇²q, the mode sin(x)sin(y) has ∇²q = -2q, so it decays as
  // e^(-2νt). Backward Euler gives 1/(1+2ν·dt) per step, which is what an
  // implicit solve should reproduce -- this checks the solve, not the scheme's
  // time accuracy.
  const g = new Grid(64, 64);
  const nu = 0.05, dt = 0.01;
  const q = g.fillP(g.p(), (x, y) => Math.sin(x) * Math.sin(y));
  const out = g.p();
  diffuse(g, q, out, nu, dt, {});

  const expected = 1 / (1 + 2 * nu * dt);
  // Measure the amplitude by projecting onto the mode.
  let num = 0, den = 0;
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      const b = Math.sin(g.pX(i)) * Math.sin(g.pY(j));
      num += out[g.idxP(i, j)] * b;
      den += b * b;
    }
  }
  const amp = num / den;
  assert.ok(Math.abs(amp - expected) / expected < 2e-3,
    `implicit diffusion amplitude ${amp}, expected ~${expected}`);
});

test("diffusion conserves the mean of the field", () => {
  // ∇² has no zeroth moment, so diffusion cannot create or destroy the
  // integral. A drifting mean means the operator or its boundary wrapping is
  // wrong.
  const g = new Grid(32, 32);
  const q = g.fillP(g.p(), (x, y) => 2 + Math.sin(x) * Math.cos(y));
  const before = q.reduce((s, x) => s + x, 0);
  const out = g.p();
  diffuse(g, q, out, 0.1, 0.05, {});
  const after = out.reduce((s, x) => s + x, 0);
  assert.ok(Math.abs(after - before) / Math.abs(before) < 1e-9,
    `mean drifted from ${before} to ${after}`);
});

// ================================================================== advection

test("advection leaves a constant field constant", () => {
  // Any advection scheme must be exact on constants; failing this means the
  // interpolation weights do not sum to one, which shows up as the whole
  // simulation slowly gaining or losing mass.
  const g = new Grid(32, 32);
  const u = g.u().fill(0.7), v = g.v().fill(-0.4);
  const q = g.p().fill(2.5);

  for (const [name, fn] of [["SL", advectScalarSL], ["MacCormack", advectScalarMacCormack]]) {
    const out = g.p();
    fn(g, u, v, q, out, 0.05, { fwd: g.p(), back: g.p() });
    let m = 0;
    for (let i = 0; i < out.length; i++) m = Math.max(m, Math.abs(out[i] - 2.5));
    assert.ok(m < 1e-12, `${name} disturbed a constant field by ${m}`);
  }
});

test("MacCormack is measurably less diffusive than semi-Lagrangian", () => {
  // The headline comparison between the two schemes, on the pure-advection
  // test case where nothing else can be responsible for the difference.
  const g = new Grid(128, 128);
  const opts = { sigma: 0.4, ax: 1.0, ay: 0.6 };
  const [u, v] = translatingGaussian.initVelocity(g, g.u(), g.v(), opts);
  const dt = 0.02;
  const nSteps = 50;

  const results = {};
  for (const [name, fn] of [["sl", advectScalarSL], ["mc", advectScalarMacCormack]]) {
    let q = translatingGaussian.init(g, g.p(), opts);
    let tmp = g.p();
    const scratch = { fwd: g.p(), back: g.p() };
    for (let s = 0; s < nSteps; s++) {
      fn(g, u, v, q, tmp, dt, scratch);
      [q, tmp] = [tmp, q];
    }
    const exact = g.fillP(g.p(), (x, y) =>
      translatingGaussian.scalar(x, y, nSteps * dt, opts));
    results[name] = { err: relL2(q, exact), peak: Math.max(...q) };
  }

  assert.ok(results.mc.err < results.sl.err,
    `MacCormack error ${results.mc.err} was not below SL ${results.sl.err}`);
  // Peak amplitude is the clearest signature of numerical diffusion: the exact
  // solution's peak is 1, and a diffusive scheme flattens it.
  assert.ok(results.mc.peak > results.sl.peak,
    `MacCormack peak ${results.mc.peak} not above SL ${results.sl.peak}`);
});

test("the MacCormack limiter prevents new extrema", () => {
  // Unlimited MacCormack overshoots at sharp gradients, and on velocity those
  // overshoots feed back through advection and grow. A scalar that starts
  // within [0,1] must stay within [0,1] under pure advection.
  const g = new Grid(64, 64);
  const u = g.u().fill(1.0), v = g.v().fill(0.5);
  // A sharp top-hat, the hardest case for a high-order scheme.
  const q = g.fillP(g.p(), (x, y) =>
    (x > 2 && x < 4 && y > 2 && y < 4) ? 1 : 0);

  let cur = q, tmp = g.p();
  const scratch = { fwd: g.p(), back: g.p() };
  for (let s = 0; s < 40; s++) {
    advectScalarMacCormack(g, u, v, cur, tmp, 0.02, scratch);
    [cur, tmp] = [tmp, cur];
  }
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < cur.length; i++) { lo = Math.min(lo, cur[i]); hi = Math.max(hi, cur[i]); }
  assert.ok(lo > -1e-9 && hi < 1 + 1e-9,
    `limited MacCormack produced values outside [0,1]: [${lo}, ${hi}]`);
});

// ============================================================ the full solver

test("the solver keeps the velocity field divergence-free every step", () => {
  // PHASE 1 GATE.
  const s = new FluidSolver({ n: 32, nu: 0.01, validation: true, projTol: 1e-11 });
  s.init((x, y) => taylorGreen.u(x, y, 0, 0.01), (x, y) => taylorGreen.v(x, y, 0, 0.01));
  for (let i = 0; i < 20; i++) {
    s.step(0.02);
    assert.ok(s.maxDivergence() < 1e-9,
      `step ${i}: max|div| = ${s.maxDivergence()}`);
  }
});

test("the solver preserves a uniform flow exactly", () => {
  // Uniform flow is a stationary solution: no advection effect (it advects
  // itself trivially), no diffusion, no pressure. Anything that changes is a
  // bug in the machinery rather than in the physics.
  const s = new FluidSolver({ n: 32, nu: 0.05, validation: true });
  s.init(() => 1.3, () => -0.7);
  for (let i = 0; i < 10; i++) s.step(0.05);
  let m = 0;
  for (let i = 0; i < s.u.length; i++) {
    m = Math.max(m, Math.abs(s.u[i] - 1.3), Math.abs(s.v[i] + 0.7));
  }
  assert.ok(m < 1e-9, `uniform flow drifted by ${m}`);
});

test("the solver refuses vorticity confinement in validation mode", () => {
  // The honesty constraint, enforced rather than documented. A validation run
  // with confinement on would be measuring the confinement parameter.
  assert.throws(
    () => new FluidSolver({ n: 16, validation: true, confinement: 0.5 }),
    /artificial energy source/,
  );
  // ...and permits it outside validation, where the demo lives.
  assert.doesNotThrow(() => new FluidSolver({ n: 16, confinement: 0.5 }));
});

test("the solver rejects an unknown advection scheme instead of silently defaulting", () => {
  assert.throws(() => new FluidSolver({ n: 16, advection: "quicK" }), /unknown advection/);
});

test("Taylor-Green decays and stays close to the analytic solution", () => {
  // PHASE 1 GATE. Not a tight tolerance -- this is a coarse grid with a
  // first-order scheme, and the size of the gap is itself a measurement the
  // validation harness reports. What is asserted is that the solver is
  // tracking the right solution rather than drifting off it.
  const nu = 0.05, dt = 0.01, T = 0.5;
  const s = new FluidSolver({ n: 64, nu, validation: true, advection: "maccormack" });
  s.init((x, y) => taylorGreen.u(x, y, 0, nu), (x, y) => taylorGreen.v(x, y, 0, nu));

  for (let t = 0; t < T / dt; t++) s.step(dt);

  const err = velocityErrorTG(s.g, s.u, s.v, s.t, nu);
  assert.ok(err < 0.05, `Taylor-Green relative L2 error ${err} after t=${s.t}`);

  // The energy must DECAY, and by roughly the analytic factor. A solver that
  // gained energy here would be badly wrong in a way the error norm alone
  // might not reveal.
  const ke = s.kineticEnergy();
  const exactKE = taylorGreen.kineticEnergy(s.t, nu);
  assert.ok(ke < taylorGreen.kineticEnergy(0, nu), "energy did not decay");
  assert.ok(ke < exactKE * 1.02,
    `energy ${ke} exceeded the analytic ${exactKE}; the solver is adding energy`);
});

test("Taylor-Green converges at 1st order in time (operator splitting)", () => {
  // PHASE 1 GATE, and a correction to how this was first written.
  //
  // The original version held dt fixed, refined the grid, and expected the
  // error to fall. It did the opposite: 8.9e-5 -> 2.3e-4 -> 2.9e-4 for
  // N = 16, 32, 64. That looked like a broken solver and was not.
  //
  // Sweeping dt and N independently showed why. At a fixed dt the TEMPORAL
  // error dominates and is identical at every resolution, so refining the grid
  // cannot reduce it. Worse, at N=16 the spatial error happened to have the
  // opposite sign and partially cancelled the temporal error, making the
  // coarsest grid look the most accurate. Refining removed the cancellation
  // and the error rose.
  //
  // The scheme is first-order in time because the step is Lie-split
  // (advect, then diffuse, then project) and Lie splitting is O(dt). That is a
  // real property of Stam-style solvers and is rarely stated; it caps the whole
  // scheme at first order no matter how accurate the individual operators are.
  // See the limitations section of the report.
  const nu = 0.05, T = 0.25, n = 64;
  const dts = [0.02, 0.01, 0.005, 0.0025];
  const errs = [];
  for (const dt of dts) {
    const s = new FluidSolver({ n, nu, validation: true, advection: "maccormack", projTol: 1e-12 });
    s.init((x, y) => taylorGreen.u(x, y, 0, nu), (x, y) => taylorGreen.v(x, y, 0, nu));
    for (let t = 0; t < Math.round(T / dt); t++) s.step(dt, { advectDye: false });
    errs.push(velocityErrorTG(s.g, s.u, s.v, s.t, nu));
  }
  for (let i = 1; i < errs.length; i++) {
    assert.ok(errs[i] < errs[i - 1],
      `error did not fall as dt shrank: ${errs.map(e => e.toExponential(2)).join(" -> ")}`);
  }
  const { order } = convergenceOrder(dts, errs);
  assert.ok(order > 0.85 && order < 1.35,
    `expected ~1st order in time from Lie splitting, measured ${order}`);
});

test("the spatial discretisation converges once the temporal error is removed", () => {
  // The companion to the test above. To see the SPATIAL order at all, dt must
  // be small enough that the O(dt) splitting error sits below the O(h^p)
  // truncation error -- otherwise every grid reports the same temporal floor.
  //
  // Only the coarse grids are used here, because the finer the grid the
  // smaller its spatial error and the smaller dt has to be to expose it; at
  // N=128 the spatial error is already below the splitting error at
  // dt = 6e-4 and resolving it would cost tens of thousands of steps. The
  // validation harness reports the full dt x N table rather than one number,
  // which shows exactly where each grid stops being spatially limited.
  const nu = 0.05, T = 0.1, dt = 0.0005;
  const hs = [], errs = [];
  for (const n of [16, 24, 32]) {
    const s = new FluidSolver({ n, nu, validation: true, advection: "maccormack", projTol: 1e-12 });
    s.init((x, y) => taylorGreen.u(x, y, 0, nu), (x, y) => taylorGreen.v(x, y, 0, nu));
    for (let t = 0; t < Math.round(T / dt); t++) s.step(dt, { advectDye: false });
    hs.push(s.g.h);
    errs.push(velocityErrorTG(s.g, s.u, s.v, s.t, nu));
  }
  for (let i = 1; i < errs.length; i++) {
    assert.ok(errs[i] < errs[i - 1],
      `spatial error rose on refinement: ${errs.map(e => e.toExponential(2)).join(" -> ")}`);
  }
  const { order } = convergenceOrder(hs, errs);
  assert.ok(order > 1.5, `spatial convergence order ${order} is lower than expected`);
});

test("blow-up detection actually fires", () => {
  // The stability sweeps depend on this returning true when a run diverges. A
  // detector that never fires would report every configuration as stable.
  const s = new FluidSolver({ n: 16, nu: 0.01, validation: true });
  s.init(() => 1, () => 0);
  assert.equal(s.isBlownUp(), false);
  s.u[5] = NaN;
  assert.equal(s.isBlownUp(), true);
  s.u[5] = Infinity;
  assert.equal(s.isBlownUp(), true);
});
