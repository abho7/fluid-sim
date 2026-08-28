/**
 * Phase 2 gate: mutation tests on the validators themselves.
 *
 * Every headline number in this project comes from a measurement routine, and a
 * measurement routine that has never been observed to react is not evidence of
 * anything. Two of the sweeps in this project genuinely were broken and
 * reported everything as fine:
 *
 *   - the explicit-diffusion sweep called the scheme stable at nu*dt/h^2 = 0.6,
 *     more than twice the theoretical bound, because it seeded a smooth field
 *     containing none of the mode that goes unstable
 *   - the confinement sweep reported "no blow-up" while its own data showed the
 *     energy growing 4667x
 *
 * Both looked like passing validations. So each validator here is fed a
 * deliberately degraded solver and required to notice, and fed a good one and
 * required not to complain.
 */

import test from "node:test";
import assert from "node:assert/strict";

import { Grid } from "../src/core/grid.js";
import { taylorGreen, translatingGaussian, relL2 } from "../src/core/analytic.js";
import { FluidSolver, diffuse } from "../src/cpu/solver.js";
import { advectScalarSL, advectScalarMacCormack } from "../src/cpu/advect.js";
import {
  advectionDiffusion, taylorGreenStudy, stabilityStudy,
} from "../validate/studies.js";

// The stability study takes ~11s and four tests below interrogate different
// parts of the same result. Running it once and sharing it keeps the suite
// usable; each test still asserts independently.
let _stability = null;
const stability = () => (_stability ??= stabilityStudy({ n: 32, nu: 0.02 }));

// =============================================== numerical viscosity detection

test("the numerical-viscosity fit separates a diffusive scheme from an accurate one", () => {
  // The headline measurement. Semi-Lagrangian is known to be far more
  // dissipative than MacCormack, so the fit must rank them that way. If it did
  // not, the number would be measuring something other than dissipation.
  const r = taylorGreenStudy({ n: 64, nu: 0.02, dt: 0.005, T: 1.0 });
  const sl = r.schemes["semi-lagrangian"], mc = r.schemes["maccormack"];

  assert.ok(sl.nuNumerical > mc.nuNumerical * 5,
    `expected SL to be much more dissipative; SL ${sl.nuNumerical}, MC ${mc.nuNumerical}`);
  // Both must be POSITIVE: a numerical scheme of this type removes energy, it
  // does not add it. A negative value would mean the fit or the sign is wrong.
  assert.ok(sl.nuNumerical > 0 && mc.nuNumerical > 0,
    `numerical viscosity should be positive, got SL ${sl.nuNumerical}, MC ${mc.nuNumerical}`);
  // The exponential fit must actually be exponential, or nu_eff is meaningless.
  assert.ok(sl.decayFitR2 > 0.99 && mc.decayFitR2 > 0.99,
    `decay fit quality too poor to quote a rate: ${sl.decayFitR2}, ${mc.decayFitR2}`);
});

test("the numerical-viscosity fit reacts to extra dissipation injected on purpose", () => {
  // Directly mutational: run the solver, then damp the velocity a little every
  // step, and require the measured effective viscosity to rise. This is the
  // check that the fit is measuring energy loss rather than reporting a
  // constant that happens to look plausible.
  const nu = 0.02, dt = 0.005, T = 1.0, n = 64;   // power of two: the fft projection requires it

  function measureRate(damp) {
    const s = new FluidSolver({ n, nu, validation: true, advection: "maccormack", projection: "fft" });
    s.init((x, y) => taylorGreen.u(x, y, 0, nu), (x, y) => taylorGreen.v(x, y, 0, nu));
    const ts = [], les = [];
    for (let k = 0; k < Math.round(T / dt); k++) {
      s.step(dt, { advectDye: false });
      if (damp !== 1) {
        for (let i = 0; i < s.u.length; i++) { s.u[i] *= damp; s.v[i] *= damp; }
      }
      if (k % 10 === 0) { ts.push(s.t); les.push(Math.log(s.kineticEnergy())); }
    }
    const m = ts.length;
    const mt = ts.reduce((a, b) => a + b, 0) / m, ml = les.reduce((a, b) => a + b, 0) / m;
    let stl = 0, stt = 0;
    for (let i = 0; i < m; i++) { stl += (ts[i] - mt) * (les[i] - ml); stt += (ts[i] - mt) ** 2; }
    return -stl / stt;
  }

  const clean = measureRate(1);
  const damped = measureRate(0.999);      // 0.1% energy removed per step
  assert.ok(damped > clean * 1.2,
    `injected dissipation was not detected: clean rate ${clean}, damped ${damped}`);
});

// ============================================= advection diffusion detection

test("the peak-retention measure distinguishes the two advection schemes", () => {
  const r = advectionDiffusion({ n: 64, dt: 0.02, steps: 50 });
  const sl = r.schemes["semi-lagrangian"], mc = r.schemes["maccormack"];
  assert.ok(mc.peakRetained > sl.peakRetained + 0.1,
    `peak retention did not separate the schemes: SL ${sl.peakRetained}, MC ${mc.peakRetained}`);
  assert.ok(sl.numericalDiffusion > mc.numericalDiffusion,
    `diffusion coefficient ranked the schemes backwards`);
  // Both must be positive: numerical diffusion spreads a blob, it does not
  // sharpen one. A negative D would indicate the recovery formula is inverted.
  assert.ok(mc.numericalDiffusion > 0, `negative numerical diffusion ${mc.numericalDiffusion}`);
});

test("peak retention reads ~1 for a scheme that is not asked to do anything", () => {
  // Zero velocity means zero advection, so the blob must not move or spread.
  // A measure that reported decay here would be charging the scheme for the
  // measurement's own error.
  const g = new Grid(64, 64);
  const opts = { sigma: 0.4, ax: 0, ay: 0 };
  const u = g.u(), v = g.v();                      // both zero
  let q = translatingGaussian.init(g, g.p(), opts);
  const q0 = Float64Array.from(q);
  let tmp = g.p();
  const scratch = { fwd: g.p(), back: g.p() };
  for (let k = 0; k < 30; k++) {
    advectScalarSL(g, u, v, q, tmp, 0.02, scratch);
    [q, tmp] = [tmp, q];
  }
  assert.ok(relL2(q, q0) < 1e-12, `still flow advected the blob by ${relL2(q, q0)}`);
});

// ============================================== stability detection

test("the explicit-diffusion sweep locates the theoretical boundary", () => {
  // PHASE 2 GATE, and a regression on a validator that was genuinely broken.
  // The theory is exact: the checkerboard mode is amplified by |1 - 8d| per
  // step, so d = 0.25 is the boundary. The isolated sweep must reproduce it.
  const r = stability();
  const b = r.explicitDiffusion.measuredBoundary;

  assert.equal(b.isolated.lastStable, 0.25,
    `operator reported stable up to ${b.isolated.lastStable}, theory says 0.25`);
  assert.equal(b.isolated.firstUnstable, 0.26,
    `operator reported unstable from ${b.isolated.firstUnstable}`);

  // Amplification must match |1 - 8d| pointwise, not merely bracket the bound.
  for (const p of r.explicitDiffusion.isolated) {
    assert.ok(Math.abs(p.measuredAmplification - p.predictedAmplification) < 1e-6,
      `d=${p.diffusionNumber}: measured ${p.measuredAmplification}, ` +
      `predicted ${p.predictedAmplification}`);
  }

  // And the sweep must find instability inside the full solver too -- the
  // version of this study that reported "stable everywhere" is the reason this
  // assertion exists.
  assert.ok(b.inSolver.firstUnstable !== null && b.inSolver.firstUnstable <= 0.30,
    `the in-solver sweep failed to detect instability (firstUnstable=${b.inSolver.firstUnstable})`);
});

test("semi-Lagrangian advection is confirmed stable well past CFL 1", () => {
  // Not a bug hunt: the point of Stam's method is unconditional stability, and
  // the report claims it. This is the evidence, and it also guards against a
  // future change that quietly makes the advection conditionally stable.
  const r = stability();
  const beyondOne = r.advectiveCFL.points.filter(p => p.cfl > 1);
  assert.ok(beyondOne.length >= 2, "the sweep did not reach CFL > 1");
  for (const p of beyondOne) {
    assert.equal(p.blewUp, false, `blew up at CFL ${p.cfl}, which should be stable`);
    assert.ok(Number.isFinite(p.relL2), `non-finite error at CFL ${p.cfl}`);
  }
});

test("the confinement sweep detects that confinement adds energy", () => {
  // The honesty check. Vorticity confinement is a fabricated energy source, and
  // the sweep exists to show that in data. At eps=0 the flow must LOSE energy
  // (physics); at some small eps it must GAIN it.
  const r = stability();
  const zero = r.confinement.points.find(p => p.epsilon === 0);
  assert.ok(zero.keRatio < 1, `with no confinement the energy should decay, got ${zero.keRatio}`);
  assert.ok(r.confinement.smallestEpsilonThatAddsEnergy !== null,
    "the sweep never found confinement adding energy, which contradicts what it is");
  assert.ok(r.confinement.smallestEpsilonThatAddsEnergy <= 5,
    `confinement only began adding energy at eps=${r.confinement.smallestEpsilonThatAddsEnergy}`);
});

test("the projection sweep shows divergence falling with iteration count", () => {
  // The "not a toy approximation" evidence. Under-converging the pressure solve
  // must leave measurable divergence, and converging it must remove it.
  const r = stability();
  const pts = r.projectionIterations.points;
  const first = pts[0], last = pts[pts.length - 1];
  assert.ok(first.maxDivergence > 1e-3,
    `a single iteration should leave obvious divergence, got ${first.maxDivergence}`);
  assert.ok(last.maxDivergence < 1e-12,
    `a converged solve should reach machine precision, got ${last.maxDivergence}`);
  // Monotone in the right direction.
  for (let i = 1; i < pts.length; i++) {
    assert.ok(pts[i].maxDivergence <= pts[i - 1].maxDivergence * 1.5,
      `divergence rose between ${pts[i - 1].maxIter} and ${pts[i].maxIter} iterations`);
  }
});

// ================================================== the honesty constraint

test("no validation study can silently enable vorticity confinement", () => {
  // The solver enforces this, but the enforcement itself needs a test: if the
  // guard were removed, every study would keep running and quietly start
  // measuring the confinement parameter instead of the fluid.
  assert.throws(() => new FluidSolver({ n: 16, validation: true, confinement: 0.001 }),
    /artificial energy source/);
});

test("the FFT projection and conjugate gradient agree to machine precision", () => {
  // Two completely independent solvers -- one direct in Fourier space, one
  // iterative in physical space -- landing on the same answer is strong
  // evidence for both. It is also what licenses using the fast one for the
  // long turbulence runs.
  const nu = 0.02, dt = 0.01;
  const mk = (proj) => {
    const s = new FluidSolver({ n: 32, nu, validation: true, projection: proj, projTol: 1e-14, projMaxIter: 5000 });
    s.init((x, y) => taylorGreen.u(x, y, 0, nu) + 0.1 * Math.sin(3 * x + y),
           (x, y) => taylorGreen.v(x, y, 0, nu) + 0.1 * Math.cos(x - 2 * y));
    for (let k = 0; k < 5; k++) s.step(dt, { advectDye: false });
    return s;
  };
  const a = mk("cg"), b = mk("fft");
  let m = 0, scale = 0;
  for (let i = 0; i < a.u.length; i++) {
    m = Math.max(m, Math.abs(a.u[i] - b.u[i]), Math.abs(a.v[i] - b.v[i]));
    scale = Math.max(scale, Math.abs(b.u[i]), Math.abs(b.v[i]));
  }
  assert.ok(m / scale < 1e-9,
    `CG and FFT projections disagree by ${m / scale} relative`);
});
