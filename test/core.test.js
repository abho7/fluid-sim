/**
 * Phase 0 gate: the primitives everything else is measured with.
 *
 * These tests are load-bearing in an unusual way. Most tests protect a feature;
 * these protect the MEASUREMENTS. A wrong "exact" solution or a mis-normalised
 * FFT would not break the simulation -- it would silently make a broken solver
 * look accurate, or a correct one look broken, and every number on the report
 * downstream would inherit the error.
 *
 * So the analytic solutions are checked against the PDE they claim to solve,
 * not against themselves, and the FFT is checked against a naive DFT
 * transcribed from the definition.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  Grid, wrap, bilerp, sampleU, sampleV, sampleP,
  divergence, subtractGradient, vorticityCenter, velocityAtCenters,
  kineticEnergy, enstrophy, maxDivergence, cflNumber,
} from "../src/core/grid.js";
import {
  taylorGreen, translatingGaussian, relL2, absL2, velocityErrorTG,
} from "../src/core/analytic.js";
import {
  fft, ifft, dftNaive, fft2, freqIndex, isPow2,
  energySpectrum, parsevalCheck, fitSlope, convergenceOrder,
} from "../src/core/fft.js";

const TWO_PI = 2 * Math.PI;

// ==================================================================== grid

test("wrap handles negative and over-range indices", () => {
  assert.equal(wrap(0, 8), 0);
  assert.equal(wrap(7, 8), 7);
  assert.equal(wrap(8, 8), 0);
  assert.equal(wrap(-1, 8), 7);
  assert.equal(wrap(-9, 8), 7);
  assert.equal(wrap(17, 8), 1);
});

test("bilerp reproduces node values exactly and interpolates linearly", () => {
  const nx = 4, ny = 4;
  const a = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) a[j * nx + i] = i;

  // On a node, exact.
  assert.equal(bilerp(a, nx, ny, 1, 1), 1);
  assert.equal(bilerp(a, nx, ny, 2, 0), 2);
  // Halfway, exactly the mean.
  assert.ok(Math.abs(bilerp(a, nx, ny, 1.5, 0) - 1.5) < 1e-15);
  assert.ok(Math.abs(bilerp(a, nx, ny, 1.25, 0) - 1.25) < 1e-15);
});

test("bilerp wraps periodically", () => {
  const nx = 4, ny = 4;
  const a = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) a[j * nx + i] = i;
  // Sampling at 4.0 must equal sampling at 0.0.
  assert.equal(bilerp(a, nx, ny, 4, 0), bilerp(a, nx, ny, 0, 0));
  assert.equal(bilerp(a, nx, ny, -1, 0), bilerp(a, nx, ny, 3, 0));
});

test("MAC samplers round-trip their own sample positions", () => {
  // Sampling a field at the exact location a value is stored must return that
  // value. This is the test that catches a half-cell offset -- the classic MAC
  // bug, which produces a plausible-looking but subtly drifting simulation.
  const g = new Grid(16, 16);
  const u = g.fillU(g.u(), (x, y) => Math.sin(x) * Math.cos(y));
  const v = g.fillV(g.v(), (x, y) => Math.cos(x) * Math.sin(y));
  const p = g.fillP(g.p(), (x, y) => Math.sin(x + y));

  for (const [i, j] of [[0, 0], [3, 5], [15, 15], [8, 2]]) {
    assert.ok(Math.abs(sampleU(g, u, g.uX(i), g.uY(j)) - u[g.idxU(i, j)]) < 1e-12,
      `u sampler off at (${i},${j})`);
    assert.ok(Math.abs(sampleV(g, v, g.vX(i), g.vY(j)) - v[g.idxV(i, j)]) < 1e-12,
      `v sampler off at (${i},${j})`);
    assert.ok(Math.abs(sampleP(g, p, g.pX(i), g.pY(j)) - p[g.idxP(i, j)]) < 1e-12,
      `p sampler off at (${i},${j})`);
  }
});

test("divergence and gradient are exact negative adjoints", () => {
  // <div(w), q> == -<w, grad(q)> for all w, q on a periodic domain.
  //
  // This identity is what makes the pressure projection an ORTHOGONAL
  // projection. If it fails, the projection removes some of the rotational
  // part of the flow along with the divergent part -- the simulation loses
  // energy for a reason that looks exactly like physical viscosity and would
  // be misread as such by the numerical-viscosity measurement.
  const g = new Grid(16, 16);
  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff - 0.5;
  };

  const w_u = g.u(), w_v = g.v(), q = g.p();
  for (let k = 0; k < w_u.length; k++) w_u[k] = rnd();
  for (let k = 0; k < w_v.length; k++) w_v[k] = rnd();
  for (let k = 0; k < q.length; k++) q[k] = rnd();

  // <div(w), q>
  const d = divergence(g, w_u, w_v);
  let lhs = 0;
  for (let k = 0; k < d.length; k++) lhs += d[k] * q[k];

  // <w, grad(q)>, obtained by subtracting the gradient from a zero field.
  const gu = g.u(), gv = g.v();
  subtractGradient(g, gu, gv, q, -1); // gu = +grad_x(q), gv = +grad_y(q)
  let rhs = 0;
  for (let k = 0; k < gu.length; k++) rhs += w_u[k] * gu[k];
  for (let k = 0; k < gv.length; k++) rhs += w_v[k] * gv[k];

  const scale = Math.max(Math.abs(lhs), Math.abs(rhs), 1e-30);
  assert.ok(Math.abs(lhs + rhs) / scale < 1e-12,
    `adjointness violated: <div w, q> = ${lhs}, <w, grad q> = ${rhs}`);
});

test("a constant velocity field has zero divergence", () => {
  const g = new Grid(32, 32);
  const u = g.u().fill(1.3), v = g.v().fill(-0.7);
  assert.ok(maxDivergence(g, u, v) < 1e-14);
});

test("cfl number matches its definition", () => {
  const g = new Grid(64, 64);           // dx = 2π/64
  const u = g.u().fill(2), v = g.v().fill(0);
  const dt = 0.01;
  const expected = dt * 2 / g.dx;
  assert.ok(Math.abs(cflNumber(g, u, v, dt) - expected) < 1e-12);
});

// =============================================================== analytic

test("Taylor-Green is divergence-free on the MAC grid to machine precision", () => {
  // PHASE 0 GATE. If the exact solution is not discretely divergence-free on
  // the grid we discretise it onto, then the solver's very first projection
  // step will "correct" the initial condition away from truth, and every error
  // measured afterwards is against a field the solver was never given.
  for (const n of [16, 32, 64]) {
    const g = new Grid(n, n);
    const [u, v] = taylorGreen.init(g, g.u(), g.v(), 0, 0.01);
    const md = maxDivergence(g, u, v);
    assert.ok(md < 1e-14, `N=${n}: max|div| = ${md}, expected < 1e-14`);
  }
});

test("Taylor-Green satisfies its own claimed decay rate", () => {
  const nu = 0.05;
  const e0 = taylorGreen.kineticEnergy(0, nu);
  const e1 = taylorGreen.kineticEnergy(1, nu);
  const rate = -Math.log(e1 / e0) / 1;
  assert.ok(Math.abs(rate - taylorGreen.energyDecayRate(nu)) < 1e-12,
    `decay rate ${rate} != 4ν = ${4 * nu}`);
});

test("Taylor-Green's analytic kinetic energy matches the grid computation", () => {
  // Ties the closed form ¼·e^(-4νt) to what the grid code actually measures.
  // These are computed by completely different routes, so agreement is real
  // evidence and not a tautology.
  const nu = 0.03, t = 0.7;
  const g = new Grid(128, 128);
  const [u, v] = taylorGreen.init(g, g.u(), g.v(), t, nu);
  const measured = kineticEnergy(g, u, v);
  const exact = taylorGreen.kineticEnergy(t, nu);
  // Centre-interpolation of a smooth field is 2nd order, so a small gap is
  // expected and shrinks as N grows -- checked in the next test.
  assert.ok(Math.abs(measured - exact) / exact < 2e-3,
    `measured ${measured} vs exact ${exact}`);
});

test("the kinetic-energy discretisation error converges at 2nd order", () => {
  const nu = 0.0, t = 0;
  const errs = [], hs = [];
  for (const n of [32, 64, 128]) {
    const g = new Grid(n, n);
    const [u, v] = taylorGreen.init(g, g.u(), g.v(), t, nu);
    errs.push(Math.abs(kineticEnergy(g, u, v) - taylorGreen.kineticEnergy(t, nu)));
    hs.push(g.h);
  }
  const { order } = convergenceOrder(hs, errs);
  assert.ok(order > 1.8 && order < 2.2, `expected ~2nd order, got ${order}`);
});

test("the discrete curl converges to Taylor-Green's analytic vorticity at 2nd order", () => {
  // Checks the closed-form vorticity against the discrete curl of the closed-
  // form velocity. Two independent derivations agreeing is what makes the
  // vorticity plot on the demo trustworthy.
  //
  // This asserts the CONVERGENCE ORDER, not an error threshold. The first
  // version of this test asserted relL2 < 1e-4 at N=256 and failed at
  // 1.76e-4 -- but the operator was correct and the threshold was a guess.
  // A fixed tolerance on a discretisation error is a number pulled out of the
  // air: too tight and it fails on correct code, too loose and it passes on a
  // first-order operator that has silently lost an order of accuracy. The
  // order itself is the property being claimed, so it is the property tested.
  const nu = 0.02, t = 0.4;
  const hs = [], errs = [];
  for (const n of [32, 64, 128, 256]) {
    const g = new Grid(n, n);
    const [u, v] = taylorGreen.init(g, g.u(), g.v(), t, nu);
    const w = vorticityCenter(g, u, v);
    const exact = g.fillP(g.p(), (x, y) => taylorGreen.vorticity(x, y, t, nu));
    hs.push(g.h);
    errs.push(relL2(w, exact));
  }
  const { order, r2 } = convergenceOrder(hs, errs);
  assert.ok(order > 1.9 && order < 2.1, `expected 2nd order, measured ${order}`);
  assert.ok(r2 > 0.999, `convergence fit was not clean, r2 = ${r2}`);
});

test("velocityErrorTG is zero on the exact solution and grows off it", () => {
  // A validator that cannot fail proves nothing. This checks both directions:
  // it reads zero on truth, and it reads large when the field is wrong.
  const nu = 0.02, t = 0.3;
  const g = new Grid(32, 32);
  const [u, v] = taylorGreen.init(g, g.u(), g.v(), t, nu);
  assert.ok(velocityErrorTG(g, u, v, t, nu) < 1e-14, "should be exact on truth");

  // Perturb by 10% and confirm the metric notices.
  for (let k = 0; k < u.length; k++) u[k] *= 1.1;
  const err = velocityErrorTG(g, u, v, t, nu);
  assert.ok(err > 0.03, `perturbed field reported error ${err}, too small`);
});

test("translating Gaussian has the exact solution it claims", () => {
  // The blob at time t must equal the blob at time 0 shifted by (a·t, b·t).
  const opts = { sigma: 0.4, ax: 1.0, ay: 0.6, cx: Math.PI, cy: Math.PI };
  const t = 1.3;
  for (const [x, y] of [[1.0, 2.0], [3.5, 0.2], [6.0, 5.5]]) {
    const later = translatingGaussian.scalar(x, y, t, opts);
    const shifted = translatingGaussian.scalar(x - opts.ax * t, y - opts.ay * t, 0, opts);
    assert.ok(Math.abs(later - shifted) < 1e-12,
      `at (${x},${y}): ${later} vs ${shifted}`);
  }
});

test("translating Gaussian is periodic", () => {
  const opts = { sigma: 0.4 };
  const a = translatingGaussian.scalar(0.7, 1.1, 0, opts);
  const b = translatingGaussian.scalar(0.7 + TWO_PI, 1.1, 0, opts);
  const c = translatingGaussian.scalar(0.7, 1.1 + TWO_PI, 0, opts);
  assert.ok(Math.abs(a - b) < 1e-12 && Math.abs(a - c) < 1e-12);
});

// ==================================================================== fft

test("isPow2", () => {
  for (const n of [1, 2, 4, 8, 1024]) assert.ok(isPow2(n), `${n}`);
  for (const n of [0, 3, 6, 12, 1000, -4, 2.5]) assert.ok(!isPow2(n), `${n}`);
});

test("fft matches a naive DFT", () => {
  // PHASE 0 GATE. The naive DFT is transcribed straight from the definition,
  // so agreement means the fast version implements the transform it claims.
  const n = 64;
  const re = new Float64Array(n);
  let seed = 7;
  for (let i = 0; i < n; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    re[i] = seed / 0x7fffffff - 0.5;
  }
  const [fr, fi] = fft(re, null);
  const [dr, di] = dftNaive(re, null, -1);
  let maxErr = 0;
  for (let i = 0; i < n; i++) {
    maxErr = Math.max(maxErr, Math.abs(fr[i] - dr[i]), Math.abs(fi[i] - di[i]));
  }
  assert.ok(maxErr < 1e-12, `fft vs naive DFT max error ${maxErr}`);
});

test("ifft(fft(x)) round-trips", () => {
  const n = 128;
  const re = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = Math.sin(3 * i) + 0.4 * Math.cos(11 * i);
  const [fr, fi] = fft(re, null);
  const [br, bi] = ifft(fr, fi);
  let maxErr = 0;
  for (let i = 0; i < n; i++) {
    maxErr = Math.max(maxErr, Math.abs(br[i] - re[i]), Math.abs(bi[i]));
  }
  assert.ok(maxErr < 1e-12, `round-trip error ${maxErr}`);
});

test("fft rejects non-power-of-two lengths rather than returning nonsense", () => {
  assert.throws(() => fft(new Float64Array(50), null), /power of two/);
});

test("freqIndex maps the upper half to negative wavenumbers", () => {
  const n = 8;
  assert.deepEqual([0, 1, 2, 3, 4, 5, 6, 7].map(i => freqIndex(i, n)),
    [0, 1, 2, 3, 4, -3, -2, -1]);
});

test("fft2 of a single cosine mode puts energy in exactly that mode", () => {
  // A field cos(k₀x) must transform to two nonzero bins, at +k₀ and -k₀, and
  // nothing anywhere else. This catches transposed loops and row/column mixups
  // that a magnitude-only check would miss.
  const nx = 32, ny = 32, k0 = 5;
  const f = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) f[j * nx + i] = Math.cos(2 * Math.PI * k0 * i / nx);
  }
  const [re, im] = fft2(f, nx, ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const mag = Math.hypot(re[j * nx + i], im[j * nx + i]) / (nx * ny);
      const expected = (j === 0 && (i === k0 || i === nx - k0)) ? 0.5 : 0;
      assert.ok(Math.abs(mag - expected) < 1e-12,
        `bin (${i},${j}) had ${mag}, expected ${expected}`);
    }
  }
});

test("energy spectrum satisfies Parseval", () => {
  // PHASE 0 GATE, and the self-check the spectrum code relies on permanently.
  // Nearly every way of getting a spectrum wrong is a normalisation error, and
  // a normalisation error breaks Parseval by a factor of N, N² or 2.
  const g = new Grid(64, 64);
  const [u, v] = taylorGreen.init(g, g.u(), g.v(), 0, 0.01);
  const [cu, cv] = velocityAtCenters(g, u, v);
  const { spectral, physical, relDiff } = parsevalCheck(cu, cv, 64, 64);
  assert.ok(relDiff < 1e-12,
    `Parseval violated: spectral ${spectral} vs physical ${physical} (rel ${relDiff})`);
});

test("Parseval holds for a random field, not just a smooth one", () => {
  // Taylor-Green lives in a couple of modes, so it would satisfy a spectrum
  // routine that got the shell binning wrong. White noise fills every shell.
  const nx = 32, ny = 32;
  const cu = new Float64Array(nx * ny), cv = new Float64Array(nx * ny);
  let seed = 99;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff - 0.5;
  };
  for (let i = 0; i < cu.length; i++) { cu[i] = rnd(); cv[i] = rnd(); }
  const { relDiff } = parsevalCheck(cu, cv, nx, ny);
  // Corner modes beyond kmax are deliberately dropped, so the shortfall here is
  // real and bounded, not a bug. It is documented in energySpectrum().
  assert.ok(relDiff < 0.25, `unexpectedly large Parseval gap ${relDiff}`);
});

test("energy spectrum of Taylor-Green concentrates at k=1", () => {
  // TG is a single wavenumber-1 mode in each direction, i.e. |k| = √2 ≈ 1.41,
  // which rounds into shell 1. Essentially all the energy must land there.
  const g = new Grid(64, 64);
  const [u, v] = taylorGreen.init(g, g.u(), g.v(), 0, 0.01);
  const [cu, cv] = velocityAtCenters(g, u, v);
  const { E } = energySpectrum(cu, cv, 64, 64);
  let total = 0;
  for (let i = 0; i < E.length; i++) total += E[i];
  assert.ok(E[1] / total > 0.999, `shell 1 holds ${E[1] / total} of the energy`);
});

test("fitSlope recovers a known power law", () => {
  // A validator that has never been observed to succeed on a known input is
  // not a validator. Build an exact k^(-5/3) and confirm the fit returns it.
  const n = 64;
  const k = new Float64Array(n), E = new Float64Array(n);
  for (let i = 0; i < n; i++) { k[i] = i; E[i] = i > 0 ? 2.5 * Math.pow(i, -5 / 3) : 0; }
  const fit = fitSlope(k, E, 4, 40);
  assert.ok(Math.abs(fit.slope + 5 / 3) < 1e-9, `slope ${fit.slope}`);
  assert.ok(fit.r2 > 0.9999, `r2 ${fit.r2}`);
});

test("fitSlope reports a poor r2 when the data is not a power law", () => {
  // The other direction: the fit must be able to say "this is not a power law"
  // rather than silently returning a number that gets quoted as a match.
  const n = 64;
  const k = new Float64Array(n), E = new Float64Array(n);
  for (let i = 0; i < n; i++) { k[i] = i; E[i] = i > 0 ? Math.exp(-i / 8) : 0; }
  const fit = fitSlope(k, E, 4, 40);
  assert.ok(fit.r2 < 0.95, `exponential decay fitted a power law with r2 ${fit.r2}`);
});

test("fitSlope returns NaN rather than a fabricated slope on too few points", () => {
  const k = Float64Array.from([1, 2, 3]);
  const E = Float64Array.from([1, 0.5, 0.25]);
  const fit = fitSlope(k, E, 10, 20);   // window contains nothing
  assert.ok(Number.isNaN(fit.slope));
  assert.equal(fit.n, 0);
});

test("convergenceOrder recovers a known order and exposes a flattening tail", () => {
  const hs = [1 / 8, 1 / 16, 1 / 32, 1 / 64];
  const second = hs.map(h => 3 * h * h);
  assert.ok(Math.abs(convergenceOrder(hs, second).order - 2) < 1e-9);

  // Errors that stall at a floor must show it in the pairwise orders rather
  // than being averaged into a single respectable-looking number.
  const stalled = [3 / 64, 3 / 256, 1e-9, 1e-9];
  const { pairwise } = convergenceOrder(hs, stalled);
  assert.ok(Math.abs(pairwise[pairwise.length - 1]) < 0.1,
    `a stalled pair should report order ~0, got ${pairwise[pairwise.length - 1]}`);
});
