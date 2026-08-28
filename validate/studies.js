/**
 * The five validation studies.
 *
 * Each function here runs a real simulation and returns measured numbers. None
 * of them takes an "expected" value as input or compares against one -- they
 * report what happened, and the report page states how that compares to theory.
 * Keeping the measurement and the judgement separate is what stops a study from
 * quietly becoming a test that passes.
 *
 * Every solver constructed here passes `validation: true`, which makes the
 * solver throw if vorticity confinement is requested. Confinement injects
 * energy; a decay rate or a spectrum measured with it on would be measuring the
 * confinement parameter rather than the fluid.
 */

import { FluidSolver, bandForcing, linearDrag, diffuse } from "../src/cpu/solver.js";
import { taylorGreen, translatingGaussian, relL2, velocityErrorTG } from "../src/core/analytic.js";
import { velocityAtCenters, kineticEnergy } from "../src/core/grid.js";
import { energySpectrum, fitSlope, convergenceOrder, parsevalCheck } from "../src/core/fft.js";
import { advectScalarSL, advectScalarMacCormack } from "../src/cpu/advect.js";
import { RigidDisk, applyCoupling, slipError, fluidMomentum } from "../src/cpu/solid.js";
import { Grid } from "../src/core/grid.js";

const SCHEMES = ["semi-lagrangian", "maccormack"];

// ============================================ 1. pure advection: numerical diffusion

/**
 * Advect a Gaussian blob in uniform flow and measure how much the scheme
 * smears it.
 *
 * WHY THIS AND NOT JUST TAYLOR-GREEN. In the Taylor-Green vortex the nonlinear
 * advection term is exactly cancelled by the pressure gradient, so a badly
 * diffusive advection scheme can still score well there. This test has nothing
 * else in it -- no pressure, no viscosity, no nonlinearity -- so every bit of
 * spreading is the advection scheme's own artificial diffusion.
 *
 * The peak amplitude is the cleanest signal: the exact solution's peak stays at
 * 1.0 forever, so 1 − peak is a direct readout of how much the scheme has
 * flattened the blob.
 *
 * An effective diffusion coefficient is then recovered from the width. A
 * Gaussian of width σ₀ diffusing with coefficient D for time t has
 * σ²(t) = σ₀² + 2Dt, and a 2D Gaussian's peak scales as 1/σ², so
 *     peak(t) = σ₀²/(σ₀² + 2Dt)  =>  D = σ₀²(1/peak − 1)/(2t).
 * That is a real number in real units for "artificial numerical dissipation".
 */
export function advectionDiffusion({ n = 128, dt = 0.02, steps = 100 } = {}) {
  const opts = { sigma: 0.4, ax: 1.0, ay: 0.6, cx: Math.PI, cy: Math.PI };
  const out = { n, dt, steps, T: dt * steps, sigma0: opts.sigma, schemes: {} };

  for (const scheme of SCHEMES) {
    const s = new FluidSolver({ n, nu: 0, validation: true, advection: scheme });
    const g = s.g;
    const [u, v] = translatingGaussian.initVelocity(g, g.u(), g.v(), opts);
    const fn = scheme === "maccormack" ? advectScalarMacCormack : advectScalarSL;

    let q = translatingGaussian.init(g, g.p(), opts);
    let tmp = g.p();
    const scratch = { fwd: g.p(), back: g.p() };

    const trace = [];
    for (let k = 0; k < steps; k++) {
      fn(g, u, v, q, tmp, dt, scratch);
      [q, tmp] = [tmp, q];
      if (k % 10 === 9 || k === 0) {
        const t = (k + 1) * dt;
        const exact = g.fillP(g.p(), (x, y) => translatingGaussian.scalar(x, y, t, opts));
        let peak = -Infinity;
        for (let i = 0; i < q.length; i++) peak = Math.max(peak, q[i]);
        trace.push({ step: k + 1, t, relL2: relL2(q, exact), peak });
      }
    }

    const last = trace[trace.length - 1];
    const s0sq = opts.sigma * opts.sigma;
    // Peak of the discrete initial field, not the analytic 1.0: on a finite
    // grid the sample points miss the true peak slightly, and using 1.0 would
    // charge that sampling offset to the scheme as diffusion.
    const q0 = translatingGaussian.init(g, g.p(), opts);
    let peak0 = -Infinity;
    for (let i = 0; i < q0.length; i++) peak0 = Math.max(peak0, q0[i]);
    const ratio = last.peak / peak0;
    const D = ratio > 0 && ratio < 1
      ? s0sq * (1 / ratio - 1) / (2 * last.t)
      : null;

    out.schemes[scheme] = {
      trace,
      finalRelL2: last.relL2,
      peakInitial: peak0,
      peakFinal: last.peak,
      peakRetained: ratio,
      numericalDiffusion: D,
      cfl: dt * (Math.abs(opts.ax) / g.dx + Math.abs(opts.ay) / g.dy),
    };
  }
  return out;
}

// ==================================== 2. Taylor-Green: accuracy + numerical viscosity

/**
 * Run the Taylor-Green vortex and compare against the closed-form solution.
 *
 * THE HEADLINE MEASUREMENT. Kinetic energy decays exactly as e^(-4νt). The
 * simulation decays faster, because the advection scheme's interpolation is
 * removing energy on top of the physical viscosity. Fitting the measured decay
 * gives an effective viscosity ν_eff, and
 *
 *     ν_num = ν_eff − ν
 *
 * is the artificial viscosity, in the same units as the real one. That converts
 * "the scheme is dissipative" from an adjective into a number that can be
 * compared between schemes and across resolutions.
 *
 * The fit is a straight line through log(KE) vs t, which is exact for a pure
 * exponential; r² is reported so a decay that is NOT exponential (which would
 * make ν_eff meaningless) is visible rather than hidden behind a slope.
 */
export function taylorGreenStudy({ n = 128, nu = 0.02, dt = 0.005, T = 2.0 } = {}) {
  const out = { n, nu, dt, T, exactDecayRate: taylorGreen.energyDecayRate(nu), schemes: {} };
  const steps = Math.round(T / dt);

  for (const scheme of SCHEMES) {
    const s = new FluidSolver({ n, nu, validation: true, advection: scheme, projTol: 1e-11 });
    s.init((x, y) => taylorGreen.u(x, y, 0, nu), (x, y) => taylorGreen.v(x, y, 0, nu));

    const trace = [];
    const record = () => {
      trace.push({
        t: s.t,
        relL2: velocityErrorTG(s.g, s.u, s.v, s.t, nu),
        ke: s.kineticEnergy(),
        keExact: taylorGreen.kineticEnergy(s.t, nu),
        enstrophy: s.enstrophy(),
        maxDiv: s.maxDivergence(),
        cgIterations: s.lastProjection ? s.lastProjection.iterations : 0,
      });
    };
    record();
    for (let k = 0; k < steps; k++) {
      s.step(dt, { advectDye: false });
      if ((k + 1) % Math.max(1, Math.round(steps / 40)) === 0) record();
    }

    // Fit log(KE) = log(KE0) − rate·t.
    const ts = trace.map(p => p.t), les = trace.map(p => Math.log(p.ke));
    const nPts = ts.length;
    const mt = ts.reduce((a, b) => a + b, 0) / nPts;
    const ml = les.reduce((a, b) => a + b, 0) / nPts;
    let stl = 0, stt = 0, sll = 0;
    for (let i = 0; i < nPts; i++) {
      stl += (ts[i] - mt) * (les[i] - ml);
      stt += (ts[i] - mt) ** 2;
      sll += (les[i] - ml) ** 2;
    }
    const slope = stl / stt;
    const measuredRate = -slope;
    const nuEff = measuredRate / 4;

    out.schemes[scheme] = {
      trace,
      measuredDecayRate: measuredRate,
      exactDecayRate: out.exactDecayRate,
      nuEffective: nuEff,
      nuNumerical: nuEff - nu,
      nuNumericalRatio: (nuEff - nu) / nu,
      decayFitR2: sll > 0 ? (stl * stl) / (stt * sll) : NaN,
      finalRelL2: trace[trace.length - 1].relL2,
      maxDivergence: Math.max(...trace.map(p => p.maxDiv)),
      meanCGIterations:
        trace.reduce((a, p) => a + p.cgIterations, 0) / trace.length,
    };
  }
  return out;
}

// ======================================================== 3. convergence table

/**
 * The full dt x N error table.
 *
 * Reported as a TABLE rather than a single "order of accuracy" because a single
 * number is misleading for a split scheme. Sweeping dt at fixed N gives the
 * temporal order; sweeping N at fixed dt gives nothing useful unless dt is
 * already small enough, and a naive study that holds dt fixed and refines the
 * grid can show the error RISING -- which is what happened here first and is
 * documented on the report.
 *
 * The table makes the structure visible: rows flatten onto a spatial floor as
 * dt shrinks, and the floor is what falls with resolution.
 */
export function convergenceStudy({
  nu = 0.05, T = 0.25,
  Ns = [16, 32, 64, 128],
  dts = [0.02, 0.01, 0.005, 0.0025, 0.00125],
  scheme = "maccormack",
} = {}) {
  const table = [];
  for (const n of Ns) {
    const row = { n, h: (2 * Math.PI) / n, errors: [] };
    for (const dt of dts) {
      const s = new FluidSolver({ n, nu, validation: true, advection: scheme, projTol: 1e-12 });
      s.init((x, y) => taylorGreen.u(x, y, 0, nu), (x, y) => taylorGreen.v(x, y, 0, nu));
      for (let k = 0; k < Math.round(T / dt); k++) s.step(dt, { advectDye: false });
      row.errors.push({ dt, relL2: velocityErrorTG(s.g, s.u, s.v, s.t, nu) });
    }
    // Temporal order along this row.
    const c = convergenceOrder(row.errors.map(e => e.dt), row.errors.map(e => e.relL2));
    row.temporalOrder = c.order;
    row.temporalR2 = c.r2;
    row.pairwise = c.pairwise;
    table.push(row);
  }

  // Spatial order at the smallest dt, using only the grids that have actually
  // reached their spatial floor -- a grid still limited by the temporal error
  // would drag the fitted order toward zero and misrepresent the scheme.
  const finestDtIdx = dts.length - 1;
  const spatial = table.map(r => ({ h: r.h, err: r.errors[finestDtIdx].relL2 }));
  const usable = [];
  for (let i = 0; i < spatial.length; i++) {
    if (i === 0 || spatial[i].err < spatial[i - 1].err) usable.push(spatial[i]);
    else break;   // stop at the first grid where refining stopped helping
  }
  const spatialFit = usable.length >= 2
    ? convergenceOrder(usable.map(s => s.h), usable.map(s => s.err))
    : { order: NaN, r2: NaN, pairwise: [] };

  return {
    nu, T, scheme, Ns, dts, table,
    spatialAtFinestDt: { dt: dts[finestDtIdx], points: spatial, usedPoints: usable.length,
      order: spatialFit.order, r2: spatialFit.r2 },
  };
}

// ================================================ 4. stability characterization

/**
 * Where the solver stays stable and where it blows up.
 *
 * A CORRECTION TO THE OBVIOUS FRAMING, stated because the brief asks for "the
 * CFL condition boundary". Semi-Lagrangian advection is UNCONDITIONALLY stable
 * -- tracing backward and interpolating can never produce a value outside the
 * range of the field it sampled, so it cannot amplify anything. That is the
 * entire reason Stam's method exists. There is therefore no advective CFL
 * boundary to find, and reporting one would be inventing a result.
 *
 * What is actually measured:
 *   (a) advective CFL swept well past 1 -- confirming stability persists while
 *       ACCURACY degrades, which is the real cost of a large timestep
 *   (b) explicit diffusion swept across the theoretical ν·dt/h² = 1/4 bound,
 *       where there IS a genuine stability boundary, located from data
 *   (c) vorticity confinement strength swept until the energy diverges
 */
export function stabilityStudy({ n = 64, nu = 0.02 } = {}) {
  const out = {};

  // (a) Advective CFL: does it blow up, and how much accuracy is lost?
  out.advectiveCFL = { n, nu, note: "semi-Lagrangian is unconditionally stable; this measures accuracy loss, not a stability boundary", points: [] };
  {
    const T = 0.5;
    for (const dt of [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.25]) {
      const s = new FluidSolver({ n, nu, validation: true, advection: "semi-lagrangian", projTol: 1e-10 });
      s.init((x, y) => taylorGreen.u(x, y, 0, nu), (x, y) => taylorGreen.v(x, y, 0, nu));
      const cfl = s.cfl(dt);
      const steps = Math.max(1, Math.round(T / dt));
      for (let k = 0; k < steps; k++) s.step(dt, { advectDye: false });
      out.advectiveCFL.points.push({
        dt, cfl, steps,
        blewUp: s.isBlownUp(),
        relL2: s.isBlownUp() ? null : velocityErrorTG(s.g, s.u, s.v, s.t, nu),
        keRatio: s.isBlownUp() ? null : s.kineticEnergy() / taylorGreen.kineticEnergy(s.t, nu),
      });
    }
  }

  // (b) Explicit diffusion: the one genuine stability boundary in this solver,
  // at ν·dt/h² = 1/4 in 2D.
  //
  // THE MODE HAS TO BE SEEDED. The first version of this sweep started from a
  // smooth Taylor-Green field and watched for the kinetic energy to exceed a
  // threshold. It reported the scheme as stable all the way to d = 0.6, which
  // is false. Two reasons, both instructive: a smooth initial condition
  // contains essentially none of the checkerboard mode that actually goes
  // unstable, so there was nothing to amplify; and the field was decaying
  // under viscosity, so an absolute energy threshold was never going to be
  // crossed however unstable the scheme was.
  //
  // Testing the operator in isolation showed it was correct all along --
  // amplification exactly |1 − 8d|, crossing 1 at d = 0.25 -- so the fault was
  // in the instrument, not the solver. It now seeds the grid-scale mode
  // explicitly and measures its GROWTH, which is the quantity the stability
  // analysis is actually about.
  const DIFF_NUMBERS = [0.05, 0.1, 0.15, 0.2, 0.23, 0.24, 0.25, 0.26, 0.28, 0.3, 0.4, 0.6];
  out.explicitDiffusion = {
    n, nu,
    theoreticalBound: 0.25,
    theory: "amplification factor |1 - 8d| for the checkerboard mode, d = nu*dt/h^2",
    note: "explicit diffusion is enabled only for this study; the solver uses implicit diffusion everywhere else",
    isolated: [],
    inSolver: [],
  };
  {
    const g0 = new FluidSolver({ n, nu, validation: true }).g;
    const h = g0.h;

    // (b1) The operator alone, seeded with the checkerboard. This is where the
    // textbook bound applies and where it should be reproduced exactly.
    for (const d of DIFF_NUMBERS) {
      const dt = d * h * h / nu;
      let q = g0.p();
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) q[g0.idxP(i, j)] = ((i + j) % 2 === 0) ? 1 : -1;
      }
      let tmp = g0.p();
      const amps = [];
      for (let k = 0; k < 10; k++) {
        diffuse(g0, q, tmp, nu, dt, { explicit: true });
        [q, tmp] = [tmp, q];
        let m = 0;
        for (let i = 0; i < q.length; i++) m = Math.max(m, Math.abs(q[i]));
        amps.push(m);
      }
      // Growth per step, measured from the last two amplitudes.
      const growth = amps[amps.length - 1] / (amps[amps.length - 2] || 1);
      out.explicitDiffusion.isolated.push({
        diffusionNumber: d, dt,
        predictedAmplification: Math.abs(1 - 8 * d),
        measuredAmplification: growth,
        amplitudes: amps,
        unstable: growth > 1 + 1e-9,
      });
    }

    // (b2) The same sweep inside the full solver, with a seeded perturbation.
    // Interesting because the pressure projection removes the divergent part of
    // the grid-scale mode every step, so the boundary the SOLVER exhibits need
    // not equal the boundary the operator does.
    for (const d of DIFF_NUMBERS) {
      const dt = d * h * h / nu;
      const s = new FluidSolver({
        n, nu, validation: true, advection: "semi-lagrangian",
        explicitDiffusion: true, projection: "fft",
      });
      s.init((x, y) => taylorGreen.u(x, y, 0, nu), (x, y) => taylorGreen.v(x, y, 0, nu));
      // Seed the grid-scale mode at 1e-6 -- small enough not to disturb the
      // flow, large enough to be far above round-off so growth is measurable.
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const sgn = ((i + j) % 2 === 0) ? 1 : -1;
          s.u[s.g.idxU(i, j)] += 1e-6 * sgn;
          s.v[s.g.idxV(i, j)] += 1e-6 * sgn;
        }
      }
      const amp0 = checkerboardAmplitude(s);
      let blew = false, atStep = null;
      for (let k = 0; k < 300; k++) {
        s.step(dt, { advectDye: false });
        if (s.isBlownUp() || checkerboardAmplitude(s) > 1e6 * amp0) {
          blew = true; atStep = k + 1; break;
        }
      }
      out.explicitDiffusion.inSolver.push({
        diffusionNumber: d, dt, blewUp: blew, blewUpAtStep: atStep,
        modeGrowth: blew ? null : checkerboardAmplitude(s) / amp0,
        modeAmplitudeInitial: amp0,
      });
    }

    const isoStable = out.explicitDiffusion.isolated.filter(p => !p.unstable);
    const isoUnstable = out.explicitDiffusion.isolated.filter(p => p.unstable);
    // Instability means the mode GROWS, i.e. amplification > 1. Using the
    // blow-up flag instead would classify d = 0.30 as stable merely because
    // 300 steps was not long enough for a 42x growth to reach the threshold --
    // the mode is unmistakably growing and the run is unstable.
    const solStable = out.explicitDiffusion.inSolver.filter(
      p => !p.blewUp && p.modeGrowth !== null && p.modeGrowth <= 1);
    const solUnstable = out.explicitDiffusion.inSolver.filter(
      p => p.blewUp || (p.modeGrowth !== null && p.modeGrowth > 1));
    out.explicitDiffusion.measuredBoundary = {
      isolated: {
        lastStable: isoStable.length ? Math.max(...isoStable.map(p => p.diffusionNumber)) : null,
        firstUnstable: isoUnstable.length ? Math.min(...isoUnstable.map(p => p.diffusionNumber)) : null,
      },
      inSolver: {
        lastStable: solStable.length ? Math.max(...solStable.map(p => p.diffusionNumber)) : null,
        firstUnstable: solUnstable.length ? Math.min(...solUnstable.map(p => p.diffusionNumber)) : null,
      },
    };
  }

  // (c) Vorticity confinement: an artificial energy source, swept to find where
  // it overwhelms dissipation entirely. Runs OUTSIDE validation mode by
  // necessity -- the point is to characterise the thing validation forbids.
  out.confinement = {
    n, nu,
    note: "confinement is non-physical; this locates where it destabilises, and is why it is disabled in every other study",
    points: [],
  };
  {
    const dt = 0.01;
    // Swept far enough to actually reach divergence. The first version stopped
    // at eps=80, where the energy had already grown 4667x but had not crossed
    // the blow-up threshold -- so the sweep reported "stable everywhere" while
    // its own data showed the energy exploding. The interesting number is not
    // where it becomes non-finite but where it stops decaying at all, which
    // happens as early as eps=1.
    for (const eps of [0, 0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320]) {
      const s = new FluidSolver({ n, nu, advection: "semi-lagrangian", confinement: eps, projection: "fft" });
      s.init((x, y) => taylorGreen.u(x, y, 0, nu), (x, y) => taylorGreen.v(x, y, 0, nu));
      const ke0 = s.kineticEnergy();
      let blew = false, atStep = null;
      for (let k = 0; k < 200; k++) {
        s.step(dt, { advectDye: false });
        if (s.isBlownUp() || s.kineticEnergy() > 1e6 * ke0) { blew = true; atStep = k + 1; break; }
      }
      const keRatio = blew ? null : s.kineticEnergy() / ke0;
      out.confinement.points.push({
        epsilon: eps, blewUp: blew, blewUpAtStep: atStep,
        keRatio,
        keExactRatio: taylorGreen.kineticEnergy(200 * dt, nu) / taylorGreen.kineticEnergy(0, nu),
        // The honest headline: is the simulation gaining energy that physics
        // says it should be losing?
        addsEnergy: keRatio === null ? true : keRatio > 1,
      });
    }
    const gains = out.confinement.points.filter(p => p.addsEnergy);
    out.confinement.smallestEpsilonThatAddsEnergy =
      gains.length ? Math.min(...gains.map(p => p.epsilon)) : null;
  }

  // (d) Pressure iterations vs residual divergence. Under-converging the
  // projection is the most common way a fluid sim is quietly wrong: the flow
  // still looks plausible while mass is not conserved.
  out.projectionIterations = { n, nu, points: [] };
  {
    const dt = 0.01;
    for (const maxIter of [1, 2, 5, 10, 20, 50, 100, 400]) {
      const s = new FluidSolver({
        n, nu, validation: true, advection: "semi-lagrangian",
        projTol: 1e-14, projMaxIter: maxIter,
      });
      s.init((x, y) => taylorGreen.u(x, y, 0, nu), (x, y) => taylorGreen.v(x, y, 0, nu));
      for (let k = 0; k < 50; k++) s.step(dt, { advectDye: false });
      out.projectionIterations.points.push({
        maxIter,
        maxDivergence: s.maxDivergence(),
        l2Divergence: s.l2Divergence(),
        relL2: s.isBlownUp() ? null : velocityErrorTG(s.g, s.u, s.v, s.t, nu),
      });
    }
  }

  return out;
}

/**
 * Amplitude of the grid-scale (checkerboard) mode in the velocity field.
 *
 * Obtained by projecting onto sgn(i+j), which is exactly the mode the explicit
 * diffusion stability analysis is about. An earlier version measured the
 * departure from the exact Taylor-Green solution instead, and reported growth
 * of ~1e5 even at d = 0.05 where the scheme is deeply stable -- because that
 * norm is dominated by the solver's ordinary truncation error, which has
 * nothing to do with the seeded mode. Projecting picks out the one component
 * whose growth the theory predicts.
 */
function checkerboardAmplitude(s) {
  const g = s.g;
  let cu = 0, cv = 0;
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      const sgn = ((i + j) % 2 === 0) ? 1 : -1;
      cu += sgn * s.u[g.idxU(i, j)];
      cv += sgn * s.v[g.idxV(i, j)];
    }
  }
  const N = g.nx * g.ny;
  return Math.hypot(cu / N, cv / N);
}

// ========================================== 5. energy spectrum vs Kraichnan

/**
 * Forced 2D turbulence, and its energy spectrum against Kraichnan-Batchelor.
 *
 * THE 2D THEORY, since it is the whole reason this study is shaped as it is.
 * Kolmogorov's k^(-5/3) describes THREE-dimensional turbulence, where vortex
 * stretching drives energy from large scales to small. In two dimensions vortex
 * stretching does not exist, vorticity is materially conserved, and there is a
 * second inviscid invariant (enstrophy). Kraichnan (1967) and Batchelor (1969)
 * showed the consequence is a DUAL cascade:
 *
 *   - energy travels UPSCALE from the forcing, k < k_f, with slope -5/3
 *   - enstrophy travels DOWNSCALE from the forcing, k > k_f, with slope -3
 *
 * So forcing must be injected at an intermediate k_f for both ranges to exist
 * at all, and large-scale drag is required or the inverse cascade condenses
 * into a box-sized vortex pair and no steady state is ever reached.
 *
 * The fit windows are fixed from k_f and the grid before the spectrum is
 * looked at. Choosing a window after seeing the data is how almost any curve
 * can be made to fit -5/3; the full spectrum is reported alongside the fits so
 * the choice can be checked.
 */
export function spectrumStudy({
  n = 256, nu = 1e-4, dt = 0.004,
  kf = 28, forcingWidth = 2, amp = 1.2, drag = 0.08,
  spinUpTime = 12, sampleTime = 8, sampleEvery = 20,
  scheme = "maccormack", seed = 1234,
} = {}) {
  const s = new FluidSolver({ n, nu, validation: true, advection: scheme, projTol: 1e-8, projMaxIter: 300 });

  // Start from rest; the forcing builds the flow. Seeding with noise would put
  // energy at scales the forcing never chose and contaminate both ranges.
  s.init(() => 0, () => 0);

  let forceSeed = seed;
  const force = (g, u, v, stepDt) => {
    bandForcing(g, u, v, { kf, width: forcingWidth, amp, seed: forceSeed++, dt: stepDt });
  };

  const spinUpSteps = Math.round(spinUpTime / dt);
  const sampleSteps = Math.round(sampleTime / dt);
  const energyTrace = [];

  for (let k = 0; k < spinUpSteps; k++) {
    s.step(dt, { force, drag, advectDye: false });
    if (k % 100 === 0) {
      energyTrace.push({ t: s.t, ke: s.kineticEnergy(), enstrophy: s.enstrophy(), phase: "spinup" });
    }
  }

  // Time-average E(k) over the sampling window. A single snapshot of a
  // turbulent field is noisy enough that its slope is not reproducible.
  const kmax = Math.floor(n / 2);
  const Esum = new Float64Array(kmax + 1);
  let samples = 0;
  let kArr = null;

  for (let k = 0; k < sampleSteps; k++) {
    s.step(dt, { force, drag, advectDye: false });
    if (k % sampleEvery === 0) {
      const [cu, cv] = velocityAtCenters(s.g, s.u, s.v);
      const spec = energySpectrum(cu, cv, n, n);
      for (let i = 0; i <= kmax; i++) Esum[i] += spec.E[i];
      kArr = spec.k;
      samples++;
    }
    if (k % 100 === 0) {
      energyTrace.push({ t: s.t, ke: s.kineticEnergy(), enstrophy: s.enstrophy(), phase: "sample" });
    }
  }

  const E = new Float64Array(kmax + 1);
  for (let i = 0; i <= kmax; i++) E[i] = Esum[i] / samples;

  // Fit windows, fixed from physics rather than from the data:
  //   inverse-cascade range: above the box scale, below the forcing
  //   enstrophy range: above the forcing, below where the grid starts to bite
  const invLo = 3, invHi = Math.max(invLo + 3, Math.floor(kf * 0.6));
  const enstLo = Math.ceil(kf * 1.5), enstHi = Math.floor(kmax * 0.5);

  const inverseFit = fitSlope(kArr, E, invLo, invHi);
  const enstrophyFit = fitSlope(kArr, E, enstLo, enstHi);

  const [cu, cv] = velocityAtCenters(s.g, s.u, s.v);
  const parseval = parsevalCheck(cu, cv, n, n);

  return {
    params: { n, nu, dt, kf, forcingWidth, amp, drag, spinUpTime, sampleTime, scheme, seed },
    samples,
    spectrum: { k: Array.from(kArr), E: Array.from(E) },
    energyTrace,
    fits: {
      inverseCascade: { ...inverseFit, theory: -5 / 3, range: [invLo, invHi] },
      enstrophyCascade: { ...enstrophyFit, theory: -3, range: [enstLo, enstHi] },
    },
    parseval,
    finalKE: s.kineticEnergy(),
    finalEnstrophy: s.enstrophy(),
    maxDivergence: s.maxDivergence(),
  };
}


// ============================== 6. two-way fluid-structure interaction

/**
 * The immersed rigid disk: what the coupling conserves, and where it breaks.
 *
 * Three measurements, because "two-way coupling works" is three separate
 * claims and they fail in different ways:
 *
 *   1. Newton's third law -- the momentum the fluid gains is exactly what the
 *      solid loses. Exact by construction, so measured to machine precision.
 *   2. No-slip -- how well the fluid inside the body actually moves with it.
 *      A single direct-forcing pass leaves (1 - chi); more passes drive it down
 *      geometrically.
 *   3. Stability -- explicit coupling diverges when the fluid's inertia rivals
 *      the body's. Swept in both translation (density ratio) and rotation
 *      (moment of inertia), because they have separate thresholds.
 */
export function fsiStudy({ n = 64, steps = 60, dt = 0.01 } = {}) {
  const out = {};

  // --- 1. momentum conservation, with and without the added-mass correction
  out.momentum = {
    note: "the coupling exchanges momentum exactly; the added-mass correction " +
          "trades that exactness for stability, and is off by default",
    points: [],
  };
  for (const addedMassCorrection of [false, true]) {
    const g = new Grid(n, n);
    const u = g.u().fill(1.2), v = g.v().fill(-0.4);
    const d = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.7, density: 2, addedMassCorrection });
    const [fx0, fy0] = fluidMomentum(g, u, v);
    const [sx0, sy0] = d.momentum();
    applyCoupling(g, u, v, [d], dt);
    d.integrate(dt);
    const [fx1, fy1] = fluidMomentum(g, u, v);
    const [sx1, sy1] = d.momentum();
    const before = Math.hypot(fx0 + sx0, fy0 + sy0) || 1;
    out.momentum.points.push({
      addedMassCorrection,
      relativeDrift: Math.hypot((fx1 + sx1) - (fx0 + sx0), (fy1 + sy1) - (fy0 + sy0)) / before,
    });
  }

  // --- 2. no-slip residual vs forcing passes
  out.noSlip = {
    note: "a direct-forcing pass moves the fluid a fraction chi toward the body, " +
          "leaving (1 - chi); repeating drives it down geometrically",
    points: [],
  };
  for (const passes of [1, 2, 3, 4]) {
    const g = new Grid(96, 96);
    const u = g.u().fill(1.0), v = g.v();
    const d = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.9, fixed: true });
    applyCoupling(g, u, v, [d], dt, { passes });
    const e = slipError(g, u, v, d);
    out.noSlip.points.push({ passes, maxSlip: e.max, meanSlip: e.mean, samples: e.samples });
  }

  // --- 3a. translational stability vs density ratio
  const runTranslation = (density, addedMassCorrection) => {
    const d = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.6, density, addedMassCorrection });
    const s = new FluidSolver({
      n, nu: 0, validation: true, projection: "fft", advection: "maccormack", solids: [d],
    });
    s.init(() => 1.0, () => 0);
    for (let k = 0; k < steps; k++) {
      s.step(dt, { advectDye: false });
      if (!Number.isFinite(d.vx) || Math.abs(d.vx) > 50) {
        return { blewUp: true, vx: null, atStep: k + 1 };
      }
    }
    return { blewUp: false, vx: d.vx, atStep: null };
  };
  out.translationStability = {
    note: "in 2D a disk's added mass equals the displaced fluid mass, so at a " +
          "density ratio of 1 the fluid's inertia already equals the body's",
    steps,
    points: [0.1, 0.25, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2, 3, 5, 8].map(density => ({
      density,
      plain: runTranslation(density, false),
      corrected: runTranslation(density, true),
    })),
  };
  const stablePlain = out.translationStability.points.filter(p => !p.plain.blewUp);
  const stableCorr = out.translationStability.points.filter(p => !p.corrected.blewUp);
  out.translationStability.lowestStableDensity = {
    plain: stablePlain.length ? Math.min(...stablePlain.map(p => p.density)) : null,
    corrected: stableCorr.length ? Math.min(...stableCorr.map(p => p.density)) : null,
  };

  // --- 3b. rotational stability vs moment of inertia
  out.rotationStability = {
    note: "u = y - PI has vorticity -1; a torque-free body should approach " +
          "omega = -0.5, half the vorticity",
    steps,
    points: [],
  };
  for (const density of [1.5, 3, 10, 50, 200, 1000]) {
    const d = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.5, density });
    const s = new FluidSolver({
      n, nu: 0, validation: true, projection: "fft", advection: "maccormack", solids: [d],
    });
    s.init((x, y) => y - Math.PI, () => 0);
    let blew = false;
    for (let k = 0; k < steps; k++) {
      s.step(dt, { advectDye: false });
      if (!Number.isFinite(d.omega) || Math.abs(d.omega) > 10) { blew = true; break; }
    }
    out.rotationStability.points.push({
      density, inertia: d.inertia, blewUp: blew,
      omega: blew ? null : d.omega,
      theoreticalLimit: -0.5,
    });
  }
  const rotStable = out.rotationStability.points.filter(p => !p.blewUp);
  out.rotationStability.lowestStableInertia =
    rotStable.length ? Math.min(...rotStable.map(p => p.inertia)) : null;

  return out;
}
