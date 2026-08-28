/**
 * The forced-turbulence spectrum run, as a standalone long job.
 *
 * Split out of studies.js because it is the one study whose cost is measured in
 * tens of minutes rather than seconds. A 2D inverse cascade has to carry energy
 * from the forcing wavenumber up through every intervening octave, and each
 * octave takes several eddy turnover times, so a run that is "long" by the
 * standards of the other studies is nowhere near statistical steady state. An
 * early attempt stopped at t=4 (about 21 turnover times) and reported an
 * inverse-range slope of +1.46 -- not a wrong measurement of a cascade, but a
 * correct measurement of a cascade that had not formed yet.
 *
 * Snapshots are written as it goes, so the spectrum's approach to steady state
 * is itself part of the record rather than something asserted at the end.
 *
 * Usage:  node validate/longrun-spectrum.js [steps] [n]
 */

import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { FluidSolver, bandForcing } from "../src/cpu/solver.js";
import { velocityAtCenters } from "../src/core/grid.js";
import { energySpectrum, fitSlope, parsevalCheck } from "../src/core/fft.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "..", "results");

const STEPS = Number(process.argv[2] || 25000);
const N = Number(process.argv[3] || 256);

const params = {
  n: N,
  nu: 1e-5,
  dt: 0.004,
  kf: 24,
  forcingWidth: 1.5,
  amp: 3.0,
  drag: 0.05,
  scheme: "maccormack",
  projection: "fft",
  steps: STEPS,
  seed: 1000,
};

// Fit windows fixed from the forcing wavenumber BEFORE any data is seen.
// Choosing them afterwards is how almost any curve can be made to agree with a
// power law, so they are derived from k_f and the grid and then left alone.
const kmax = Math.floor(N / 2);
const WINDOWS = {
  inverse: [4, Math.floor(params.kf * 0.62)],          // above the box, below forcing
  enstrophy: [Math.ceil(params.kf * 1.5), Math.floor(kmax * 0.7)],
};

function run() {
  const s = new FluidSolver({
    n: params.n, nu: params.nu, validation: true,
    advection: params.scheme, projection: params.projection,
  });
  // From rest. Seeding with noise would place energy at scales the forcing did
  // not choose, contaminating both ranges being measured.
  s.init(() => 0, () => 0);

  let seed = params.seed;
  const force = (g, u, v, d) =>
    bandForcing(g, u, v, { kf: params.kf, width: params.forcingWidth, amp: params.amp, seed: seed++, dt: d });

  const snapshots = [];
  const trace = [];
  const t0 = Date.now();

  // Time-averaging accumulator, started only once the flow has plateaued.
  let accum = null, accumCount = 0;
  const AVG_FROM = Math.floor(STEPS * 0.6);

  for (let k = 0; k < STEPS; k++) {
    s.step(params.dt, { force, drag: params.drag, advectDye: false });

    if (k % 200 === 0) {
      trace.push({ step: k, t: s.t, ke: s.kineticEnergy(), enstrophy: s.enstrophy() });
    }

    if (k >= AVG_FROM && k % 25 === 0) {
      const [cu, cv] = velocityAtCenters(s.g, s.u, s.v);
      const sp = energySpectrum(cu, cv, params.n, params.n);
      if (!accum) accum = new Float64Array(sp.E.length);
      for (let i = 0; i < sp.E.length; i++) accum[i] += sp.E[i];
      accumCount++;
    }

    if (k % 2500 === 2499 || k === STEPS - 1) {
      const [cu, cv] = velocityAtCenters(s.g, s.u, s.v);
      const sp = energySpectrum(cu, cv, params.n, params.n);
      const inv = fitSlope(sp.k, sp.E, WINDOWS.inverse[0], WINDOWS.inverse[1]);
      const ens = fitSlope(sp.k, sp.E, WINDOWS.enstrophy[0], WINDOWS.enstrophy[1]);
      snapshots.push({
        step: k + 1, t: s.t,
        ke: s.kineticEnergy(), enstrophy: s.enstrophy(),
        inverseSlope: inv.slope, inverseR2: inv.r2,
        enstrophySlope: ens.slope, enstrophyR2: ens.r2,
        E: Array.from(sp.E),
      });
      console.log(
        `step ${String(k + 1).padStart(6)} t=${s.t.toFixed(1).padStart(6)} ` +
        `KE=${s.kineticEnergy().toExponential(3)} Z=${s.enstrophy().toExponential(3)} ` +
        `inv=${inv.slope.toFixed(3)}(r2 ${inv.r2.toFixed(3)}) ` +
        `ens=${ens.slope.toFixed(3)}(r2 ${ens.r2.toFixed(3)}) ` +
        `[${((Date.now() - t0) / 1000).toFixed(0)}s]`
      );
    }
  }

  // Time-averaged spectrum -- a single turbulent snapshot is too noisy for a
  // slope that reproduces.
  const E = new Float64Array(accum.length);
  for (let i = 0; i < accum.length; i++) E[i] = accum[i] / accumCount;
  const kArr = Array.from({ length: E.length }, (_, i) => i);

  const inverseFit = fitSlope(kArr, E, WINDOWS.inverse[0], WINDOWS.inverse[1]);
  const enstrophyFit = fitSlope(kArr, E, WINDOWS.enstrophy[0], WINDOWS.enstrophy[1]);
  const [cu, cv] = velocityAtCenters(s.g, s.u, s.v);

  const result = {
    kind: "forced-2d-turbulence-spectrum",
    params,
    windows: WINDOWS,
    averagedOverSamples: accumCount,
    averagedFromStep: AVG_FROM,
    wallSeconds: (Date.now() - t0) / 1000,
    spectrum: { k: kArr, E: Array.from(E) },
    fits: {
      inverseCascade: { ...inverseFit, theory: -5 / 3 },
      enstrophyCascade: { ...enstrophyFit, theory: -3 },
    },
    parseval: parsevalCheck(cu, cv, params.n, params.n),
    trace,
    snapshots,
    finalKE: s.kineticEnergy(),
    finalEnstrophy: s.enstrophy(),
    maxDivergence: s.maxDivergence(),
  };

  mkdirSync(OUT, { recursive: true });
  writeFileSync(join(OUT, "spectrum.json"), JSON.stringify(result));
  console.log(`\nwrote results/spectrum.json  (${(result.wallSeconds / 60).toFixed(1)} min)`);
  console.log(`inverse  ${inverseFit.slope.toFixed(3)} vs theory -1.667  r2=${inverseFit.r2.toFixed(3)}`);
  console.log(`enstrophy ${enstrophyFit.slope.toFixed(3)} vs theory -3.000  r2=${enstrophyFit.r2.toFixed(3)}`);
}

run();
