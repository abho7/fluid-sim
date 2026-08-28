/**
 * Runs every validation study that is cheap enough to be reproducible on
 * demand, and writes results/*.json.
 *
 * The forced-turbulence spectrum is deliberately NOT here. It needs tens of
 * thousands of steps to reach statistical steady state and lives in
 * `longrun-spectrum.js`, run separately. Bundling a 40-minute job into the
 * command people use to check the numbers would mean nobody ever runs it.
 *
 *   node validate/run.js           full
 *   node validate/run.js --quick   smaller sweeps, for iterating
 */

import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { cpus, totalmem, platform, arch } from "node:os";

import {
  advectionDiffusion, taylorGreenStudy, convergenceStudy, stabilityStudy, fsiStudy,
} from "./studies.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "..", "results");
const quick = process.argv.includes("--quick");

function section(name, fn) {
  process.stdout.write(`${name} ... `);
  const t0 = Date.now();
  const r = fn();
  const secs = (Date.now() - t0) / 1000;
  console.log(`${secs.toFixed(1)}s`);
  return { ...r, wallSeconds: secs };
}

function main() {
  mkdirSync(OUT, { recursive: true });
  const t0 = Date.now();

  const env = {
    node: process.version,
    platform: platform(),
    arch: arch(),
    cpu: cpus()[0]?.model ?? "unknown",
    cores: cpus().length,
    memGB: Math.round(totalmem() / 1e9),
    generatedAt: new Date().toISOString(),
    quick,
  };

  const advection = section("advection diffusion", () =>
    advectionDiffusion({ n: quick ? 64 : 128, dt: 0.02, steps: quick ? 50 : 100 }));

  const taylorGreen = section("taylor-green", () =>
    taylorGreenStudy({ n: quick ? 64 : 128, nu: 0.02, dt: 0.005, T: quick ? 1 : 2 }));

  const convergence = section("convergence table", () =>
    convergenceStudy(quick
      ? { Ns: [16, 32, 64], dts: [0.02, 0.01, 0.005, 0.0025] }
      : { Ns: [16, 32, 64, 128], dts: [0.02, 0.01, 0.005, 0.0025, 0.00125] }));

  const stability = section("stability sweeps", () =>
    stabilityStudy({ n: quick ? 32 : 64, nu: 0.02 }));

  const fsi = section("fluid-structure interaction", () =>
    fsiStudy({ n: quick ? 32 : 64, steps: quick ? 40 : 60 }));

  const payload = {
    env,
    totalWallSeconds: (Date.now() - t0) / 1000,
    advection,
    taylorGreen,
    convergence,
    stability,
    fsi,
  };

  writeFileSync(join(OUT, "validation.json"), JSON.stringify(payload));
  console.log(`\nwrote results/validation.json in ${payload.totalWallSeconds.toFixed(1)}s`);

  // A short human-readable summary, so a run is self-checking at a glance
  // rather than requiring the JSON to be opened.
  console.log("\n--- headline numbers ---");
  for (const [k, r] of Object.entries(advection.schemes)) {
    console.log(`advection ${k.padEnd(16)} peak retained ${(r.peakRetained * 100).toFixed(1)}%  ` +
      `D_num=${r.numericalDiffusion?.toExponential(3)}`);
  }
  for (const [k, r] of Object.entries(taylorGreen.schemes)) {
    console.log(`taylor-green ${k.padEnd(16)} nu_num=${r.nuNumerical.toExponential(3)} ` +
      `(${(r.nuNumericalRatio * 100).toFixed(1)}% of nu=${taylorGreen.nu})  ` +
      `relL2=${r.finalRelL2.toExponential(2)}  fit r2=${r.decayFitR2.toFixed(5)}`);
  }
  const b = stability.explicitDiffusion.measuredBoundary;
  console.log(`explicit diffusion (theory: unstable for nu*dt/h^2 > 0.25)`);
  console.log(`  operator alone: stable to ${b.isolated.lastStable}, unstable from ${b.isolated.firstUnstable}`);
  console.log(`  in solver:      stable to ${b.inSolver.lastStable}, unstable from ${b.inSolver.firstUnstable}`);
  const c = stability.confinement;
  console.log(`vorticity confinement starts ADDING energy at eps=${c.smallestEpsilonThatAddsEnergy}`);
  const m = fsi.momentum.points;
  console.log(`FSI momentum drift: exact path ${m[0].relativeDrift.toExponential(1)}, ` +
    `added-mass-corrected ${m[1].relativeDrift.toExponential(1)}`);
  console.log(`FSI no-slip: 1 pass ${fsi.noSlip.points[0].maxSlip.toExponential(1)}, ` +
    `4 passes ${fsi.noSlip.points[3].maxSlip.toExponential(1)}`);
  console.log(`FSI stable density >= ${fsi.translationStability.lowestStableDensity.plain} ` +
    `(plain), ${fsi.translationStability.lowestStableDensity.corrected} (corrected)`);
  const pi = stability.projectionIterations.points;
  console.log(`projection: 1 iter -> maxDiv ${pi[0].maxDivergence.toExponential(1)}, ` +
    `${pi[pi.length-1].maxIter} iters -> ${pi[pi.length-1].maxDivergence.toExponential(1)}`);
}

main();
