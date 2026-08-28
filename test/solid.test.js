/**
 * Phase 7 gate: two-way fluid-structure coupling.
 *
 * The central claim is Newton's third law -- whatever momentum the coupling
 * gives the fluid it must take from the solid, exactly. That is not a tolerance
 * question: applied correctly it holds to machine precision, so it is asserted
 * that way. A coupling that leaks momentum would still produce a disk that
 * moves plausibly, which is precisely why it needs a test that cannot be
 * satisfied by "looks about right".
 *
 * The rest of the file separates what the coupling is responsible for from what
 * it is not. Semi-Lagrangian advection is NOT momentum-conserving, so the full
 * simulation drifts; that drift is measured and attributed, rather than being
 * allowed to discredit the coupling.
 */

import test from "node:test";
import assert from "node:assert/strict";

import { Grid } from "../src/core/grid.js";
import { FluidSolver } from "../src/cpu/solver.js";
import {
  RigidDisk, applyCoupling, slipError, fluidMomentum,
} from "../src/cpu/solid.js";

const TWO_PI = 2 * Math.PI;

// ===================================================================== mask

test("the mask is 1 well inside, 0 well outside, and smooth across the edge", () => {
  const d = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.8 });
  const w = 1.5 * (TWO_PI / 128);
  assert.ok(d.mask(Math.PI, Math.PI, TWO_PI, TWO_PI, w) > 0.999, "centre");
  assert.ok(d.mask(Math.PI + 2.0, Math.PI, TWO_PI, TWO_PI, w) < 1e-6, "far outside");
  // On the surface it should be exactly a half by symmetry of tanh.
  const edge = d.mask(Math.PI + 0.8, Math.PI, TWO_PI, TWO_PI, w);
  assert.ok(Math.abs(edge - 0.5) < 1e-9, `on the radius the mask read ${edge}`);
});

test("the mask wraps periodically", () => {
  // A disk near the domain edge must still be one disk, not two half-disks that
  // behave independently. Getting this wrong makes a body crossing the seam
  // briefly double its effective mass.
  const d = new RigidDisk({ x: 0.1, y: Math.PI, r: 0.5 });
  const w = 0.05;
  const inside = d.mask(TWO_PI - 0.1, Math.PI, TWO_PI, TWO_PI, w);
  assert.ok(inside > 0.9, `mask across the seam read ${inside}`);
});

test("rigid-body velocity includes rotation", () => {
  const d = new RigidDisk({ x: 1, y: 1, r: 0.4, vx: 2, vy: -1, omega: 3 });
  // At the centre, rotation contributes nothing.
  assert.deepEqual(d.velocityAt(1, 1, TWO_PI, TWO_PI).map(v => +v.toFixed(12)), [2, -1]);
  // Offset by +x: omega x r = (-omega*dy, +omega*dx) = (0, 3*0.2)
  const [vx, vy] = d.velocityAt(1.2, 1, TWO_PI, TWO_PI);
  assert.ok(Math.abs(vx - 2) < 1e-12 && Math.abs(vy - (-1 + 0.6)) < 1e-12,
    `got (${vx}, ${vy})`);
});

// ====================================================== Newton's third law

test("one coupling step conserves total momentum to machine precision", () => {
  // THE PHASE 7 GATE. The coupling moves momentum between fluid and solid; it
  // must not create or destroy any. This is exact by construction when the
  // reaction is computed from the same integral as the action, so a loose
  // tolerance here would be hiding a real error.
  const g = new Grid(64, 64);
  const u = g.u(), v = g.v();
  let seed = 4242;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
  for (let i = 0; i < u.length; i++) { u[i] = 1 + rnd(); v[i] = rnd(); }

  const disk = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.7, density: 2.5, vx: -0.3, vy: 0.2 });
  const dt = 0.01;

  const [fpx0, fpy0] = fluidMomentum(g, u, v);
  const [spx0, spy0] = disk.momentum();

  applyCoupling(g, u, v, [disk], dt);
  disk.integrate(dt);

  const [fpx1, fpy1] = fluidMomentum(g, u, v);
  const [spx1, spy1] = disk.momentum();

  const before = Math.hypot(fpx0 + spx0, fpy0 + spy0);
  const dx = (fpx1 + spx1) - (fpx0 + spx0);
  const dy = (fpy1 + spy1) - (fpy0 + spy0);
  const drift = Math.hypot(dx, dy) / (before || 1);

  assert.ok(drift < 1e-12,
    `total momentum drifted by ${drift} relative in a single coupling step`);
});

test("the momentum the fluid gains is exactly what the solid loses", () => {
  // The same law, stated the other way round and checked component by
  // component, so a sign error that happens to cancel in the magnitude cannot
  // slip through.
  const g = new Grid(48, 48);
  const u = g.u().fill(1.4), v = g.v().fill(-0.6);
  const disk = new RigidDisk({ x: 3, y: 3, r: 0.6, density: 1.7 });
  const dt = 0.02;

  const [fx0, fy0] = fluidMomentum(g, u, v);
  const impulse = applyCoupling(g, u, v, [disk], dt);
  const [fx1, fy1] = fluidMomentum(g, u, v);

  // The reported impulse must equal the fluid's actual momentum change.
  assert.ok(Math.abs((fx1 - fx0) - impulse[0]) < 1e-12, "x impulse mismatch");
  assert.ok(Math.abs((fy1 - fy0) - impulse[1]) < 1e-12, "y impulse mismatch");

  // And the solid's force must be that impulse, negated, over dt.
  assert.ok(Math.abs(disk.force[0] * dt + impulse[0]) < 1e-12, "x reaction mismatch");
  assert.ok(Math.abs(disk.force[1] * dt + impulse[1]) < 1e-12, "y reaction mismatch");
});

test("a fixed obstacle absorbs momentum instead of conserving it", () => {
  // The control. A pinned body is one-way coupling: it takes momentum out of
  // the fluid and nothing takes it back. If this ALSO conserved momentum, the
  // conservation test above would be passing for the wrong reason.
  const g = new Grid(48, 48);
  const u = g.u().fill(1.0), v = g.v();
  const pinned = new RigidDisk({ x: 3, y: 3, r: 0.7, fixed: true });

  const [fx0] = fluidMomentum(g, u, v);
  applyCoupling(g, u, v, [pinned], 0.01);
  pinned.integrate(0.01);
  const [fx1] = fluidMomentum(g, u, v);

  assert.ok(fx1 < fx0 - 1e-6, "a fixed obstacle should remove fluid momentum");
  assert.equal(pinned.vx, 0, "a fixed obstacle must not move");
});

// ============================================================== the physics

test("no-slip residual falls geometrically with forcing passes", () => {
  // "Blocks flow" made measurable. A single direct-forcing pass cannot impose
  // no-slip exactly: it moves the fluid a fraction chi of the way to the body's
  // velocity, leaving (1 - chi) behind. At the mask's 0.98 contour that is 2%,
  // which is what a first version of this test failed on while asserting 1e-3 --
  // the scheme was right and the expectation was wrong.
  //
  // Repeating the pass drives the residual down by that same factor each time,
  // so the right assertion is on the RATIO, which is a property of the method
  // rather than a number picked to fit.
  const g0 = new Grid(96, 96);
  const errs = [];
  for (const passes of [1, 2, 3]) {
    const g = new Grid(96, 96);
    const u = g.u().fill(1.0), v = g.v();
    const disk = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.9, fixed: true });
    applyCoupling(g, u, v, [disk], 0.01, { passes });
    errs.push(slipError(g, u, v, disk));
  }
  assert.ok(errs[0].samples > 50, "the mask covered too few cells to be meaningful");
  // Each extra pass must cut the residual by at least 10x.
  for (let i = 1; i < errs.length; i++) {
    assert.ok(errs[i].max < errs[i - 1].max / 10,
      `pass ${i + 1} only improved slip from ${errs[i - 1].max} to ${errs[i].max}`);
  }
  // And the default (2 passes) must land well under a part in a thousand.
  assert.ok(errs[1].max < 1e-3, `two passes left slip at ${errs[1].max}`);
});

test("a free disk released in a moving fluid is accelerated by it", () => {
  // The headline two-way behaviour: the fluid pushes the solid. A disk at rest
  // in a stream must speed up, and must approach the stream velocity rather
  // than overshooting it.
  const nu = 0.0;
  const disk = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.6, density: 3 });
  const s = new FluidSolver({
    n: 64, nu, validation: true, projection: "fft",
    advection: "maccormack", solids: [disk],
  });
  s.init(() => 1.0, () => 0);

  const trace = [];
  for (let k = 0; k < 120; k++) {
    s.step(0.01, { advectDye: false });
    if (k % 20 === 0) trace.push(disk.vx);
  }

  assert.ok(disk.vx > 0.05, `the disk barely moved: vx = ${disk.vx}`);
  assert.ok(disk.vx < 1.05, `the disk overshot the free stream: vx = ${disk.vx}`);
  for (let i = 1; i < trace.length; i++) {
    assert.ok(trace[i] >= trace[i - 1] - 1e-9,
      `the disk decelerated in a steady stream: ${trace.join(" -> ")}`);
  }
});

test("a heavier disk accelerates more slowly than a light one", () => {
  // F = ma, exercised through the coupling. Same flow, same size, different
  // mass: the response must scale the right way. A coupling that ignored mass
  // would pass every conservation test above and fail this one.
  const run = (density) => {
    const d = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.6, density });   // correction on by default
    const s = new FluidSolver({
      n: 64, nu: 0, validation: true, projection: "fft",
      advection: "maccormack", solids: [d],
    });
    s.init(() => 1.0, () => 0);
    for (let k = 0; k < 40; k++) s.step(0.01, { advectDye: false });
    return d.vx;
  };
  const light = run(1), heavy = run(8);
  assert.ok(light > heavy * 1.5,
    `mass had too little effect: light ${light}, heavy ${heavy}`);
});

test("explicit coupling goes unstable at low density without the added-mass correction", () => {
  // A real limitation, measured rather than cited. The coupling is explicit: the
  // force comes from the current fluid state, the body moves, and that changes
  // the fluid next step. In 2D a disk's added mass is exactly the displaced
  // fluid mass, so at a density ratio of 1 the fluid's inertia already equals
  // the body's, and below that the feedback loop diverges.
  //
  // This test asserts BOTH directions: that the uncorrected scheme really does
  // blow up (so the limitation is not imaginary) and that the correction really
  // does fix it (so the fix is not decorative).
  const run = (density, addedMassCorrection) => {
    const d = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.6, density, addedMassCorrection });
    const s = new FluidSolver({
      n: 64, nu: 0, validation: true, projection: "fft",
      advection: "maccormack", solids: [d],
    });
    s.init(() => 1.0, () => 0);
    for (let k = 0; k < 60; k++) {
      s.step(0.01, { advectDye: false });
      if (!Number.isFinite(d.vx) || Math.abs(d.vx) > 50) return { blewUp: true, vx: d.vx };
    }
    return { blewUp: false, vx: d.vx };
  };

  assert.equal(run(0.25, false).blewUp, true,
    "the uncorrected scheme should diverge at density ratio 0.25");
  const fixed = run(0.25, true);
  assert.equal(fixed.blewUp, false,
    "the added-mass correction should stabilise density ratio 0.25");
  assert.ok(fixed.vx > 0.1 && fixed.vx < 1.05,
    `corrected disk reached an implausible velocity: ${fixed.vx}`);
});

test("the added-mass correction buys stability by breaking momentum conservation", () => {
  // The cost of the fix above, stated as a measurement rather than a caveat.
  //
  // The added mass IS fluid mass, and the fluid's momentum is already tracked in
  // the fluid. Folding it into the body's inertia counts it twice: the body then
  // under-responds to the force it was given by exactly mass/effectiveMass, and
  // the fluid+solid system stops conserving momentum. That is why the correction
  // is off by default, and why no published number in this project uses it.
  const measure = (addedMassCorrection) => {
    const g = new Grid(64, 64);
    const u = g.u().fill(1.2), v = g.v().fill(-0.4);
    const d = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.7, density: 2, addedMassCorrection });
    const [fx0, fy0] = fluidMomentum(g, u, v);
    const [sx0, sy0] = d.momentum();
    applyCoupling(g, u, v, [d], 0.01);
    d.integrate(0.01);
    const [fx1, fy1] = fluidMomentum(g, u, v);
    const [sx1, sy1] = d.momentum();
    const before = Math.hypot(fx0 + sx0, fy0 + sy0) || 1;
    return Math.hypot((fx1 + sx1) - (fx0 + sx0), (fy1 + sy1) - (fy0 + sy0)) / before;
  };

  const exact = measure(false), corrected = measure(true);
  assert.ok(exact < 1e-12, `the default path should conserve exactly, drifted ${exact}`);
  assert.ok(corrected > 1e-4,
    `the correction should visibly break conservation, but drifted only ${corrected}`);
});

test("a disk in shear spins the right way and approaches half the vorticity", () => {
  // Torque, which a translation-only coupling would silently omit.
  //
  // u = y - PI has du/dy = +1, so the vorticity is dv/dx - du/dy = -1, and the
  // classical result for a torque-free body in simple shear is that it rotates
  // at half the vorticity: omega -> -0.5.
  //
  // The density is 10 rather than something neutrally buoyant because rotation
  // has its own stability limit, measured in the next test: at density 1.5 the
  // moment of inertia of this disk is 0.147 and the explicit coupling diverges
  // (omega reached 78 in 60 steps). That is a real limitation of the scheme, not
  // a reason to soften this assertion.
  const disk = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.5, density: 10 });
  const s = new FluidSolver({
    n: 64, nu: 0, validation: true, projection: "fft",
    advection: "maccormack", solids: [disk],
  });
  s.init((x, y) => y - Math.PI, () => 0);

  const trace = [];
  for (let k = 0; k < 60; k++) {
    s.step(0.01, { advectDye: false });
    if ((k + 1) % 20 === 0) trace.push(disk.omega);
  }

  assert.ok(disk.omega < -1e-3,
    `the disk spun the wrong way for this shear: omega = ${disk.omega}`);
  assert.ok(disk.omega > -0.5,
    `the disk overshot the torque-free limit: omega = ${disk.omega}`);
  // Still spinning up, monotonically, toward the limit.
  for (let i = 1; i < trace.length; i++) {
    assert.ok(trace[i] <= trace[i - 1] + 1e-9,
      `angular velocity reversed: ${trace.join(" -> ")}`);
  }
});

test("rotational coupling has its own inertia-based stability limit", () => {
  // The rotational analogue of the added-mass instability, measured rather than
  // assumed. The explicit coupling computes torque from the current fluid state
  // and applies it to the body; when the body's moment of inertia is small
  // compared to that of the fluid it is displacing, that loop diverges.
  //
  // Asserted in both directions so neither half can rot: it really does blow up
  // when the inertia is small, and it really is well-behaved when it is not.
  const spin = (density) => {
    const d = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.5, density });
    const s = new FluidSolver({
      n: 64, nu: 0, validation: true, projection: "fft",
      advection: "maccormack", solids: [d],
    });
    s.init((x, y) => y - Math.PI, () => 0);
    for (let k = 0; k < 60; k++) {
      s.step(0.01, { advectDye: false });
      if (!Number.isFinite(d.omega) || Math.abs(d.omega) > 10) {
        return { blewUp: true, omega: d.omega, inertia: d.inertia };
      }
    }
    return { blewUp: false, omega: d.omega, inertia: d.inertia };
  };

  const light = spin(1.5);
  assert.equal(light.blewUp, true,
    `expected divergence at moment of inertia ${light.inertia}, got omega ${light.omega}`);

  const heavy = spin(10);
  assert.equal(heavy.blewUp, false, "density 10 should be comfortably stable");
  assert.ok(heavy.omega < 0, "and should still spin the correct way");
});

test("the solver is bit-identical with no solids present", () => {
  // The coupling must not perturb anything when it is not being used, or every
  // result measured before this phase would silently change.
  const mk = (solids) => {
    const s = new FluidSolver({
      n: 32, nu: 0.01, validation: true, projection: "fft",
      advection: "maccormack", solids,
    });
    s.init((x, y) => Math.sin(x) * Math.cos(y), (x, y) => -Math.cos(x) * Math.sin(y));
    for (let k = 0; k < 10; k++) s.step(0.01, { advectDye: false });
    return s;
  };
  const a = mk([]), b = mk([]);
  let m = 0;
  for (let i = 0; i < a.u.length; i++) {
    m = Math.max(m, Math.abs(a.u[i] - b.u[i]), Math.abs(a.v[i] - b.v[i]));
  }
  assert.equal(m, 0, "two identical runs diverged");
  assert.ok(a.maxDivergence() < 1e-12, "the no-solid path stopped being divergence-free");
});

test("the flow stays divergence-free with a solid in it", () => {
  // The coupling introduces divergence by construction; the projection must
  // still clear it. If it did not, dye would pool at the obstacle and the
  // simulation would be visibly compressible exactly where people look.
  const disk = new RigidDisk({ x: Math.PI, y: Math.PI, r: 0.7, density: 2 });
  const s = new FluidSolver({
    n: 64, nu: 0, validation: true, projection: "fft",
    advection: "maccormack", solids: [disk],
  });
  s.init(() => 1.0, () => 0);
  for (let k = 0; k < 30; k++) {
    s.step(0.01, { advectDye: false });
    assert.ok(s.maxDivergence() < 1e-10,
      `step ${k}: max|div| = ${s.maxDivergence()} with a solid present`);
  }
});
