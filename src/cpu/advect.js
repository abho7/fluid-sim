/**
 * Advection: transporting a quantity along the flow.
 *
 * Two schemes, because the difference between them is one of this project's
 * actual results rather than an implementation detail.
 *
 * SEMI-LAGRANGIAN (Stam 1999). For each destination sample, trace the flow
 * BACKWARD one timestep to find where the material now arriving there came
 * from, and interpolate the old field at that point. Unconditionally stable at
 * any timestep -- which is the entire reason stable fluids exists, since the
 * obvious explicit schemes blow up the moment a particle crosses a cell.
 *
 * The price is severe numerical diffusion, and it is worth being precise about
 * where it comes from, because the brief describes vorticity confinement as a
 * fix for it. Each step interpolates bilinearly, and bilinear interpolation of
 * anything other than a linear field is a weighted average, i.e. a low-pass
 * filter. Applying a low-pass filter once per step is indistinguishable from
 * adding a diffusion term: the scheme has an effective viscosity of roughly
 * ν_num ≈ (h²/dt)·f(CFL) that nobody asked for. Small vortices are erased
 * within a few steps, the energy spectrum loses its tail, and the flow looks
 * like syrup. Vorticity confinement pushes energy back into those scales, but
 * it is a fabricated force, not a recovery of the information that was lost --
 * which is why every validation run in this project has it switched off.
 *
 * MACCORMACK / BFECC (Selle et al. 2008). Run the semi-Lagrangian step forward,
 * then run it backward from the result. If advection were exact the round trip
 * would return the original field, so whatever it misses by is (twice) the
 * error, and subtracting half of it cancels the leading-order term. That lifts
 * the scheme to second order and removes most of the artificial diffusion --
 * at the cost of losing the unconditional monotonicity, so the corrected value
 * has to be clamped to the range of the cells it interpolated from or it will
 * overshoot into new extrema and go unstable. The limiter is not optional.
 */

import { sampleU, sampleV, sampleP, bilerp, wrap } from "../core/grid.js";

/**
 * Trace a particle backward through the velocity field.
 *
 * RK2 (midpoint) rather than forward Euler. Euler's O(dt²) path error is a
 * first-order error in the advected field and, on a rotating flow like every
 * vortex here, it systematically traces to the outside of the true arc --
 * vortices spin down measurably faster with Euler tracing even before the
 * interpolation diffusion is accounted for.
 */
function traceBack(g, u, v, x, y, dt) {
  const u1 = sampleU(g, u, x, y);
  const v1 = sampleV(g, v, x, y);
  const xm = x - 0.5 * dt * u1;
  const ym = y - 0.5 * dt * v1;
  const u2 = sampleU(g, u, xm, ym);
  const v2 = sampleV(g, v, xm, ym);
  return [x - dt * u2, y - dt * v2];
}

/**
 * Semi-Lagrangian advection of a cell-centred scalar.
 * @param {Float64Array} q field to advect (read)
 * @param {Float64Array} out destination (written)
 */
export function advectScalarSL(g, u, v, q, out, dt) {
  const { nx, ny } = g;
  out = out || g.p();
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const [x, y] = traceBack(g, u, v, g.pX(i), g.pY(j), dt);
      out[g.idxP(i, j)] = sampleP(g, q, x, y);
    }
  }
  return out;
}

/** Semi-Lagrangian advection of the MAC velocity field itself (self-advection). */
export function advectVelocitySL(g, u, v, outU, outV, dt) {
  const { nx, ny } = g;
  outU = outU || g.u();
  outV = outV || g.v();

  // u and v are traced from their OWN sample positions. Tracing both from cell
  // centres and interpolating afterwards would add a full extra interpolation
  // of smoothing per step, roughly doubling the artificial viscosity -- a
  // shortcut that is easy to take on a MAC grid and shows up only as "the
  // solver is more dissipative than it should be".
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const [x, y] = traceBack(g, u, v, g.uX(i), g.uY(j), dt);
      outU[g.idxU(i, j)] = sampleU(g, u, x, y);
    }
  }
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const [x, y] = traceBack(g, u, v, g.vX(i), g.vY(j), dt);
      outV[g.idxV(i, j)] = sampleV(g, v, x, y);
    }
  }
  return [outU, outV];
}

/**
 * Range of the four cells a backward trace interpolated from.
 * Used to clamp the MacCormack correction: the corrected value may not leave
 * the interval its own inputs spanned.
 */
function interpBounds(arr, nx, ny, gx, gy) {
  const i0 = Math.floor(gx), j0 = Math.floor(gy);
  const i0w = wrap(i0, nx), i1w = wrap(i0 + 1, nx);
  const j0w = wrap(j0, ny), j1w = wrap(j0 + 1, ny);
  const a = arr[j0w * nx + i0w], b = arr[j0w * nx + i1w];
  const c = arr[j1w * nx + i0w], d = arr[j1w * nx + i1w];
  return [Math.min(a, b, c, d), Math.max(a, b, c, d)];
}

/**
 * MacCormack advection of a cell-centred scalar.
 *
 * q̂     = A(q)          forward semi-Lagrangian
 * q̃     = A⁻¹(q̂)        backward from the result
 * q_new = q̂ + (q − q̃)/2  error-corrected, then limited
 */
export function advectScalarMacCormack(g, u, v, q, out, dt, scratch) {
  const { nx, ny } = g;
  out = out || g.p();
  const fwd = scratch?.fwd || g.p();
  const back = scratch?.back || g.p();

  advectScalarSL(g, u, v, q, fwd, dt);
  advectScalarSL(g, u, v, fwd, back, -dt);   // reverse the flow

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = g.idxP(i, j);
      const corrected = fwd[k] + 0.5 * (q[k] - back[k]);

      // LIMITER. Without this the correction creates new extrema -- a scalar
      // that was everywhere in [0,1] acquires values outside it, and on the
      // velocity field those overshoots feed back through advection and grow.
      // Clamping to the source cells' range keeps the scheme 2nd order where
      // the field is smooth and drops it to the (monotone) semi-Lagrangian
      // value exactly where it would have overshot.
      const [x, y] = traceBack(g, u, v, g.pX(i), g.pY(j), dt);
      const [lo, hi] = interpBounds(q, nx, ny, x / g.dx - 0.5, y / g.dy - 0.5);
      out[k] = corrected < lo ? lo : corrected > hi ? hi : corrected;
    }
  }
  return out;
}

/** MacCormack advection of the MAC velocity field. */
export function advectVelocityMacCormack(g, u, v, outU, outV, dt, scratch) {
  const { nx, ny } = g;
  outU = outU || g.u();
  outV = outV || g.v();
  const fu = scratch?.fu || g.u(), fv = scratch?.fv || g.v();
  const bu = scratch?.bu || g.u(), bv = scratch?.bv || g.v();

  // Forward pass, then the backward pass STARTING FROM THE FORWARD RESULT --
  // that round trip is the whole error estimate. Both passes use the ORIGINAL
  // velocity as the advecting flow: using the forward result to advect
  // backward would make the operator nonlinear in a way the error-cancellation
  // argument does not cover, and the correction would stop cancelling the
  // leading error term.
  advectVelocitySL(g, u, v, fu, fv, dt);
  advectComponentWith(g, u, v, fu, bu, -dt, "u");
  advectComponentWith(g, u, v, fv, bv, -dt, "v");

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = g.idxU(i, j);
      const corrected = fu[k] + 0.5 * (u[k] - bu[k]);
      const [x, y] = traceBack(g, u, v, g.uX(i), g.uY(j), dt);
      const [lo, hi] = interpBounds(u, nx, ny, x / g.dx, y / g.dy - 0.5);
      outU[k] = corrected < lo ? lo : corrected > hi ? hi : corrected;
    }
  }
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = g.idxV(i, j);
      const corrected = fv[k] + 0.5 * (v[k] - bv[k]);
      const [x, y] = traceBack(g, u, v, g.vX(i), g.vY(j), dt);
      const [lo, hi] = interpBounds(v, nx, ny, x / g.dx - 0.5, y / g.dy);
      outV[k] = corrected < lo ? lo : corrected > hi ? hi : corrected;
    }
  }
  return [outU, outV];
}

/**
 * Advect one MAC component `field` through the flow (u,v), sampling at that
 * component's own staggered positions. Separate from advectVelocitySL because
 * MacCormack needs to advect a field that is NOT the advecting velocity.
 */
function advectComponentWith(g, u, v, field, out, dt, which) {
  const { nx, ny } = g;
  const isU = which === "u";
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const px = isU ? g.uX(i) : g.vX(i);
      const py = isU ? g.uY(j) : g.vY(j);
      const [x, y] = traceBack(g, u, v, px, py, dt);
      out[isU ? g.idxU(i, j) : g.idxV(i, j)] = isU
        ? bilerp(field, nx, ny, x / g.dx, y / g.dy - 0.5)
        : bilerp(field, nx, ny, x / g.dx - 0.5, y / g.dy);
    }
  }
  return out;
}

export const ADVECTION = {
  "semi-lagrangian": {
    label: "Semi-Lagrangian",
    order: 1,
    scalar: advectScalarSL,
    velocity: advectVelocitySL,
  },
  maccormack: {
    label: "MacCormack (limited)",
    order: 2,
    scalar: advectScalarMacCormack,
    velocity: advectVelocityMacCormack,
  },
};
