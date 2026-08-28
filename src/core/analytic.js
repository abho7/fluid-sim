/**
 * Closed-form solutions of the incompressible Navier-Stokes equations.
 *
 * These are the ground truth. Everything the solver claims about its own
 * accuracy is measured against the functions in this file, so they are stated
 * with their derivations and are themselves tested (a wrong "exact" solution
 * would make a broken solver look perfect, which is the worst possible failure
 * mode for a validation suite).
 */

// ========================================================== Taylor-Green vortex

/**
 * The 2D Taylor-Green vortex on a periodic [0,2π]² domain.
 *
 *   u(x,y,t) =  sin(x)cos(y)·e^(-2νt)
 *   v(x,y,t) = -cos(x)sin(y)·e^(-2νt)
 *   p(x,y,t) =  ¼(cos2x + cos2y)·e^(-4νt)
 *
 * WHY THIS IS AN EXACT SOLUTION. Write u = U(t)·f(x,y). The field is
 * divergence-free by construction: ∂u/∂x = cos(x)cos(y)·U and
 * ∂v/∂y = -cos(x)cos(y)·U, which cancel identically.
 *
 * The nonlinear term is not zero, it is a pure gradient:
 *   (u·∇)u = -½∇(sin²x + sin²y)·U²  ... which the pressure gradient absorbs
 * exactly. What remains is ∂u/∂t = ν∇²u, and since ∇²f = -2f for this f, the
 * amplitude obeys U' = -2νU, giving U(t) = e^(-2νt).
 *
 * THE CONSEQUENCE FOR VALIDATION, which matters and is easy to miss: because
 * advection is exactly cancelled by pressure, Taylor-Green barely exercises the
 * advection scheme. A solver with a badly diffusive advection step can still
 * score well here. That is why `translatingGaussian` below exists as a separate
 * test -- it isolates advection with nothing to hide behind.
 *
 * Kinetic energy ½⟨u²+v²⟩ decays as e^(-4νt) exactly, which is the quantity the
 * numerical-viscosity fit uses.
 */
export const taylorGreen = {
  name: "Taylor-Green vortex",
  domain: [2 * Math.PI, 2 * Math.PI],

  u(x, y, t, nu) { return Math.sin(x) * Math.cos(y) * Math.exp(-2 * nu * t); },
  v(x, y, t, nu) { return -Math.cos(x) * Math.sin(y) * Math.exp(-2 * nu * t); },
  p(x, y, t, nu) {
    return 0.25 * (Math.cos(2 * x) + Math.cos(2 * y)) * Math.exp(-4 * nu * t);
  },

  /** ω = ∂v/∂x − ∂u/∂y = 2·sin(x)sin(y)·e^(-2νt) */
  vorticity(x, y, t, nu) {
    return 2 * Math.sin(x) * Math.sin(y) * Math.exp(-2 * nu * t);
  },

  /**
   * Exact mean kinetic energy per unit area.
   * ⟨sin²x cos²y⟩ = ¼ over the periodic box, and likewise for v, so
   * ½(¼ + ¼)·e^(-4νt) = ¼·e^(-4νt).
   */
  kineticEnergy(t, nu) { return 0.25 * Math.exp(-4 * nu * t); },

  /** Exact mean enstrophy ½⟨ω²⟩. ⟨4sin²x sin²y⟩ = 1, so ½·e^(-4νt). */
  enstrophy(t, nu) { return 0.5 * Math.exp(-4 * nu * t); },

  /** The decay exponent of kinetic energy: KE(t) = KE(0)·e^(-rate·t). */
  energyDecayRate(nu) { return 4 * nu; },

  /** Initialise a MAC velocity field to the exact solution at time t. */
  init(g, u, v, t, nu) {
    g.fillU(u, (x, y) => this.u(x, y, t, nu));
    g.fillV(v, (x, y) => this.v(x, y, t, nu));
    return [u, v];
  },
};

// ==================================================== pure-advection test case

/**
 * A Gaussian blob carried by a uniform velocity field.
 *
 * With u = (a, b) constant, the velocity field is divergence-free and its own
 * exact solution for all time, and a passive scalar obeys ∂φ/∂t + u·∇φ = 0,
 * whose solution is the initial blob translated by (a·t, b·t). On a periodic
 * domain it wraps.
 *
 * WHY THIS TEST EXISTS. It is the cleanest possible measurement of advection
 * error, because there is nothing else happening: no pressure, no viscosity, no
 * nonlinearity. Any spreading of the blob is numerical diffusion introduced by
 * the advection scheme, and the amount of spreading converts directly into an
 * effective diffusion coefficient.
 *
 * A semi-Lagrangian step interpolates bilinearly, and bilinear interpolation of
 * a smooth field loses variance every step. That loss is the artificial
 * dissipation the brief asks vorticity confinement to counteract; this test is
 * where it gets a number instead of an adjective.
 *
 * The exact solution is stated with periodic images summed, because after
 * enough time the blob's tail wraps onto itself and ignoring that would make
 * the "exact" solution wrong at the 1e-8 level -- below the scheme's error, but
 * it would put a floor under the convergence study.
 */
export const translatingGaussian = {
  name: "Translating Gaussian",
  domain: [2 * Math.PI, 2 * Math.PI],

  /**
   * @param {number} sigma blob width
   * @param {number} ax,ay uniform advecting velocity
   * @param {number} cx,cy initial centre
   */
  scalar(x, y, t, { sigma = 0.35, ax = 1.0, ay = 0.6, cx = Math.PI, cy = Math.PI } = {}) {
    const L = 2 * Math.PI;
    const mx = cx + ax * t;
    const my = cy + ay * t;
    // Sum periodic images. Three either side is far more than enough: the
    // fourth contributes < e^(-(3L/σ)²/2) which underflows f64 for any σ we use.
    let s = 0;
    for (let p = -3; p <= 3; p++) {
      for (let q = -3; q <= 3; q++) {
        const dx = x - mx + p * L;
        const dy = y - my + q * L;
        s += Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      }
    }
    return s;
  },

  velocity({ ax = 1.0, ay = 0.6 } = {}) {
    return { u: () => ax, v: () => ay };
  },

  init(g, dye, opts = {}) {
    g.fillP(dye, (x, y) => this.scalar(x, y, 0, opts));
    return dye;
  },

  initVelocity(g, u, v, { ax = 1.0, ay = 0.6 } = {}) {
    u.fill(ax);
    v.fill(ay);
    return [u, v];
  },
};

// ================================================================ error norms

/**
 * Relative L2 error between a numerical field and an exact one.
 *
 * Relative rather than absolute because Taylor-Green decays exponentially: an
 * absolute error that stays flat while the solution decays by e^(-4νt) would
 * look like the solver getting worse when it is the signal getting smaller.
 * Normalising by the exact field's norm measures the thing people mean.
 *
 * Falls back to absolute error when the exact field is (near) zero, rather than
 * dividing by zero and reporting Infinity or NaN into a plot.
 */
export function relL2(numeric, exact) {
  if (numeric.length !== exact.length) {
    throw new Error(`length mismatch: ${numeric.length} vs ${exact.length}`);
  }
  let num = 0, den = 0;
  for (let k = 0; k < numeric.length; k++) {
    const d = numeric[k] - exact[k];
    num += d * d;
    den += exact[k] * exact[k];
  }
  if (den < 1e-300) return Math.sqrt(num / numeric.length);
  return Math.sqrt(num / den);
}

/** Absolute L2 error, mean-square normalised (so it is grid-size independent). */
export function absL2(numeric, exact) {
  let s = 0;
  for (let k = 0; k < numeric.length; k++) {
    const d = numeric[k] - exact[k];
    s += d * d;
  }
  return Math.sqrt(s / numeric.length);
}

/** Max absolute (L∞) error. */
export function maxError(numeric, exact) {
  let m = 0;
  for (let k = 0; k < numeric.length; k++) {
    m = Math.max(m, Math.abs(numeric[k] - exact[k]));
  }
  return m;
}

/**
 * Combined relative L2 error of a MAC velocity field against an exact solution.
 *
 * u and v are compared at their own staggered sample points -- interpolating
 * both to centres first would add the interpolation's own smoothing to the
 * measured error and flatter the solver.
 */
export function velocityErrorTG(g, u, v, t, nu) {
  const eu = g.fillU(g.u(), (x, y) => taylorGreen.u(x, y, t, nu));
  const ev = g.fillV(g.v(), (x, y) => taylorGreen.v(x, y, t, nu));
  let num = 0, den = 0;
  for (let k = 0; k < u.length; k++) {
    const du = u[k] - eu[k], dv = v[k] - ev[k];
    num += du * du + dv * dv;
    den += eu[k] * eu[k] + ev[k] * ev[k];
  }
  if (den < 1e-300) return Math.sqrt(num / (2 * u.length));
  return Math.sqrt(num / den);
}
