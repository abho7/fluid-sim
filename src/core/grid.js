/**
 * MAC (marker-and-cell) staggered grid layout.
 *
 * WHY STAGGERED AND NOT COLLOCATED. Stam's original stable-fluids stores u, v
 * and p all at cell centres. On that layout the standard 5-point Laplacian
 * cannot see a checkerboard pressure field: the pressure that alternates
 * +a,-a,+a,-a between neighbours has zero discrete Laplacian, so the projection
 * is blind to it. It accumulates, and it is visible on screen as a grid-aligned
 * shimmer that no amount of extra iterations removes, because the solver
 * genuinely believes it has converged.
 *
 * Staggering fixes it structurally rather than by damping. Velocity components
 * live on the cell FACES they are normal to, pressure at the CENTRE:
 *
 *        v[i,j+1]
 *      +-----^-----+
 *      |           |
 *  u[i,j] >  p[i,j]  > u[i+1,j]
 *      |           |
 *      +-----^-----+
 *        v[i,j]
 *
 * With this arrangement the divergence at a centre uses the four faces of that
 * cell, and the pressure gradient at a face uses the two centres either side.
 * The two operators are exact negative adjoints of each other (verified in the
 * tests), which is what makes the projection an orthogonal projection and kills
 * the null-space mode.
 *
 * The cost is that u, v and p have three different array shapes, and any
 * quantity needed somewhere it is not stored has to be interpolated. That
 * bookkeeping is the price of the guarantee.
 *
 * PERIODIC DOMAIN. Every index helper wraps. The validation work requires
 * periodicity (Taylor-Green is a periodic solution, and the FFT spectra are
 * only meaningful on a periodic domain), so periodic is the primitive case and
 * walls are imposed on top of it rather than the other way round.
 */

/** Wrap an index into [0, n). Handles negatives, unlike the bare % operator. */
export function wrap(i, n) {
  return ((i % n) + n) % n;
}

export class Grid {
  /**
   * @param {number} nx cells in x
   * @param {number} ny cells in y
   * @param {number} lx physical domain width  (default 2π, the Taylor-Green box)
   * @param {number} ly physical domain height
   */
  constructor(nx, ny, lx = 2 * Math.PI, ly = 2 * Math.PI) {
    if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 2 || ny < 2) {
      throw new Error(`grid must be integers >= 2, got ${nx}x${ny}`);
    }
    this.nx = nx;
    this.ny = ny;
    this.lx = lx;
    this.ly = ly;
    this.dx = lx / nx;
    this.dy = ly / ny;

    // On a periodic MAC grid there are exactly nx distinct u-faces per row --
    // face nx is face 0. Storing nx+1 would duplicate a column and let the two
    // copies drift apart. Walls, when added, are handled by the boundary
    // routines rather than by a different array shape.
    this.uCount = nx * ny;
    this.vCount = nx * ny;
    this.pCount = nx * ny;
  }

  /** True when the cells are square, which several error norms assume. */
  get isUniform() {
    return Math.abs(this.dx - this.dy) < 1e-15 * Math.max(this.dx, this.dy);
  }

  get h() {
    if (!this.isUniform) throw new Error("h is only defined on a uniform grid");
    return this.dx;
  }

  // ------------------------------------------------------------- flat indices

  idxP(i, j) { return wrap(j, this.ny) * this.nx + wrap(i, this.nx); }
  idxU(i, j) { return wrap(j, this.ny) * this.nx + wrap(i, this.nx); }
  idxV(i, j) { return wrap(j, this.ny) * this.nx + wrap(i, this.nx); }

  // ------------------------------------------- physical positions of samples
  //
  // These are the single source of truth for where a sample "is". Advection
  // traces particles in physical space, so an inconsistency between these and
  // the interpolation routines shows up as a half-cell drift -- a bug that
  // still looks like plausible fluid, which is exactly why it gets its own
  // test rather than a visual check.

  /** Centre of cell (i,j): pressure, dye, vorticity magnitude. */
  pX(i) { return (i + 0.5) * this.dx; }
  pY(j) { return (j + 0.5) * this.dy; }

  /** u lives on the LEFT face of cell (i,j). */
  uX(i) { return i * this.dx; }
  uY(j) { return (j + 0.5) * this.dy; }

  /** v lives on the BOTTOM face of cell (i,j). */
  vX(i) { return (i + 0.5) * this.dx; }
  vY(j) { return j * this.dy; }

  // --------------------------------------------------------------- allocation

  u() { return new Float64Array(this.uCount); }
  v() { return new Float64Array(this.vCount); }
  p() { return new Float64Array(this.pCount); }

  /** Fill a u-array from f(x, y) evaluated at u sample points. */
  fillU(arr, f) {
    for (let j = 0; j < this.ny; j++) {
      for (let i = 0; i < this.nx; i++) arr[this.idxU(i, j)] = f(this.uX(i), this.uY(j));
    }
    return arr;
  }

  fillV(arr, f) {
    for (let j = 0; j < this.ny; j++) {
      for (let i = 0; i < this.nx; i++) arr[this.idxV(i, j)] = f(this.vX(i), this.vY(j));
    }
    return arr;
  }

  fillP(arr, f) {
    for (let j = 0; j < this.ny; j++) {
      for (let i = 0; i < this.nx; i++) arr[this.idxP(i, j)] = f(this.pX(i), this.pY(j));
    }
    return arr;
  }
}

// ============================================================== interpolation
//
// Bilinear, periodic. This is deliberately the same arithmetic the GPU gets
// from hardware texture filtering (verified exact against these formulas in the
// WebGPU capability check), so CPU and GPU advection can be compared field by
// field rather than merely "looking similar".

/**
 * Bilinear sample of a periodic field whose (0,0) sample sits at physical
 * (x0, y0) with spacing (dx, dy).
 */
export function bilerp(arr, nx, ny, gx, gy) {
  const i0 = Math.floor(gx);
  const j0 = Math.floor(gy);
  const fx = gx - i0;
  const fy = gy - j0;

  const i0w = wrap(i0, nx), i1w = wrap(i0 + 1, nx);
  const j0w = wrap(j0, ny), j1w = wrap(j0 + 1, ny);

  const a = arr[j0w * nx + i0w];
  const b = arr[j0w * nx + i1w];
  const c = arr[j1w * nx + i0w];
  const d = arr[j1w * nx + i1w];

  const top = a + (b - a) * fx;
  const bot = c + (d - c) * fx;
  return top + (bot - top) * fy;
}

/** Sample u at an arbitrary physical point. */
export function sampleU(g, u, x, y) {
  // u[i,j] sits at (i·dx, (j+0.5)·dy), so grid coords are offset accordingly.
  return bilerp(u, g.nx, g.ny, x / g.dx, y / g.dy - 0.5);
}

/** Sample v at an arbitrary physical point. */
export function sampleV(g, v, x, y) {
  return bilerp(v, g.nx, g.ny, x / g.dx - 0.5, y / g.dy);
}

/** Sample a cell-centred field (pressure, dye) at an arbitrary physical point. */
export function sampleP(g, p, x, y) {
  return bilerp(p, g.nx, g.ny, x / g.dx - 0.5, y / g.dy - 0.5);
}

// =================================================== differential operators
//
// divergence and gradient below must be exact negative adjoints:
//     <div(u), p> = -<u, grad(p)>
// for all u and p on a periodic domain. That identity is what makes the
// pressure projection an orthogonal projection, and it is asserted directly in
// the tests rather than assumed from the formulas looking right.

/** Divergence of a MAC velocity field, evaluated at cell centres. */
export function divergence(g, u, v, out) {
  const { nx, ny, dx, dy } = g;
  out = out || g.p();
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      // u on the left face of this cell, u on the left face of the next cell
      // (which is this cell's right face).
      const uL = u[g.idxU(i, j)];
      const uR = u[g.idxU(i + 1, j)];
      const vB = v[g.idxV(i, j)];
      const vT = v[g.idxV(i, j + 1)];
      out[g.idxP(i, j)] = (uR - uL) / dx + (vT - vB) / dy;
    }
  }
  return out;
}

/** Subtract the pressure gradient from a MAC velocity field, in place. */
export function subtractGradient(g, u, v, p, scale = 1) {
  const { nx, ny, dx, dy } = g;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      // The u-face at i sits between cell i-1 and cell i.
      u[g.idxU(i, j)] -= scale * (p[g.idxP(i, j)] - p[g.idxP(i - 1, j)]) / dx;
      v[g.idxV(i, j)] -= scale * (p[g.idxP(i, j)] - p[g.idxP(i, j - 1)]) / dy;
    }
  }
}

/**
 * Vorticity ω = ∂v/∂x − ∂u/∂y, evaluated at cell CORNERS.
 *
 * Corners are where the MAC layout puts this for free: the four velocity
 * samples surrounding a corner are exactly the ones the curl needs, with no
 * interpolation and therefore no extra smoothing. Corner (i,j) is the point
 * (i·dx, j·dy).
 */
export function vorticityCorner(g, u, v, out) {
  const { nx, ny, dx, dy } = g;
  out = out || new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const dvdx = (v[g.idxV(i, j)] - v[g.idxV(i - 1, j)]) / dx;
      const dudy = (u[g.idxU(i, j)] - u[g.idxU(i, j - 1)]) / dy;
      out[j * nx + i] = dvdx - dudy;
    }
  }
  return out;
}

/** Vorticity averaged to cell centres, for rendering and for spectra. */
export function vorticityCenter(g, u, v, out) {
  const { nx, ny } = g;
  const corner = vorticityCorner(g, u, v);
  out = out || g.p();
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      out[g.idxP(i, j)] = 0.25 * (
        corner[wrap(j, ny) * nx + wrap(i, nx)] +
        corner[wrap(j, ny) * nx + wrap(i + 1, nx)] +
        corner[wrap(j + 1, ny) * nx + wrap(i, nx)] +
        corner[wrap(j + 1, ny) * nx + wrap(i + 1, nx)]
      );
    }
  }
  return out;
}

/**
 * Velocity interpolated to cell centres.
 *
 * Needed for the energy spectrum (the FFT wants one collocated vector field)
 * and for rendering. The averaging is itself a mild low-pass filter, which
 * matters for the spectrum at the very highest wavenumbers and is noted where
 * that result is reported.
 */
export function velocityAtCenters(g, u, v, outU, outV) {
  const { nx, ny } = g;
  outU = outU || g.p();
  outV = outV || g.p();
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      outU[g.idxP(i, j)] = 0.5 * (u[g.idxU(i, j)] + u[g.idxU(i + 1, j)]);
      outV[g.idxP(i, j)] = 0.5 * (v[g.idxV(i, j)] + v[g.idxV(i, j + 1)]);
    }
  }
  return [outU, outV];
}

// ========================================================= diagnostic scalars

/**
 * Total kinetic energy per unit area, ½⟨u²+v²⟩.
 *
 * Computed from centre-interpolated velocity so it matches what the spectrum
 * integrates to (Parseval). Using face values directly would give a slightly
 * different number and the two would disagree by a few percent for no visible
 * reason -- a discrepancy that would be blamed on the solver.
 */
export function kineticEnergy(g, u, v) {
  const [cu, cv] = velocityAtCenters(g, u, v);
  let s = 0;
  for (let k = 0; k < cu.length; k++) s += cu[k] * cu[k] + cv[k] * cv[k];
  return 0.5 * s / cu.length;
}

/** Total enstrophy per unit area, ½⟨ω²⟩. The second 2D invariant. */
export function enstrophy(g, u, v) {
  const w = vorticityCenter(g, u, v);
  let s = 0;
  for (let k = 0; k < w.length; k++) s += w[k] * w[k];
  return 0.5 * s / w.length;
}

/** Max |divergence|, the direct measure of how well the projection worked. */
export function maxDivergence(g, u, v) {
  const d = divergence(g, u, v);
  let m = 0;
  for (let k = 0; k < d.length; k++) m = Math.max(m, Math.abs(d[k]));
  return m;
}

/** L2 norm of divergence, the residual the pressure solvers are driving down. */
export function l2Divergence(g, u, v) {
  const d = divergence(g, u, v);
  let s = 0;
  for (let k = 0; k < d.length; k++) s += d[k] * d[k];
  return Math.sqrt(s / d.length);
}

/** Peak advective CFL number, |u|max·dt/h. */
export function cflNumber(g, u, v, dt) {
  let mu = 0, mv = 0;
  for (let k = 0; k < u.length; k++) mu = Math.max(mu, Math.abs(u[k]));
  for (let k = 0; k < v.length; k++) mv = Math.max(mv, Math.abs(v[k]));
  return dt * (mu / g.dx + mv / g.dy);
}
