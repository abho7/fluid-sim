/**
 * Two-way fluid-structure interaction: an immersed rigid disk.
 *
 * The disk both blocks the flow and is pushed by it. "Two-way" is the load-
 * bearing word -- a static obstacle that deflects fluid without ever moving is
 * a boundary condition, not an interaction, and is much easier.
 *
 * METHOD: direct-forcing immersed boundary (Mohd-Yusof 1997, Fadlun et al.
 * 2000). Rather than meshing the disk or imposing a boundary condition on a
 * body-fitted grid, the solid is represented by a mask on the existing Cartesian
 * grid, and the fluid velocity inside the mask is driven toward the solid's
 * velocity each step:
 *
 *     u* = u + chi(x) * (U_solid(x) - u)
 *
 * The implied force per unit volume is f = chi*(U_s - u)/dt, and by Newton's
 * third law the reaction on the solid is the negative of its integral. That
 * pairing is what closes the loop, and it is also what makes the whole thing
 * checkable: applied correctly, a coupling step moves momentum between fluid and
 * solid without creating or destroying any. That is asserted to machine
 * precision in the tests rather than assumed from the formulas looking right.
 *
 * WHY THE MASK IS SMOOTHED. A hard 0/1 mask makes the disk a staircase of whole
 * cells. The force then jumps discontinuously as the disk moves across a cell
 * boundary, and a freely-moving disk visibly stutters at exactly the grid
 * spacing -- an artifact that is easy to mistake for turbulence acting on it.
 * Smoothing over about one and a half cells makes the force a continuous
 * function of position, at the cost of a slightly fuzzy surface.
 *
 * WHAT THIS IS NOT. This is a rigid body with prescribed mass, not an elastic
 * solid; there is no collision handling between multiple disks beyond the fluid
 * that separates them; and the no-slip condition is imposed to within the
 * accuracy of one forcing pass, not exactly (see `slipError`, which measures it
 * rather than hiding it).
 */

/** Shortest signed separation on a periodic axis of length L. */
function wrapDelta(d, L) {
  return d - L * Math.round(d / L);
}

export class RigidDisk {
  /**
   * @param {object} o
   * @param {number} o.x,o.y      centre
   * @param {number} o.r          radius
   * @param {number} o.density    relative to the fluid (1 = neutrally buoyant)
   * @param {boolean} o.fixed     if true the disk blocks flow but never moves,
   *                              i.e. one-way coupling -- kept as a control, so
   *                              the difference two-way coupling makes is
   *                              measurable rather than asserted
   */
  constructor({ x, y, r, density = 1, vx = 0, vy = 0, omega = 0, fixed = false,
                addedMassCorrection = false } = {}) {
    this.x = x; this.y = y;
    this.r = r;
    this.density = density;
    this.vx = vx; this.vy = vy;
    this.omega = omega;
    this.fixed = fixed;

    // Mass and moment of inertia of a uniform disk, with the fluid density
    // taken as 1 so `density` is the density ratio.
    this.mass = density * Math.PI * r * r;
    this.inertia = 0.5 * this.mass * r * r;

    // ADDED MASS, and why it is optional rather than always on.
    //
    // Accelerating a body through a fluid also accelerates the fluid around it,
    // so the body behaves as though heavier. In 2D the added mass of a disk is
    // exactly the mass of the fluid it displaces, rho*pi*R^2 -- which means at a
    // density ratio of 1 the added mass EQUALS the body's own mass.
    //
    // That is the source of a real instability in this scheme. The coupling is
    // explicit: the force is computed from the current fluid state and applied
    // to the body, which then moves and changes the fluid next step. When the
    // added mass rivals the body mass, that feedback loop has gain near or above
    // one and diverges. Measured here: stable down to a density ratio of 0.8,
    // divergent at 0.5 (a disk at density 0.25 reached |v| = 148 in 40 steps).
    //
    // Folding the added mass into the effective inertia is the standard cheap
    // stabilisation. It is left OFF by default so the uncorrected boundary can
    // be measured, and the sweep in validate/studies.js reports both.
    this.addedMassCorrection = addedMassCorrection;
    this.addedMass = Math.PI * r * r;          // fluid density = 1
    this.effectiveMass = this.mass + (addedMassCorrection ? this.addedMass : 0);
    this.effectiveInertia = this.inertia +
      (addedMassCorrection ? 0.5 * this.addedMass * r * r : 0);

    // Diagnostics, refreshed every step.
    this.force = [0, 0];
    this.torque = 0;
    this.lastImpulse = [0, 0];
  }

  /**
   * Smoothed indicator: 1 deep inside, 0 well outside, with a transition about
   * `width` wide. tanh rather than a linear ramp because its derivative is
   * continuous, which keeps the force smooth as the disk moves.
   */
  mask(px, py, lx, ly, width) {
    const dx = wrapDelta(px - this.x, lx);
    const dy = wrapDelta(py - this.y, ly);
    const d = Math.hypot(dx, dy) - this.r;
    return 0.5 * (1 - Math.tanh(d / width));
  }

  /** Rigid-body velocity at a point: translation plus rotation. */
  velocityAt(px, py, lx, ly) {
    const dx = wrapDelta(px - this.x, lx);
    const dy = wrapDelta(py - this.y, ly);
    // omega x r, in 2D: (-omega*dy, +omega*dx)
    return [this.vx - this.omega * dy, this.vy + this.omega * dx];
  }

  /** Advance the disk under the accumulated fluid reaction, plus gravity. */
  integrate(dt, { gravity = 0 } = {}) {
    if (this.fixed) return;
    // Gravity acts on the real mass; the fluid reaction is divided by the
    // effective (possibly added-mass-corrected) inertia. Dividing gravity by the
    // effective mass too would make a corrected body fall more slowly, which is
    // not what the correction means.
    this.vx += (this.force[0] / this.effectiveMass) * dt;
    this.vy += (this.force[1] / this.effectiveMass) * dt + gravity * dt;
    this.omega += (this.torque / this.effectiveInertia) * dt;
    this.x += this.vx * dt;
    this.y += this.vy * dt;
  }

  momentum() {
    return [this.mass * this.vx, this.mass * this.vy];
  }

  kineticEnergy() {
    return 0.5 * this.mass * (this.vx ** 2 + this.vy ** 2)
         + 0.5 * this.inertia * this.omega ** 2;
  }

  toJSON() {
    return {
      x: this.x, y: this.y, r: this.r, density: this.density,
      vx: this.vx, vy: this.vy, omega: this.omega, fixed: this.fixed,
      mass: this.mass, force: this.force, torque: this.torque,
    };
  }
}

/**
 * Apply one direct-forcing coupling step, in place.
 *
 * Returns the impulse delivered to the FLUID. The reaction applied to each
 * solid is the negative of its own contribution, which is where Newton's third
 * law actually lives in this code.
 *
 * ORDERING. This must run BEFORE the pressure projection. Direct forcing sets
 * the velocity inside the solid without regard to incompressibility, so it
 * generally introduces divergence; the projection then removes it. Running the
 * projection first and forcing afterwards would leave the field divergent for
 * the rest of the step, and the dye would visibly pool at the obstacle.
 *
 * The projection does partially relax the no-slip condition it just imposed --
 * that is inherent to this class of method, and `slipError` below measures how
 * much rather than pretending it does not happen.
 */
export function applyCoupling(g, u, v, solids, dt,
                              { maskWidth = 1.5, passes = 1 } = {}) {
  const { nx, ny, lx, ly } = g;
  const width = maskWidth * g.dx;
  // Cell area: the discrete integral element that turns a per-sample force into
  // a force. Getting this wrong scales every force by a constant and shows up
  // as a disk that is far too heavy or far too light.
  const dA = g.dx * g.dy;

  let impulseX = 0, impulseY = 0;

  // A single forcing pass leaves a residual slip of (1 - chi) times the velocity
  // difference -- at the mask's 0.98 contour that is 2%, which was measured
  // before this loop existed. Repeating the pass drives the residual down
  // geometrically: (1 - chi)^passes. Two or three passes is plenty, and each is
  // far cheaper than the pressure solve.
  for (let pass = 0; pass < passes; pass++) {
  for (const s of solids) {
    let fx = 0, fy = 0, tq = 0;

    // u-faces
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const px = g.uX(i), py = g.uY(j);
        const chi = s.mask(px, py, lx, ly, width);
        if (chi < 1e-6) continue;
        const k = g.idxU(i, j);
        const [su] = s.velocityAt(px, py, lx, ly);
        const du = chi * (su - u[k]);
        u[k] += du;
        fx += du * dA;
        // Torque from this face's force about the disk centre: r x F, and the
        // z-component of (dx,dy) x (F,0) is -dy*F.
        const dy = wrapDelta(py - s.y, ly);
        tq += -dy * du * dA;
      }
    }

    // v-faces
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const px = g.vX(i), py = g.vY(j);
        const chi = s.mask(px, py, lx, ly, width);
        if (chi < 1e-6) continue;
        const k = g.idxV(i, j);
        const [, sv] = s.velocityAt(px, py, lx, ly);
        const dv = chi * (sv - v[k]);
        v[k] += dv;
        fy += dv * dA;
        const dx = wrapDelta(px - s.x, lx);
        tq += dx * dv * dA;
      }
    }

    // These are momentum changes (impulses); dividing by dt gives forces, which
    // is what `integrate` expects.
    // Accumulate across passes so the reported force is the total the body
    // actually received, not just the last pass's share.
    if (pass === 0) { s.lastImpulse = [fx, fy]; s.force = [0, 0]; s.torque = 0; }
    else { s.lastImpulse = [s.lastImpulse[0] + fx, s.lastImpulse[1] + fy]; }
    s.force = [s.force[0] - fx / dt, s.force[1] - fy / dt];   // Newton's third law
    s.torque -= tq / dt;
    impulseX += fx;
    impulseY += fy;
  }
  }

  return [impulseX, impulseY];
}

/**
 * How far the fluid inside the solid departs from moving with it.
 *
 * The no-slip condition is imposed by a single forcing pass and then partially
 * relaxed by the projection, so it holds approximately. Reporting the residual
 * is the honest alternative to claiming the boundary is exact -- and it is the
 * number that tells you whether the mask is too thin or the disk is moving too
 * fast for the timestep.
 */
export function slipError(g, u, v, solid, { maskWidth = 1.5 } = {}) {
  const { nx, ny, lx, ly } = g;
  const width = maskWidth * g.dx;
  let worst = 0, sum = 0, n = 0;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const px = g.uX(i), py = g.uY(j);
      // Only deep interior: the transition band is smoothed by construction and
      // is not supposed to satisfy no-slip.
      if (solid.mask(px, py, lx, ly, width) < 0.98) continue;
      const [su] = solid.velocityAt(px, py, lx, ly);
      const e = Math.abs(u[g.idxU(i, j)] - su);
      worst = Math.max(worst, e);
      sum += e; n++;
    }
  }
  return { max: worst, mean: n ? sum / n : 0, samples: n };
}

/** Total fluid momentum on a periodic MAC grid, per unit density. */
export function fluidMomentum(g, u, v) {
  const dA = g.dx * g.dy;
  let px = 0, py = 0;
  for (let k = 0; k < u.length; k++) px += u[k];
  for (let k = 0; k < v.length; k++) py += v[k];
  return [px * dA, py * dA];
}
