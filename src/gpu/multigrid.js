/**
 * Geometric multigrid V-cycle for the pressure Poisson equation, on the GPU.
 *
 * WHY THIS IS NECESSARY, from measurement rather than doctrine.
 *
 * The benchmark that motivated this compared both devices at EQUAL SOLUTION
 * QUALITY rather than equal iteration count, and the GPU lost: to reach
 * max|div| <= 1e-3 it needed 160 red-black sweeps at 64^2 and could not get
 * there at all at 256^2 or 512^2 within 1280 sweeps. Meanwhile the CPU's exact
 * FFT solve did it in one pass. The GPU's 123x advantage at 512^2 was an
 * advantage at running a BAD ALGORITHM quickly.
 *
 * The reason Jacobi and Gauss-Seidel stall is specific and fixable. Both are
 * local: one sweep moves information one cell. Their error-reduction factor
 * depends on the error's wavelength, and for a mode spanning L cells it is
 * roughly 1 - O(1/L^2). High-frequency error dies almost immediately; the
 * smooth, domain-spanning component -- which is most of a pressure field --
 * barely moves. Refining the grid makes this strictly worse, because the same
 * physical wavelength now spans more cells.
 *
 * Multigrid's insight is that "low frequency" is relative to the grid. A mode
 * that is smooth on a 512^2 grid is oscillatory on a 32^2 grid, where a
 * relaxation sweep kills it. So:
 *
 *   1. SMOOTH a few sweeps on the fine grid  -> removes high-frequency error
 *   2. RESTRICT the remaining residual to a grid half as fine
 *   3. recurse, until the grid is small enough to solve almost exactly
 *   4. PROLONG the coarse correction back up and add it
 *   5. SMOOTH again to clean up interpolation error
 *
 * Each level costs a quarter of the one above, so the whole V-cycle costs about
 * 4/3 of one fine-grid sweep, and it reduces error across ALL wavelengths.
 * The result is convergence that does not degrade as the grid refines, which is
 * the property the flat sweepers lack.
 *
 * IMPLEMENTATION NOTE ON THE COARSE GRIDS. Restriction here is full-weighting
 * (a 2x2 average), and prolongation is bilinear. They are chosen as a matched
 * pair: prolongation is (up to a constant) the transpose of restriction, which
 * keeps the coarse-grid correction consistent with the fine-grid residual it
 * was computed from. Mismatched transfer operators produce a V-cycle that
 * converges slowly or stalls, and the symptom looks exactly like a bad
 * smoother.
 */

import { COMMON, SAMPLING } from "./shaders.js";

/** Residual r = rhs - laplacian(p), on the current level. */
export const RESIDUAL = /* wgsl */`
@group(0) @binding(1) var pressure : texture_2d<f32>;
@group(0) @binding(2) var rhs : texture_2d<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x); let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }
  let l = loadTex(pressure, i - 1, j).x;
  let r = loadTex(pressure, i + 1, j).x;
  let b = loadTex(pressure, i, j - 1).x;
  let t = loadTex(pressure, i, j + 1).x;
  let c = loadTex(pressure, i, j).x;
  let lap = (l + r + b + t - 4.0 * c) / (P.dx * P.dx);
  let res = loadTex(rhs, i, j).x - lap;
  textureStore(dst, vec2<i32>(i, j), vec4<f32>(res, 0.0, 0.0, 0.0));
}
`;

/**
 * Restriction: fine -> coarse by full-weighting over each 2x2 block.
 * P.nx/P.ny describe the COARSE (destination) grid.
 */
export const RESTRICT = /* wgsl */`
@group(0) @binding(1) var fine : texture_2d<f32>;
@group(0) @binding(2) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x); let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }
  // Wrapping must use the FINE dimensions, which are twice the coarse ones.
  let fnx = i32(P.nx) * 2; let fny = i32(P.ny) * 2;
  let i2 = i * 2; let j2 = j * 2;
  var s = 0.0;
  for (var dj = 0; dj < 2; dj = dj + 1) {
    for (var di = 0; di < 2; di = di + 1) {
      let x = ((i2 + di) % fnx + fnx) % fnx;
      let y = ((j2 + dj) % fny + fny) % fny;
      s = s + textureLoad(fine, vec2<i32>(x, y), 0).x;
    }
  }
  textureStore(dst, vec2<i32>(i, j), vec4<f32>(s * 0.25, 0.0, 0.0, 0.0));
}
`;

/**
 * Prolongation: coarse -> fine, bilinear, ADDED to the existing fine field.
 * P.nx/P.ny describe the FINE (destination) grid.
 */
export const PROLONG = /* wgsl */`
@group(0) @binding(1) var coarse : texture_2d<f32>;
@group(0) @binding(2) var fineIn : texture_2d<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x); let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  let cnx = i32(P.nx) / 2; let cny = i32(P.ny) / 2;

  // Fine cell centre (i+0.5) maps to coarse coordinate (i+0.5)/2 - 0.5.
  let gx = (f32(i) + 0.5) * 0.5 - 0.5;
  let gy = (f32(j) + 0.5) * 0.5 - 0.5;
  let i0 = i32(floor(gx)); let j0 = i32(floor(gy));
  let fx = gx - floor(gx); let fy = gy - floor(gy);

  let x0 = ((i0 % cnx) + cnx) % cnx;
  let x1 = (((i0 + 1) % cnx) + cnx) % cnx;
  let y0 = ((j0 % cny) + cny) % cny;
  let y1 = (((j0 + 1) % cny) + cny) % cny;

  let a = textureLoad(coarse, vec2<i32>(x0, y0), 0).x;
  let b = textureLoad(coarse, vec2<i32>(x1, y0), 0).x;
  let c = textureLoad(coarse, vec2<i32>(x0, y1), 0).x;
  let d = textureLoad(coarse, vec2<i32>(x1, y1), 0).x;
  let corr = mix(mix(a, b, fx), mix(c, d, fx), fy);

  let cur = loadTex(fineIn, i, j).x;
  textureStore(dst, vec2<i32>(i, j), vec4<f32>(cur + corr, 0.0, 0.0, 0.0));
}
`;

/** Zero a texture. Needed because each level's correction starts from zero. */
export const CLEAR = /* wgsl */`
@group(0) @binding(1) var dst : texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }
  textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 0.0));
}
`;

/**
 * Manages the grid hierarchy and encodes V-cycles.
 *
 * Attached to a GPUFluidSolver rather than built into it, so a solver can be
 * constructed without paying for the hierarchy when a flat sweeper is what is
 * being measured.
 */
export class Multigrid {
  /**
   * @param {GPUFluidSolver} solver
   * @param {object} opts
   * @param {number} opts.minSize stop coarsening at this grid size
   * @param {number} opts.preSmooth  sweeps before restricting
   * @param {number} opts.postSmooth sweeps after prolonging
   * @param {number} opts.coarseSweeps sweeps on the coarsest grid
   */
  constructor(solver, {
    minSize = 8, preSmooth = 2, postSmooth = 2, coarseSweeps = 40,
  } = {}) {
    this.s = solver;
    this.device = solver.device;
    this.preSmooth = preSmooth;
    this.postSmooth = postSmooth;
    this.coarseSweeps = coarseSweeps;

    const d = this.device;
    const mkTex = (n) => d.createTexture({
      size: [n, n], format: "rgba32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
             GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });

    // Levels: 0 is finest. Each holds a correction (x), a right-hand side, a
    // residual, and a scratch texture for the ping-pong inside smoothing.
    this.levels = [];
    let n = solver.nx;
    if (solver.nx !== solver.ny) {
      throw new Error("multigrid here assumes a square grid");
    }
    while (n >= minSize) {
      const isFinest = this.levels.length === 0;
      this.levels.push({
        n,
        dx: (solver.dx * solver.nx) / n,      // physical spacing at this level
        x: [mkTex(n), mkTex(n)],
        // The finest level's right-hand side is the divergence texture the
        // solver already owns, supplied to encodeVCycle. Allocating one here
        // too would orphan it the moment encodeVCycle overwrote the field.
        rhs: isFinest ? null : mkTex(n),
        res: mkTex(n),
        xIdx: 0,
      });
      if (n / 2 < minSize) break;
      n = n / 2;
    }

    this.pipe = {
      residual: this._pipeline(RESIDUAL),
      restrict: this._pipeline(RESTRICT),
      prolong: this._pipeline(PROLONG),
      clear: this._pipeline(CLEAR),
    };
  }

  get depth() { return this.levels.length; }

  _pipeline(body) {
    const module = this.device.createShaderModule({ code: COMMON + SAMPLING + body });
    return this.device.createComputePipeline({
      layout: "auto", compute: { module, entryPoint: "main" },
    });
  }

  /** Params for a given level; nx/ny/dx/dy vary per level. */
  _levelParams(lv, { paramA = 0, paramB = 0 } = {}) {
    const buf = new ArrayBuffer(32);
    new Uint32Array(buf, 0, 2).set([lv.n, lv.n]);
    new Float32Array(buf, 8, 6).set([lv.dx, lv.dx, 0, 0, paramA, paramB]);
    const offset = (this.s._slot % this.s.uniformSlots) * this.s.uniformStride;
    this.s._slot++;
    this.device.queue.writeBuffer(this.s.params, offset, buf);
    return offset;
  }

  _bind(pipeline, lv, params, entries) {
    const offset = this._levelParams(lv, params);
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.s.params, offset, size: 32 } },
        ...entries,
      ],
    });
  }

  _dispatch(pass, pipeline, bindGroup, n) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(n / 8));
  }

  _cur(lv) { return lv.x[lv.xIdx]; }
  _next(lv) { return lv.x[1 - lv.xIdx]; }
  _swap(lv) { lv.xIdx = 1 - lv.xIdx; }

  _clear(enc, lv, tex) {
    const pass = enc.beginComputePass();
    this._dispatch(pass, this.pipe.clear,
      this._bind(this.pipe.clear, lv, {}, [{ binding: 1, resource: tex.createView() }]), lv.n);
    pass.end();
  }

  /** Red-black Gauss-Seidel sweeps on one level, solving lap(x) = rhs. */
  _smooth(enc, lv, rhsTex, sweeps) {
    for (let k = 0; k < sweeps; k++) {
      for (const colour of [0, 1]) {
        const pass = enc.beginComputePass();
        this._dispatch(pass, this.s.pipe.redBlack,
          this._bind(this.s.pipe.redBlack, lv, { paramA: 1.0, paramB: colour }, [
            { binding: 1, resource: this._cur(lv).createView() },
            { binding: 2, resource: rhsTex.createView() },
            { binding: 3, resource: this._next(lv).createView() },
          ]), lv.n);
        pass.end();
        this._swap(lv);
      }
    }
  }

  /**
   * Encode one V-cycle. `rhsTex` is the fine-grid right-hand side (divergence),
   * and the solution accumulates in level 0's current texture -- which the
   * caller should have cleared, or warm-started deliberately.
   */
  encodeVCycle(enc, rhsTex) {
    const L = this.levels;
    L[0].rhs = rhsTex;

    // --- descend
    for (let i = 0; i < L.length - 1; i++) {
      const lv = L[i], next = L[i + 1];
      this._smooth(enc, lv, lv.rhs, this.preSmooth);

      // residual on this level
      let pass = enc.beginComputePass();
      this._dispatch(pass, this.pipe.residual,
        this._bind(this.pipe.residual, lv, {}, [
          { binding: 1, resource: this._cur(lv).createView() },
          { binding: 2, resource: lv.rhs.createView() },
          { binding: 3, resource: lv.res.createView() },
        ]), lv.n);
      pass.end();

      // restrict residual -> next level's rhs
      pass = enc.beginComputePass();
      this._dispatch(pass, this.pipe.restrict,
        this._bind(this.pipe.restrict, next, {}, [
          { binding: 1, resource: lv.res.createView() },
          { binding: 2, resource: next.rhs.createView() },
        ]), next.n);
      pass.end();

      // the coarse correction starts from zero
      this._clear(enc, next, this._cur(next));
      this._clear(enc, next, this._next(next));
    }

    // --- coarsest level: smooth hard. It is tiny, so this is nearly exact and
    // costs almost nothing.
    const last = L[L.length - 1];
    this._smooth(enc, last, last.rhs, this.coarseSweeps);

    // --- ascend
    for (let i = L.length - 2; i >= 0; i--) {
      const lv = L[i], below = L[i + 1];
      const pass = enc.beginComputePass();
      this._dispatch(pass, this.pipe.prolong,
        this._bind(this.pipe.prolong, lv, {}, [
          { binding: 1, resource: this._cur(below).createView() },
          { binding: 2, resource: this._cur(lv).createView() },
          { binding: 3, resource: this._next(lv).createView() },
        ]), lv.n);
      pass.end();
      this._swap(lv);

      this._smooth(enc, lv, lv.rhs, this.postSmooth);
    }
  }

  /** The finest-level solution texture, after encodeVCycle has run. */
  get solution() { return this._cur(this.levels[0]); }

  clearSolution(enc) {
    this._clear(enc, this.levels[0], this.levels[0].x[0]);
    this._clear(enc, this.levels[0], this.levels[0].x[1]);
  }

  destroy() {
    for (const lv of this.levels) {
      lv.x[0].destroy(); lv.x[1].destroy();
      lv.res.destroy();
      // The finest level's rhs is borrowed from the solver, never owned here.
      lv.rhs?.destroy();
    }
  }
}
