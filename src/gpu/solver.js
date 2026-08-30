/**
 * The WebGPU backend: the PDE solve itself running on GPU threads.
 *
 * Every stage of the timestep -- advection, diffusion, divergence, the
 * iterative pressure solve, the gradient subtraction -- is a compute dispatch.
 * Nothing about the solve happens on the CPU; the CPU only issues commands and,
 * for diagnostics, occasionally reads a texture back.
 *
 * PRECISION, stated up front because it bounds every comparison made later.
 * WGSL is f32, and its transcendental functions are permitted relaxed accuracy
 * (a plain `sin` was measured at 6.9e-5 max absolute error on this adapter).
 * The CPU reference is f64. So GPU and CPU cannot agree below roughly 1e-5
 * relative, and any claim tighter than that would be measuring nothing. The
 * agreement test targets ~1e-4 after a handful of steps, which is what f32
 * accumulation over a stencil actually delivers.
 *
 * THE PRESSURE SOLVE is where the interesting parallel work is. Three methods
 * are implemented so the comparison is measurable rather than asserted:
 *
 *   jacobi          embarrassingly parallel, slow to converge
 *   red-black GS    two dispatches per sweep, ~2x Jacobi's rate
 *   multigrid       V-cycle, attacks low-frequency error directly
 *
 * The first two converge slowly for a reason worth stating: their error
 * reduction per sweep depends on the wavelength of the error, and the pressure
 * field's low-frequency modes -- the ones that span the whole domain -- are
 * reduced by a factor approaching 1 per sweep as the grid refines. That is not
 * a tuning problem, it is the method, and it is exactly what multigrid exists
 * to fix by solving those modes on a coarse grid where they are no longer
 * low-frequency.
 */

import {
  COMMON, SAMPLING, ADVECT_VELOCITY, ADVECT_SCALAR, DIVERGENCE,
  JACOBI, RED_BLACK_GS, SUBTRACT_GRADIENT, DIFFUSE, VORTICITY_CONFINEMENT,
  SPLAT, RENDER, FORCING, ADVECT_FIELD_BY, MACCORMACK_COMBINE, SOLID_COUPLE,
  BLOOM_BRIGHT, BLOOM_BLUR,
} from "./shaders.js";
import { Multigrid } from "./multigrid.js";

const WG = 8;   // workgroup is 8x8; 64 threads suits this iGPU's execution width

export async function createDevice({ requireTimestamps = false } = {}) {
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");

  const wanted = [];
  if (adapter.features.has("float32-filterable")) wanted.push("float32-filterable");
  if (adapter.features.has("timestamp-query")) wanted.push("timestamp-query");
  else if (requireTimestamps) throw new Error("timestamp-query unavailable; cannot measure kernel time");

  const device = await adapter.requestDevice({ requiredFeatures: wanted });
  return {
    device, adapter,
    hasTimestamps: wanted.includes("timestamp-query"),
    info: adapter.info
      ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
          device: adapter.info.device, description: adapter.info.description }
      : null,
  };
}

export class GPUFluidSolver {
  constructor(device, {
    n = 256, nx = n, ny = n, lx = 2 * Math.PI, ly = 2 * Math.PI,
    nu = 0.0, confinement = 0, dyeFade = 0,
    advection = "maccormack",
    solidMaskWidth = 1.5,
    pressureSolver = "jacobi", pressureIterations = 40,
    hasTimestamps = false,
  } = {}) {
    this.device = device;
    this.nx = nx; this.ny = ny;
    this.dx = lx / nx; this.dy = ly / ny;
    this.nu = nu;
    this.advection = advection;
    this.solidMaskWidth = solidMaskWidth;
    this.solid = null;
    this._solidBusy = false;
    this.confinement = confinement;
    this.dyeFade = dyeFade;
    this.pressureSolver = pressureSolver;
    this.pressureIterations = pressureIterations;
    this.hasTimestamps = hasTimestamps;
    this.t = 0;
    this.steps = 0;

    this._buildResources();
    this._buildPipelines();

    // Built only on demand: the hierarchy costs ~1.33x the fine grid in
    // textures, and a solver being benchmarked with a flat sweeper should not
    // pay for it.
    this.mg = pressureSolver === "multigrid"
      ? new Multigrid(this, { preSmooth: 2, postSmooth: 2, coarseSweeps: 40 })
      : null;
  }

  // ------------------------------------------------------------- resources

  _tex(format = "rgba32float", extraUsage = 0) {
    return this.device.createTexture({
      size: [this.nx, this.ny],
      format,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
             GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | extraUsage,
    });
  }

  _buildResources() {
    const d = this.device;

    // Ping-pong pairs. A stencil kernel cannot read and write one texture:
    // WGSL forbids read_write on a storage texture, and even without that
    // restriction a neighbour read while other threads write is a race whose
    // result depends on dispatch order.
    this.vel = [this._tex(), this._tex()];
    this.pressure = [this._tex(), this._tex()];
    this.dye = [this._tex(), this._tex()];
    this.div = this._tex();
    // MacCormack round-trip scratch, allocated only when that scheme is used.
    this.mcFwd = null; this.mcBack = null;
    // Immersed-solid coupling scratch, allocated only if a solid is attached.
    this.impulseTex = null; this.solidBuf = null; this.impulseRead = null;
    this.velIdx = 0; this.presIdx = 0; this.dyeIdx = 0;

    // HDR, not rgba8unorm. The bloom chain thresholds on luminance, and
    // thresholding an already tone-mapped 8-bit image finds a compressed version
    // of the highlights -- the glow comes out grey. Keeping the render linear and
    // letting values exceed 1.0 is what lets bright regions actually radiate.
    // Tone mapping and gamma happen once, in the final blit.
    this.output = this._tex("rgba16float");

    // Bloom works at half resolution: a quarter of the pixels per pass, and the
    // blur is wide enough that the lost detail is invisible.
    this.bloomW = Math.max(1, this.nx >> 1);
    this.bloomH = Math.max(1, this.ny >> 1);
    const mkBloom = () => this.device.createTexture({
      size: [this.bloomW, this.bloomH], format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.bloom = [mkBloom(), mkBloom()];

    // A RING OF UNIFORM SLOTS, not a single uniform buffer.
    //
    // This is the fix for a bug that made the whole pressure solve inert, and
    // it is worth stating precisely because the symptom looked like a physics
    // problem rather than an API one.
    //
    // `queue.writeBuffer` is ordered against SUBMITTED command buffers, not
    // against commands being encoded. Encoding a pass does not capture the
    // uniform's current contents -- the pass reads the buffer when it executes,
    // which is after every writeBuffer issued before the submit has landed. So
    // the original code, which wrote params, encoded a pass, wrote params
    // again, encoded another pass, and submitted once at the end, had every
    // pass in the step read the SAME final value.
    //
    // The consequences were invisible individually and fatal together: the
    // Jacobi relaxation factor was whatever the last write set (zero), so
    // `p_new = p + 0*(jacobi - p)` left the pressure untouched and the
    // projection subtracted a zero gradient; and the red-black colour flag was
    // always 1, so red cells were never updated and the iteration froze at
    // 6.13e-1 from the first sweep onwards.
    //
    // Both looked like a badly converging solver. Each pass now binds its own
    // 256-byte-aligned slot, so its parameters cannot be overwritten by a later
    // write in the same frame.
    this.uniformStride = Math.max(
      256, this.device.limits?.minUniformBufferOffsetAlignment ?? 256);
    this.uniformSlots = 4096;
    this.params = d.createBuffer({
      size: this.uniformStride * this.uniformSlots,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._slot = 0;
    // A RING, for exactly the reason the params buffer is one.
    //
    // This buffer was left as a single slot when `params` was fixed, and it has
    // the same defect: every splat pass in a command buffer read the LAST
    // splat's uniform, including its `isDye` flag. With ambient emitters
    // pushing a dye splat and then a velocity splat each frame, the frame always
    // ended on a velocity splat -- so the dye passes ran the velocity branch and
    // wrote `vec4(u, v, 0, 0)` into the DYE texture.
    //
    // It showed as green: those cells had red and green from the velocity
    // components and blue EXACTLY zero, and no colour in the palette has blue
    // exactly zero. 69% of lit dye cells were green-dominant.
    this.splatSlots = 256;
    this.splatParams = d.createBuffer({
      size: this.uniformStride * this.splatSlots,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._splatSlot = 0;
    // Readback staging buffer, allocated once. Creating one per readback made
    // the diagnostics dominate the frame time they were trying to measure.
    this.readback = d.createBuffer({
      size: this.nx * this.ny * 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    if (this.hasTimestamps) {
      this.querySet = d.createQuerySet({ type: "timestamp", count: 16 });
      this.queryResolve = d.createBuffer({
        size: 16 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.queryRead = d.createBuffer({
        size: 16 * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }
  }

  _pipeline(body) {
    const module = this.device.createShaderModule({ code: COMMON + SAMPLING + body });
    return this.device.createComputePipeline({
      layout: "auto", compute: { module, entryPoint: "main" },
    });
  }

  _buildPipelines() {
    this.pipe = {
      advectVel: this._pipeline(ADVECT_VELOCITY),
      advectScalar: this._pipeline(ADVECT_SCALAR),
      divergence: this._pipeline(DIVERGENCE),
      jacobi: this._pipeline(JACOBI),
      redBlack: this._pipeline(RED_BLACK_GS),
      subtractGrad: this._pipeline(SUBTRACT_GRADIENT),
      diffuse: this._pipeline(DIFFUSE),
      confinement: this._pipeline(VORTICITY_CONFINEMENT),
      splat: this._pipeline(SPLAT),
      render: this._pipeline(RENDER),
      forcing: this._pipeline(FORCING),
      advectFieldBy: this._pipeline(ADVECT_FIELD_BY),
      macCombine: this._pipeline(MACCORMACK_COMBINE),
      solidCouple: this._pipeline(SOLID_COUPLE),
      bloomBright: this._pipeline(BLOOM_BRIGHT),
      bloomBlur: this._pipeline(BLOOM_BLUR),
    };
  }

  /**
   * Write one parameter set into a fresh slot and return its byte offset.
   *
   * Private on purpose: callers use `_bind`, which allocates the slot and binds
   * it in one step. Separating "write the params" from "bind the params" is
   * what allowed the two to drift apart in the first place, so the API no
   * longer offers the chance.
   */
  _writeParams(dt, { paramA = 0, paramB = 0 } = {}) {
    const buf = new ArrayBuffer(32);
    new Uint32Array(buf, 0, 2).set([this.nx, this.ny]);
    new Float32Array(buf, 8, 6).set([this.dx, this.dy, dt, this.nu, paramA, paramB]);
    const offset = (this._slot % this.uniformSlots) * this.uniformStride;
    this._slot++;
    this.device.queue.writeBuffer(this.params, offset, buf);
    return offset;
  }

  /** Reset the slot ring. Called once at the start of each submitted frame. */
  _resetSlots() { this._slot = 0; }

  /**
   * Bind a pipeline together with its OWN parameter slot.
   *
   * Taking the parameters here rather than relying on a previous `_writeParams`
   * call is the structural half of the uniform-aliasing fix: a pass cannot be
   * encoded without simultaneously reserving the slot it reads, so no later
   * write in the same command buffer can overwrite it.
   */
  _bind(pipeline, params, entries) {
    const offset = this._writeParams(params.dt ?? 0, params);
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params, offset, size: 32 } },
        ...entries,
      ],
    });
  }

  _dispatch(pass, pipeline, bindGroup) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.nx / WG), Math.ceil(this.ny / WG));
  }

  get velCur() { return this.vel[this.velIdx]; }
  get velNext() { return this.vel[1 - this.velIdx]; }
  get presCur() { return this.pressure[this.presIdx]; }
  get presNext() { return this.pressure[1 - this.presIdx]; }
  get dyeCur() { return this.dye[this.dyeIdx]; }
  get dyeNext() { return this.dye[1 - this.dyeIdx]; }

  _swapVel() { this.velIdx = 1 - this.velIdx; }
  _swapPres() { this.presIdx = 1 - this.presIdx; }
  _swapDye() { this.dyeIdx = 1 - this.dyeIdx; }

  // ------------------------------------------------------------------ upload

  /** Upload a MAC velocity field from CPU arrays (Float64 or Float32). */
  uploadVelocity(u, v) {
    const data = new Float32Array(this.nx * this.ny * 4);
    for (let j = 0; j < this.ny; j++) {
      for (let i = 0; i < this.nx; i++) {
        const k = (j * this.nx + i) * 4;
        data[k] = u[j * this.nx + i];
        data[k + 1] = v[j * this.nx + i];
      }
    }
    this.device.queue.writeTexture(
      { texture: this.velCur }, data,
      { bytesPerRow: this.nx * 16, rowsPerImage: this.ny },
      [this.nx, this.ny],
    );
  }

  uploadDye(rgb) {
    this.device.queue.writeTexture(
      { texture: this.dyeCur }, rgb,
      { bytesPerRow: this.nx * 16, rowsPerImage: this.ny },
      [this.nx, this.ny],
    );
  }

  /**
   * Read a texture back to the CPU as a Float32Array of vec4s.
   *
   * Only valid for rgba32float. The bytes-per-row here is 16 per texel, and
   * passing an rgba16float texture (such as `output`, since the render path
   * went HDR) silently reinterprets half-floats as floats and returns numbers
   * that look plausible and mean nothing -- it reported render values of 92
   * where the shader cannot produce more than about 7. Guarded rather than
   * generalised: nothing needs to read the half-float targets back, and a
   * wrong answer is worse than a refusal.
   */
  async readTexture(tex) {
    if (tex.format !== "rgba32float") {
      throw new Error(
        `readTexture expects rgba32float, got ${tex.format}; ` +
        "the 16-byte stride would misread it");
    }
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: tex },
      { buffer: this.readback, bytesPerRow: this.nx * 16, rowsPerImage: this.ny },
      [this.nx, this.ny],
    );
    this.device.queue.submit([enc.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    return out;
  }

  /** Read the velocity field back as separate u and v arrays. */
  async readVelocity() {
    const raw = await this.readTexture(this.velCur);
    const n = this.nx * this.ny;
    const u = new Float32Array(n), v = new Float32Array(n);
    for (let i = 0; i < n; i++) { u[i] = raw[i * 4]; v[i] = raw[i * 4 + 1]; }
    return { u, v };
  }

  /**
   * Install a band-limited forcing mode set. Pass null to disable.
   * @param {Array<{kx:number,ky:number,phase:number}>} modes
   * @param {number} amplitude
   * @param {number} drag large-scale linear friction coefficient
   */
  setForcing(modes, { amplitude = 1, drag = 0, phaseDrift = 0.3 } = {}) {
    this.forcingModes?.destroy?.();
    if (!modes || modes.length === 0) { this.forcingModes = null; return; }
    const data = new Float32Array(modes.length * 4);
    modes.forEach((m, i) => {
      data[i * 4] = m.kx; data[i * 4 + 1] = m.ky; data[i * 4 + 2] = m.phase;
    });
    this.forcingModes = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.forcingModes, 0, data);
    this.forcingAmp = amplitude;
    this.forcingDrag = drag;
    this.forcingPhaseDrift = phaseDrift;
  }

  _encodeForcing(enc, dt) {
    if (!this.forcingModes) return;
    // `nu` in the params block doubles as the drag coefficient for this kernel,
    // and param_b carries the slow phase drift that decorrelates the forcing.
    const buf = new ArrayBuffer(32);
    new Uint32Array(buf, 0, 2).set([this.nx, this.ny]);
    new Float32Array(buf, 8, 6).set([
      this.dx, this.dy, dt, this.forcingDrag,
      this.forcingAmp, this.forcingPhaseDrift * this.t,
    ]);
    const offset = (this._slot % this.uniformSlots) * this.uniformStride;
    this._slot++;
    this.device.queue.writeBuffer(this.params, offset, buf);

    const bg = this.device.createBindGroup({
      layout: this.pipe.forcing.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params, offset, size: 32 } },
        { binding: 1, resource: { buffer: this.forcingModes } },
        { binding: 2, resource: this.velCur.createView() },
        { binding: 3, resource: this.velNext.createView() },
      ],
    });
    const pass = enc.beginComputePass();
    this._dispatch(pass, this.pipe.forcing, bg);
    pass.end();
    this._swapVel();
  }


  /**
   * Attach a single immersed rigid disk. Pass null to remove it.
   *
   * Deliberately one body rather than a list: the force integration needs a
   * readback per body per step, so N bodies cost N readbacks, and a demo with
   * one obstacle is the honest scope for this path. The CPU solver takes an
   * array and is where multi-body work would go.
   */
  setSolid(disk) {
    this.solid = disk || null;
    if (!disk) return;
    if (!this.impulseTex) {
      this.impulseTex = this._tex();
      this.solidBuf = this.device.createBuffer({
        size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.impulseRead = this.device.createBuffer({
        size: this.nx * this.ny * 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }
  }

  _encodeSolid(enc, dt) {
    const d = this.solid;
    if (!d) return;

    const sb = new ArrayBuffer(32);
    new Float32Array(sb).set([
      d.x, d.y, d.vx, d.vy, d.omega, d.r, this.solidMaskWidth * this.dx, 0,
    ]);
    this.device.queue.writeBuffer(this.solidBuf, 0, sb);

    const offset = this._writeParams(dt);
    const bg = this.device.createBindGroup({
      layout: this.pipe.solidCouple.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params, offset, size: 32 } },
        { binding: 1, resource: { buffer: this.solidBuf } },
        { binding: 2, resource: this.velCur.createView() },
        { binding: 3, resource: this.velNext.createView() },
        { binding: 4, resource: this.impulseTex.createView() },
      ],
    });
    const pass = enc.beginComputePass();
    this._dispatch(pass, this.pipe.solidCouple, bg);
    pass.end();
    this._swapVel();
  }

  /**
   * Sum the impulse field and advance the body.
   *
   * Async and one step behind: the readback cannot complete inside the step that
   * produced it without stalling the pipeline, so the force applied here comes
   * from the previous step's field. That extra lag makes the explicit coupling
   * slightly less stable than the CPU version, which is why the demo uses a
   * heavy disk. Stated rather than hidden -- the CPU path in src/cpu/solid.js is
   * the one the report's numbers come from, and it has no such lag.
   */
  async integrateSolid(dt) {
    const d = this.solid;
    if (!d || this._solidBusy) return;
    this._solidBusy = true;
    try {
      const enc = this.device.createCommandEncoder();
      enc.copyTextureToBuffer(
        { texture: this.impulseTex },
        { buffer: this.impulseRead, bytesPerRow: this.nx * 16, rowsPerImage: this.ny },
        [this.nx, this.ny],
      );
      this.device.queue.submit([enc.finish()]);
      await this.impulseRead.mapAsync(GPUMapMode.READ);
      const raw = new Float32Array(this.impulseRead.getMappedRange());
      const dA = this.dx * this.dy;
      let fx = 0, fy = 0, tq = 0;
      for (let i = 0; i < this.nx * this.ny; i++) {
        fx += raw[i * 4]; fy += raw[i * 4 + 1]; tq += raw[i * 4 + 2];
      }
      this.impulseRead.unmap();

      // Newton's third law: the body gets the negative of what the fluid got.
      d.force = [-fx * dA / dt, -fy * dA / dt];
      d.torque = -tq * dA / dt;
      d.integrate(dt);

      // Keep the body inside the periodic box so the mask wraps correctly.
      const L = this.nx * this.dx;
      d.x = ((d.x % L) + L) % L;
      d.y = ((d.y % L) + L) % L;
    } finally {
      this._solidBusy = false;
    }
  }

  // -------------------------------------------------------------- the step

  /**
   * One timestep. Same ordering as the CPU solver: advect, diffuse, force,
   * project, then advect dye through the now-divergence-free field.
   */
  step(dt, { splats = [] } = {}) {
    const d = this.device;
    this._resetSlots();
    this._splatSlot = 0;
    const enc = d.createCommandEncoder();
    let pass;

    // --- advect velocity
    this._encodeAdvection(enc, dt);

    // --- diffuse (explicit; only stable for nu*dt/h^2 <= 1/4, and the demo
    //     runs far below that. Kept explicit because an implicit solve here
    //     would need a second iterative system per step for a term that is
    //     negligible at these viscosities.)
    if (this.nu > 0) {
      pass = enc.beginComputePass();
      this._dispatch(pass, this.pipe.diffuse, this._bind(this.pipe.diffuse, { dt }, [
        { binding: 1, resource: this.velCur.createView() },
        { binding: 2, resource: this.velNext.createView() },
      ]));
      pass.end();
      this._swapVel();
    }

    // --- band-limited forcing + large-scale drag (turbulence runs)
    this._encodeForcing(enc, dt);

    // --- interaction forces
    for (const s of splats) {
      this._writeSplat(s);
      pass = enc.beginComputePass();
      this._dispatch(pass, this.pipe.splat, this._bindSplat(
        s.isDye ? this.dyeCur : this.velCur,
        s.isDye ? this.dyeNext : this.velNext,
        dt,
      ));
      pass.end();
      if (s.isDye) this._swapDye(); else this._swapVel();
    }

    // --- vorticity confinement (demo only; never in a measured run)
    if (this.confinement > 0) {
      pass = enc.beginComputePass();
      this._dispatch(pass, this.pipe.confinement, this._bind(this.pipe.confinement,
        { dt, paramA: this.confinement }, [
        { binding: 1, resource: this.velCur.createView() },
        { binding: 2, resource: this.velNext.createView() },
      ]));
      pass.end();
      this._swapVel();
    }

    // --- immersed solid, BEFORE the projection: direct forcing introduces
    //     divergence that the projection then clears.
    this._encodeSolid(enc, dt);

    // --- projection
    this._encodeProjection(enc, dt);

    // --- dye
    pass = enc.beginComputePass();
    this._dispatch(pass, this.pipe.advectScalar, this._bind(this.pipe.advectScalar,
      { dt, paramA: this.dyeFade }, [
      { binding: 1, resource: this.velCur.createView() },
      { binding: 2, resource: this.dyeCur.createView() },
      { binding: 3, resource: this.dyeNext.createView() },
    ]));
    pass.end();
    this._swapDye();

    d.queue.submit([enc.finish()]);
    this.t += dt;
    this.steps++;
  }

  /**
   * Velocity self-advection: semi-Lagrangian, or MacCormack when selected.
   *
   * MacCormack costs three passes instead of one and is roughly 36x less
   * dissipative on the CPU measurements. It matters here for two reasons that
   * are really the same reason: the energy spectrum's enstrophy range came out
   * at -6.1 against Kraichnan's -3 because the first-order scheme removes energy
   * across the whole resolved band, and the demo needed vorticity confinement
   * cranked up to show any small-scale structure at all -- which then amplified
   * the grid-scale mode into visible speckle.
   */
  _encodeAdvection(enc, dt) {
    let pass;
    if (this.advection === "maccormack") {
      if (!this.mcFwd) { this.mcFwd = this._tex(); this.mcBack = this._tex(); }

      // forward
      pass = enc.beginComputePass();
      this._dispatch(pass, this.pipe.advectVel, this._bind(this.pipe.advectVel, { dt }, [
        { binding: 1, resource: this.velCur.createView() },
        { binding: 2, resource: this.mcFwd.createView() },
      ]));
      pass.end();

      // backward FROM the forward result, through the ORIGINAL velocity
      pass = enc.beginComputePass();
      this._dispatch(pass, this.pipe.advectFieldBy,
        this._bind(this.pipe.advectFieldBy, { dt: -dt }, [
          { binding: 1, resource: this.velCur.createView() },
          { binding: 2, resource: this.mcFwd.createView() },
          { binding: 3, resource: this.mcBack.createView() },
        ]));
      pass.end();

      // combine + limit
      pass = enc.beginComputePass();
      this._dispatch(pass, this.pipe.macCombine, this._bind(this.pipe.macCombine, { dt }, [
        { binding: 1, resource: this.velCur.createView() },
        { binding: 2, resource: this.mcFwd.createView() },
        { binding: 3, resource: this.mcBack.createView() },
        { binding: 4, resource: this.velNext.createView() },
      ]));
      pass.end();
      this._swapVel();
      return;
    }

    pass = enc.beginComputePass();
    this._dispatch(pass, this.pipe.advectVel, this._bind(this.pipe.advectVel, { dt }, [
      { binding: 1, resource: this.velCur.createView() },
      { binding: 2, resource: this.velNext.createView() },
    ]));
    pass.end();
    this._swapVel();
  }

  /**
   * Divergence, iterate the Poisson solve, subtract the gradient.
   *
   * The pressure field is zeroed each step rather than warm-started from the
   * previous one. Warm-starting is tempting and does help, but it makes the
   * residual-vs-iteration measurement meaningless -- iteration 1 would already
   * carry the accumulated work of every previous step, so the curve would
   * describe the history rather than the solver.
   */
  _encodeProjection(enc, dt) {
    let pass = enc.beginComputePass();
    this._dispatch(pass, this.pipe.divergence, this._bind(this.pipe.divergence, { dt }, [
      { binding: 1, resource: this.velCur.createView() },
      { binding: 2, resource: this.div.createView() },
    ]));
    pass.end();

    // Zero the pressure. clearBuffer has no texture equivalent, so a Jacobi
    // sweep from a zeroed source is used -- writeTexture of a full zero array
    // every step would move nx*ny*16 bytes over PCIe for no reason.
    this.device.queue.writeTexture(
      { texture: this.presCur }, new Float32Array(this.nx * this.ny * 4),
      { bytesPerRow: this.nx * 16, rowsPerImage: this.ny }, [this.nx, this.ny],
    );

    this._encodePoisson(enc, dt, this.pressureIterations);

    pass = enc.beginComputePass();
    this._dispatch(pass, this.pipe.subtractGrad, this._bind(this.pipe.subtractGrad, { dt }, [
      { binding: 1, resource: this.velCur.createView() },
      { binding: 2, resource: this.presCur.createView() },
      { binding: 3, resource: this.velNext.createView() },
    ]));
    pass.end();
    this._swapVel();
  }

  _encodePoisson(enc, dt, iterations) {
    if (this.pressureSolver === "multigrid") {
      // `iterations` counts V-CYCLES here, not sweeps. A V-cycle costs roughly
      // 4/3 of one fine-grid sweep but reduces error at every wavelength, so
      // the two numbers are not comparable and the report states which is which.
      this.mg.clearSolution(enc);
      // Zero cycles means zero cycles. Forcing a minimum of one made the
      // convergence curve's first point read 6.4e-3 when it should read 1.0,
      // which quietly hid one V-cycle's worth of work at the origin.
      for (let k = 0; k < iterations; k++) {
        this.mg.encodeVCycle(enc, this.div);
      }
      // Copy the multigrid solution into the pressure texture the gradient
      // subtraction reads.
      enc.copyTextureToTexture(
        { texture: this.mg.solution }, { texture: this.presCur },
        [this.nx, this.ny],
      );
      return;
    }
    if (this.pressureSolver === "red-black") {
      for (let k = 0; k < iterations; k++) {
        // Two dispatches per sweep, one per colour. Every cell's four
        // neighbours have the opposite parity on this stencil, so a colour can
        // be updated entirely in parallel using only the other colour.
        for (const colour of [0, 1]) {
          // Each colour reserves its OWN slot. Sharing one is what froze
          // this iteration at its first-sweep residual.
          const pass = enc.beginComputePass();
          this._dispatch(pass, this.pipe.redBlack, this._bind(this.pipe.redBlack,
            { dt, paramA: 1.0, paramB: colour }, [
            { binding: 1, resource: this.presCur.createView() },
            { binding: 2, resource: this.div.createView() },
            { binding: 3, resource: this.presNext.createView() },
          ]));
          pass.end();
          this._swapPres();
        }
      }
    } else {
      for (let k = 0; k < iterations; k++) {
        const pass = enc.beginComputePass();
        this._dispatch(pass, this.pipe.jacobi, this._bind(this.pipe.jacobi,
          { dt, paramA: 1.0 }, [
          { binding: 1, resource: this.presCur.createView() },
          { binding: 2, resource: this.div.createView() },
          { binding: 3, resource: this.presNext.createView() },
        ]));
        pass.end();
        this._swapPres();
      }
    }
  }

  // ------------------------------------------------------------- interaction

  /** Write one splat into a fresh slot; returns its byte offset. */
  _writeSplat(s) {
    const buf = new ArrayBuffer(48);
    const f = new Float32Array(buf);
    f[0] = s.x; f[1] = s.y;
    f[2] = s.dx || 0; f[3] = s.dy || 0;
    f[4] = s.r || 0; f[5] = s.g || 0; f[6] = s.b || 0; f[7] = 0;
    f[8] = s.radius || 0.2;
    f[9] = s.isDye ? 1 : 0;
    const offset = (this._splatSlot % this.splatSlots) * this.uniformStride;
    this._splatSlot++;
    this.device.queue.writeBuffer(this.splatParams, offset, buf);
    return offset;
  }

  _bindSplat(src, dst, dt, splatOffset) {
    const offset = this._writeParams(dt);
    return this.device.createBindGroup({
      layout: this.pipe.splat.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params, offset, size: 32 } },
        { binding: 1, resource: { buffer: this.splatParams, offset: splatOffset, size: 48 } },
        { binding: 2, resource: src.createView() },
        { binding: 3, resource: dst.createView() },
      ],
    });
  }

  /**
   * Render the current state, then build a bloom texture from it.
   *
   * Display only: this reads the simulation fields and produces pixels, and
   * nothing it computes is fed back into the solve.
   *
   * @param {number} bloomThreshold luminance above which a pixel glows
   * @param {number} bloomPasses    blur iterations; 2 is a wider, softer glow
   */
  render({ mode = 0, scale = 1.5, bloomThreshold = 0.75, bloomPasses = 2 } = {}) {
    this._resetSlots();
    const enc = this.device.createCommandEncoder();

    let pass = enc.beginComputePass();
    this._dispatch(pass, this.pipe.render, this._bind(this.pipe.render,
      { dt: 0, paramA: scale, paramB: mode }, [
        { binding: 1, resource: this.velCur.createView() },
        { binding: 2, resource: this.dyeCur.createView() },
        { binding: 3, resource: this.output.createView() },
      ]));
    pass.end();

    if (bloomPasses > 0) {
      // The bright pass reads the full-res render and writes the half-res
      // bloom texture, so every dispatch here is sized to the BLOOM texture
      // rather than the simulation grid.
      const wg = (n) => Math.ceil(n / WG);

      pass = enc.beginComputePass();
      pass.setPipeline(this.pipe.bloomBright);
      pass.setBindGroup(0, this._bind(this.pipe.bloomBright,
        { dt: 0, paramA: bloomThreshold }, [
          { binding: 1, resource: this.output.createView() },
          { binding: 2, resource: this.bloom[0].createView() },
        ]));
      pass.dispatchWorkgroups(wg(this.bloomW), wg(this.bloomH));
      pass.end();

      for (let k = 0; k < bloomPasses; k++) {
        for (const axis of [0, 1]) {
          const src = this.bloom[0], dst = this.bloom[1];
          pass = enc.beginComputePass();
          pass.setPipeline(this.pipe.bloomBlur);
          pass.setBindGroup(0, this._bind(this.pipe.bloomBlur,
            { dt: 0, paramB: axis }, [
              { binding: 1, resource: src.createView() },
              { binding: 2, resource: dst.createView() },
            ]));
          pass.dispatchWorkgroups(wg(this.bloomW), wg(this.bloomH));
          pass.end();
          this.bloom.reverse();
        }
      }
    }

    this.device.queue.submit([enc.finish()]);
    return this.output;
  }

  // -------------------------------------------------------------- diagnostics

  /**
   * Max |divergence| after the current state, computed on the GPU then read
   * back. The direct measure of whether the projection did its job.
   */
  async maxDivergence() {
    this._resetSlots();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this._dispatch(pass, this.pipe.divergence, this._bind(this.pipe.divergence, { dt: 0 }, [
      { binding: 1, resource: this.velCur.createView() },
      { binding: 2, resource: this.div.createView() },
    ]));
    pass.end();
    this.device.queue.submit([enc.finish()]);

    const raw = await this.readTexture(this.div);
    let m = 0;
    for (let i = 0; i < this.nx * this.ny; i++) m = Math.max(m, Math.abs(raw[i * 4]));
    return m;
  }

  /**
   * Residual of the pressure solve after `iterations` sweeps, for the
   * solver-comparison curve. Runs the whole projection from a zeroed pressure
   * so each point is independent of the others.
   */
  async poissonResidualAfter(iterations, dt = 0.01) {
    const d = this.device;
    this._resetSlots();
    let enc = d.createCommandEncoder();
    let pass = enc.beginComputePass();
    this._dispatch(pass, this.pipe.divergence, this._bind(this.pipe.divergence, { dt }, [
      { binding: 1, resource: this.velCur.createView() },
      { binding: 2, resource: this.div.createView() },
    ]));
    pass.end();
    d.queue.submit([enc.finish()]);

    d.queue.writeTexture(
      { texture: this.presCur }, new Float32Array(this.nx * this.ny * 4),
      { bytesPerRow: this.nx * 16, rowsPerImage: this.ny }, [this.nx, this.ny],
    );

    enc = d.createCommandEncoder();
    this._encodePoisson(enc, dt, iterations);
    d.queue.submit([enc.finish()]);

    // Residual ||lap(p) - div|| / ||div||, evaluated on the CPU from the two
    // fields. Doing the norm on the GPU would need a reduction kernel, and this
    // runs once per data point rather than once per step.
    const pRaw = await this.readTexture(this.presCur);
    const dRaw = await this.readTexture(this.div);
    const nx = this.nx, ny = this.ny, h2 = this.dx * this.dx;
    const at = (arr, i, j) => arr[(((j % ny) + ny) % ny * nx + ((i % nx) + nx) % nx) * 4];
    let num = 0, den = 0;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const lap = (at(pRaw, i - 1, j) + at(pRaw, i + 1, j) +
                     at(pRaw, i, j - 1) + at(pRaw, i, j + 1) - 4 * at(pRaw, i, j)) / h2;
        const r = lap - at(dRaw, i, j);
        num += r * r;
        den += at(dRaw, i, j) ** 2;
      }
    }
    return Math.sqrt(num) / (Math.sqrt(den) || 1);
  }

  destroy() {
    this.mg?.destroy();
    this.forcingModes?.destroy();
    this.mcFwd?.destroy(); this.mcBack?.destroy();
    this.impulseTex?.destroy(); this.solidBuf?.destroy();
    this.impulseRead?.destroy();
    this.bloom?.forEach(t => t.destroy());
    for (const t of [...this.vel, ...this.pressure, ...this.dye, this.div, this.output]) {
      t.destroy();
    }
    this.params.destroy();
    this.splatParams.destroy();
    this.readback.destroy();
  }
}
