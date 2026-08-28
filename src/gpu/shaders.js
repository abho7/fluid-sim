/**
 * WGSL compute kernels: the PDE solve itself, running on GPU threads.
 *
 * DATA LAYOUT. Every field is an rgba32float storage texture, one texel per
 * grid cell, with the MAC staggering carried in the .x channel:
 *
 *     velocity texture   .x = u on this cell's LEFT face
 *                        .y = v on this cell's BOTTOM face
 *     pressure texture   .x = p at the cell centre
 *     dye texture        .xyz = colour, .w unused
 *
 * Packing u and v into one texture rather than two costs nothing (both are
 * always read together) and halves the number of bindings, which matters
 * because the bind-group limit is tight once the multigrid levels are added.
 *
 * WHY TEXTURES AND NOT BUFFERS. Semi-Lagrangian advection is a gather from an
 * arbitrary non-grid-aligned position, i.e. exactly a filtered texture fetch.
 * `float32-filterable` is available on this adapter (checked before any of this
 * was written), and the hardware bilinear unit was verified to produce exactly
 * the same weights as the CPU `bilerp` -- sampling halfway between texels 1 and
 * 2 returned 1.5, and quarter-way returned 1.25, to f32 exactness. That match is
 * what makes a field-by-field CPU/GPU comparison meaningful rather than
 * approximate.
 *
 * PING-PONG. WGSL cannot read and write the same storage texture in one
 * dispatch, and even if it could, a stencil operation reading neighbours that
 * other threads are concurrently overwriting is a race. Every kernel reads
 * `src` and writes `dst`, and the caller swaps.
 *
 * PERIODIC WRAPPING is done in the shader with modular arithmetic rather than
 * by a repeat-address sampler, because the same wrap has to apply to
 * `textureLoad` (integer, unsampled) as to `textureSampleLevel`, and having two
 * different wrapping mechanisms is how a half-texel seam appears along one edge.
 */

export const COMMON = /* wgsl */`
struct Params {
  nx      : u32,
  ny      : u32,
  dx      : f32,
  dy      : f32,
  dt      : f32,
  nu      : f32,
  param_a : f32,   // kernel-specific scalar (confinement eps, drag, ...)
  param_b : f32,
};

@group(0) @binding(0) var<uniform> P : Params;

fn wrapi(i: i32, n: i32) -> i32 {
  return ((i % n) + n) % n;
}

fn idx(i: i32, j: i32) -> vec2<i32> {
  return vec2<i32>(wrapi(i, i32(P.nx)), wrapi(j, i32(P.ny)));
}

// Physical positions of the MAC sample points. These MUST agree with the CPU
// Grid class exactly; a half-cell disagreement produces a simulation that still
// looks like fluid but drifts against the reference.
fn posU(i: i32, j: i32) -> vec2<f32> {
  return vec2<f32>(f32(i) * P.dx, (f32(j) + 0.5) * P.dy);
}
fn posV(i: i32, j: i32) -> vec2<f32> {
  return vec2<f32>((f32(i) + 0.5) * P.dx, f32(j) * P.dy);
}
fn posC(i: i32, j: i32) -> vec2<f32> {
  return vec2<f32>((f32(i) + 0.5) * P.dx, (f32(j) + 0.5) * P.dy);
}
`;

/**
 * Bilinear sampling helpers.
 *
 * Hand-rolled rather than using a repeat-address sampler. A sampler wraps in
 * normalised UV space, which is correct for a cell-centred field but half a
 * texel wrong for a face-centred one -- and u, v and p have three different
 * offsets. Doing the wrap on integer texel coordinates keeps one rule for all
 * three and matches the CPU `bilerp` exactly.
 */
export const SAMPLING = /* wgsl */`
fn loadTex(t: texture_2d<f32>, i: i32, j: i32) -> vec4<f32> {
  let c = idx(i, j);
  return textureLoad(t, c, 0);
}

// Bilinear interpolation at grid coordinates (gx, gy), where integer values
// land on texel centres. Identical arithmetic to bilerp() in core/grid.js.
fn bilerpTex(t: texture_2d<f32>, gx: f32, gy: f32) -> vec4<f32> {
  let i0 = i32(floor(gx));
  let j0 = i32(floor(gy));
  let fx = gx - floor(gx);
  let fy = gy - floor(gy);
  let a = loadTex(t, i0,     j0);
  let b = loadTex(t, i0 + 1, j0);
  let c = loadTex(t, i0,     j0 + 1);
  let d = loadTex(t, i0 + 1, j0 + 1);
  let top = mix(a, b, fx);
  let bot = mix(c, d, fx);
  return mix(top, bot, fy);
}

// Sample the u component at an arbitrary physical point.
fn sampleU(vel: texture_2d<f32>, p: vec2<f32>) -> f32 {
  return bilerpTex(vel, p.x / P.dx, p.y / P.dy - 0.5).x;
}
fn sampleV(vel: texture_2d<f32>, p: vec2<f32>) -> f32 {
  return bilerpTex(vel, p.x / P.dx - 0.5, p.y / P.dy).y;
}
fn sampleC(t: texture_2d<f32>, p: vec2<f32>) -> vec4<f32> {
  return bilerpTex(t, p.x / P.dx - 0.5, p.y / P.dy - 0.5);
}

// RK2 (midpoint) backward trace, matching traceBack() on the CPU. Forward Euler
// would systematically trace to the outside of a rotating arc and spin vortices
// down faster than the interpolation alone accounts for.
fn traceBack(vel: texture_2d<f32>, p: vec2<f32>, dt: f32) -> vec2<f32> {
  let v1 = vec2<f32>(sampleU(vel, p), sampleV(vel, p));
  let pm = p - 0.5 * dt * v1;
  let v2 = vec2<f32>(sampleU(vel, pm), sampleV(vel, pm));
  return p - dt * v2;
}
`;

/** Semi-Lagrangian advection of the velocity field. */
export const ADVECT_VELOCITY = /* wgsl */`
@group(0) @binding(1) var src : texture_2d<f32>;
@group(0) @binding(2) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  // Each component is traced from ITS OWN staggered position. Tracing both
  // from the cell centre and interpolating afterwards would add a full extra
  // interpolation of smoothing per step -- roughly doubling the scheme's
  // artificial viscosity, for a shortcut that saves almost nothing.
  let pu = traceBack(src, posU(i, j), P.dt);
  let pv = traceBack(src, posV(i, j), P.dt);
  let nu_ = sampleU(src, pu);
  let nv_ = sampleV(src, pv);
  textureStore(dst, vec2<i32>(i, j), vec4<f32>(nu_, nv_, 0.0, 0.0));
}
`;

/**
 * Advect an arbitrary MAC velocity-shaped field through a separate flow.
 *
 * Needed by MacCormack, whose backward pass must transport the FORWARD RESULT
 * through the ORIGINAL velocity. The plain advection kernel above can only
 * advect a field through itself.
 */
export const ADVECT_FIELD_BY = /* wgsl */`
@group(0) @binding(1) var vel : texture_2d<f32>;
@group(0) @binding(2) var src : texture_2d<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x); let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }
  let pu = traceBack(vel, posU(i, j), P.dt);
  let pv = traceBack(vel, posV(i, j), P.dt);
  // Sample the SOURCE field at its own staggered offsets.
  let a = bilerpTex(src, pu.x / P.dx, pu.y / P.dy - 0.5).x;
  let b = bilerpTex(src, pv.x / P.dx - 0.5, pv.y / P.dy).y;
  textureStore(dst, vec2<i32>(i, j), vec4<f32>(a, b, 0.0, 0.0));
}
`;

/**
 * MacCormack correction with a monotonicity limiter.
 *
 * u_new = fwd + (orig − back)/2, clamped to the range of the four cells the
 * backward trace interpolated from.
 *
 * THE LIMITER IS NOT OPTIONAL. The correction is a second-order extrapolation
 * and will overshoot at sharp gradients, creating velocity extrema that did not
 * exist. On a scalar that is a cosmetic artifact; on the velocity field those
 * overshoots feed straight back through advection and grow. Clamping to the
 * source cells' range keeps the scheme second order where the flow is smooth
 * and falls back to the (monotone) semi-Lagrangian value exactly where it would
 * have overshot.
 */
export const MACCORMACK_COMBINE = /* wgsl */`
@group(0) @binding(1) var orig : texture_2d<f32>;
@group(0) @binding(2) var fwd  : texture_2d<f32>;
@group(0) @binding(3) var back : texture_2d<f32>;
@group(0) @binding(4) var dst  : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x); let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  let o = loadTex(orig, i, j);
  let f = loadTex(fwd,  i, j);
  let b = loadTex(back, i, j);
  var c = f + 0.5 * (o - b);

  // Bounds from the cells the backward trace actually sampled.
  let pu = traceBack(orig, posU(i, j), P.dt);
  let pv = traceBack(orig, posV(i, j), P.dt);

  let gu = vec2<f32>(pu.x / P.dx, pu.y / P.dy - 0.5);
  let iu = vec2<i32>(i32(floor(gu.x)), i32(floor(gu.y)));
  var lo = 1e30; var hi = -1e30;
  for (var dj = 0; dj < 2; dj = dj + 1) {
    for (var di = 0; di < 2; di = di + 1) {
      let s = loadTex(orig, iu.x + di, iu.y + dj).x;
      lo = min(lo, s); hi = max(hi, s);
    }
  }
  c.x = clamp(c.x, lo, hi);

  let gv = vec2<f32>(pv.x / P.dx - 0.5, pv.y / P.dy);
  let iv = vec2<i32>(i32(floor(gv.x)), i32(floor(gv.y)));
  lo = 1e30; hi = -1e30;
  for (var dj = 0; dj < 2; dj = dj + 1) {
    for (var di = 0; di < 2; di = di + 1) {
      let s = loadTex(orig, iv.x + di, iv.y + dj).y;
      lo = min(lo, s); hi = max(hi, s);
    }
  }
  c.y = clamp(c.y, lo, hi);

  textureStore(dst, vec2<i32>(i, j), vec4<f32>(c.x, c.y, 0.0, 0.0));
}
`;

/** Semi-Lagrangian advection of a cell-centred scalar (dye). */
export const ADVECT_SCALAR = /* wgsl */`
@group(0) @binding(1) var vel : texture_2d<f32>;
@group(0) @binding(2) var src : texture_2d<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }
  let p = traceBack(vel, posC(i, j), P.dt);
  var q = sampleC(src, p);
  // Gentle fade so injected dye eventually clears; param_a = 0 disables it.
  q = q * (1.0 - P.param_a);
  textureStore(dst, vec2<i32>(i, j), q);
}
`;

/** Divergence of the MAC velocity field, written to .x of the target. */
export const DIVERGENCE = /* wgsl */`
@group(0) @binding(1) var vel : texture_2d<f32>;
@group(0) @binding(2) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  // u on this cell's left face and on the next cell's left face (= this cell's
  // right face); likewise for v. Exactly the CPU divergence() stencil.
  let uL = loadTex(vel, i,     j).x;
  let uR = loadTex(vel, i + 1, j).x;
  let vB = loadTex(vel, i, j    ).y;
  let vT = loadTex(vel, i, j + 1).y;
  let d = (uR - uL) / P.dx + (vT - vB) / P.dy;
  textureStore(dst, vec2<i32>(i, j), vec4<f32>(d, 0.0, 0.0, 0.0));
}
`;

/**
 * One damped-Jacobi sweep of the pressure Poisson equation.
 *
 * Solves the same system the CPU CG does: laplacian(p) = div, discretised with
 * the 5-point stencil, so p_new = (sum of neighbours - h^2 * div) / 4.
 *
 * Jacobi is the baseline in the solver comparison, not the recommendation. It
 * is embarrassingly parallel -- every cell reads only the previous iterate, so
 * there is no ordering constraint at all -- but its error reduction per sweep
 * is poor for the low-frequency pressure modes, which is precisely what the
 * residual-vs-iteration curve on the report shows.
 */
export const JACOBI = /* wgsl */`
@group(0) @binding(1) var pressure : texture_2d<f32>;
@group(0) @binding(2) var divergence_ : texture_2d<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  let l = loadTex(pressure, i - 1, j).x;
  let r = loadTex(pressure, i + 1, j).x;
  let b = loadTex(pressure, i, j - 1).x;
  let t = loadTex(pressure, i, j + 1).x;
  let d = loadTex(divergence_, i, j).x;
  let h2 = P.dx * P.dx;

  let jac = (l + r + b + t - h2 * d) * 0.25;
  let cur = loadTex(pressure, i, j).x;
  let next = cur + P.param_a * (jac - cur);   // param_a = omega (1.0 = plain Jacobi)
  textureStore(dst, vec2<i32>(i, j), vec4<f32>(next, 0.0, 0.0, 0.0));
}
`;

/**
 * Red-black Gauss-Seidel: one colour per dispatch.
 *
 * Gauss-Seidel uses updated neighbour values within the same sweep, which
 * roughly doubles the convergence rate over Jacobi -- but that creates a
 * sequential dependency that a GPU cannot honour. Red-black colouring removes
 * it: on the 5-point stencil every cell's four neighbours have the opposite
 * parity, so all cells of one colour can be updated simultaneously using only
 * cells of the other. Two dispatches per sweep, full parallelism, and the
 * convergence of Gauss-Seidel.
 *
 * param_b selects the colour (0 = red, 1 = black).
 */
export const RED_BLACK_GS = /* wgsl */`
@group(0) @binding(1) var pressure : texture_2d<f32>;
@group(0) @binding(2) var divergence_ : texture_2d<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  let cur = loadTex(pressure, i, j).x;
  let colour = u32((i + j) % 2);
  if (colour != u32(P.param_b)) {
    // Wrong colour this pass: copy through unchanged. The copy is required --
    // this is a ping-pong write, so a cell that is not written this pass would
    // otherwise keep whatever stale value the destination texture held.
    textureStore(dst, vec2<i32>(i, j), vec4<f32>(cur, 0.0, 0.0, 0.0));
    return;
  }

  let l = loadTex(pressure, i - 1, j).x;
  let r = loadTex(pressure, i + 1, j).x;
  let b = loadTex(pressure, i, j - 1).x;
  let t = loadTex(pressure, i, j + 1).x;
  let d = loadTex(divergence_, i, j).x;
  let h2 = P.dx * P.dx;

  let gs = (l + r + b + t - h2 * d) * 0.25;
  let next = cur + P.param_a * (gs - cur);    // param_a = over-relaxation factor
  textureStore(dst, vec2<i32>(i, j), vec4<f32>(next, 0.0, 0.0, 0.0));
}
`;

/** Subtract the pressure gradient, making the field divergence-free. */
export const SUBTRACT_GRADIENT = /* wgsl */`
@group(0) @binding(1) var vel : texture_2d<f32>;
@group(0) @binding(2) var pressure : texture_2d<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  let c = loadTex(vel, i, j);
  let pc = loadTex(pressure, i,     j).x;
  let pl = loadTex(pressure, i - 1, j).x;
  let pb = loadTex(pressure, i, j - 1).x;

  // The u-face at i sits between cells i-1 and i, so its gradient uses those
  // two centres. This is the exact adjoint of the divergence stencil above;
  // that adjointness is what makes the projection orthogonal.
  let nu_ = c.x - (pc - pl) / P.dx;
  let nv_ = c.y - (pc - pb) / P.dy;
  textureStore(dst, vec2<i32>(i, j), vec4<f32>(nu_, nv_, 0.0, 0.0));
}
`;

/** Explicit viscous diffusion. */
export const DIFFUSE = /* wgsl */`
@group(0) @binding(1) var src : texture_2d<f32>;
@group(0) @binding(2) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  let c = loadTex(src, i, j);
  let l = loadTex(src, i - 1, j);
  let r = loadTex(src, i + 1, j);
  let b = loadTex(src, i, j - 1);
  let t = loadTex(src, i, j + 1);

  // Stable only for nu*dt/h^2 <= 0.25 -- measured exactly at that boundary on
  // the CPU. The caller is responsible for respecting it; at the viscosities
  // the demo uses it is never close.
  let a = P.nu * P.dt / (P.dx * P.dx);
  let bb = P.nu * P.dt / (P.dy * P.dy);
  let out = c + a * (l + r - 2.0 * c) + bb * (b + t - 2.0 * c);
  textureStore(dst, vec2<i32>(i, j), vec4<f32>(out.x, out.y, 0.0, 0.0));
}
`;

/**
 * Vorticity confinement.
 *
 * NON-PHYSICAL, and only ever enabled in the interactive demo. It computes
 * grad|omega|, normalises it, and applies eps*h*(N x omega) to push energy back
 * into vortices that the advection scheme has smeared. It restores the LOOK of
 * small-scale detail without restoring the information, which is why the CPU
 * solver refuses to run it in validation mode and why no measured number in
 * this project comes from a run with it on.
 */
export const VORTICITY_CONFINEMENT = /* wgsl */`
@group(0) @binding(1) var vel : texture_2d<f32>;
@group(0) @binding(2) var dst : texture_storage_2d<rgba32float, write>;

fn curlAt(vel_: texture_2d<f32>, i: i32, j: i32) -> f32 {
  let dvdx = (loadTex(vel_, i, j).y - loadTex(vel_, i - 1, j).y) / P.dx;
  let dudy = (loadTex(vel_, i, j).x - loadTex(vel_, i, j - 1).x) / P.dy;
  return dvdx - dudy;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  let w  = curlAt(vel, i, j);
  let wl = abs(curlAt(vel, i - 1, j));
  let wr = abs(curlAt(vel, i + 1, j));
  let wb = abs(curlAt(vel, i, j - 1));
  let wt = abs(curlAt(vel, i, j + 1));

  var n = vec2<f32>((wr - wl) / (2.0 * P.dx), (wt - wb) / (2.0 * P.dy));
  let m = length(n) + 1e-20;   // guard: grad|omega| is exactly zero in uniform flow
  n = n / m;

  let f = P.param_a * P.dx * P.dt * vec2<f32>(n.y * w, -n.x * w);
  let c = loadTex(vel, i, j);
  textureStore(dst, vec2<i32>(i, j), vec4<f32>(c.x + f.x, c.y + f.y, 0.0, 0.0));
}
`;

/**
 * Mouse interaction: a Gaussian splat of force and dye.
 * param_a = radius, param_b = strength; the direction and colour arrive in the
 * second uniform block.
 */
export const SPLAT = /* wgsl */`
struct Splat {
  pos    : vec2<f32>,
  vel    : vec2<f32>,
  colour : vec4<f32>,
  radius : f32,
  isDye  : f32,
  _pad   : vec2<f32>,
};
@group(0) @binding(1) var<uniform> S : Splat;
@group(0) @binding(2) var src : texture_2d<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  let c = loadTex(src, i, j);
  let p = posC(i, j);

  // Shortest separation on a periodic domain, so a splat near an edge behaves
  // the same as one in the middle rather than being clipped.
  let L = vec2<f32>(f32(P.nx) * P.dx, f32(P.ny) * P.dy);
  var d = p - S.pos;
  d = d - L * round(d / L);

  let g = exp(-dot(d, d) / max(S.radius * S.radius, 1e-12));

  if (S.isDye > 0.5) {
    textureStore(dst, vec2<i32>(i, j), c + S.colour * g);
  } else {
    textureStore(dst, vec2<i32>(i, j),
      vec4<f32>(c.x + S.vel.x * g, c.y + S.vel.y * g, 0.0, 0.0));
  }
}
`;

/**
 * Band-limited forcing plus large-scale drag, in one pass.
 *
 * This started life on the CPU, between GPU steps, and it dominated everything:
 * 32 modes over 512^2 cells is 8.4 million cosines per application, plus a full
 * texture readback and re-upload each time. The GPU solve was idle waiting for
 * it. Moving it into a kernel is also simply the right place for it -- the
 * brief asks for the solve to run on GPU threads, and a body force is part of
 * the solve.
 *
 * The forcing is built as the curl of a streamfunction, so it is
 * divergence-free by construction and the projection has nothing to undo. A
 * forcing with a divergent part would be partially cancelled by the very next
 * projection, making the effective injection rate depend on the solver rather
 * than on the amplitude asked for.
 *
 * Drag is applied as an exact exponential rather than (1 - alpha*dt), so the
 * damping rate does not depend on the timestep.
 */
export const FORCING = /* wgsl */`
struct Mode { kx: f32, ky: f32, phase: f32, _pad: f32 };
@group(0) @binding(1) var<storage, read> modes : array<Mode>;
@group(0) @binding(2) var src : texture_2d<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x); let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  let pu = posU(i, j);
  let pv = posV(i, j);
  var fu = 0.0;
  var fv = 0.0;

  let count = arrayLength(&modes);
  for (var m = 0u; m < count; m = m + 1u) {
    let md = modes[m];
    let kk = sqrt(md.kx * md.kx + md.ky * md.ky);
    if (kk < 1e-6) { continue; }
    // param_b carries a slow phase drift so the forcing decorrelates in time;
    // a static forcing would drive one fixed flow pattern rather than
    // sustaining turbulence.
    fu = fu + (md.ky / kk) * cos(md.kx * pu.x + md.ky * pu.y + md.phase + P.param_b);
    fv = fv - (md.kx / kk) * cos(md.kx * pv.x + md.ky * pv.y + md.phase + P.param_b);
  }

  let scale = P.param_a * P.dt / sqrt(f32(count));
  let damp = exp(-P.nu * P.dt);      // nu reused as the drag coefficient here
  let c = loadTex(src, i, j);
  textureStore(dst, vec2<i32>(i, j),
    vec4<f32>(c.x * damp + scale * fu, c.y * damp + scale * fv, 0.0, 0.0));
}
`;

/**
 * Render kernel: velocity/vorticity -> colour.
 *
 * The colour is a function of the physics, not decoration layered on top. Two
 * mappings, both derived from fields the solver already computes:
 *   mode 0 -- vorticity, diverging: sign of rotation gives hue, magnitude gives
 *             intensity, so counter-rotating vortex pairs are visibly distinct
 *   mode 1 -- speed, sequential
 * Dye, when present, is composited over.
 */
export const RENDER = /* wgsl */`
@group(0) @binding(1) var vel : texture_2d<f32>;
@group(0) @binding(2) var dye : texture_2d<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba8unorm, write>;

fn curlAt(i: i32, j: i32) -> f32 {
  let dvdx = (loadTex(vel, i, j).y - loadTex(vel, i - 1, j).y) / P.dx;
  let dudy = (loadTex(vel, i, j).x - loadTex(vel, i, j - 1).x) / P.dy;
  return dvdx - dudy;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }

  // Vorticity averaged from the four surrounding corners to the cell centre.
  let w = 0.25 * (curlAt(i, j) + curlAt(i + 1, j) + curlAt(i, j + 1) + curlAt(i + 1, j + 1));
  let c = loadTex(vel, i, j);
  let speed = length(vec2<f32>(c.x, c.y));

  var col : vec3<f32>;
  if (P.param_b < 0.5) {
    // Diverging map. Signed vorticity scaled by a soft nonlinearity so that
    // both the strong cores and the weak filaments between them are visible --
    // a linear map shows only the cores and hides the structure that makes
    // turbulence look like turbulence.
    let s = tanh(w * P.param_a);
    let warm = vec3<f32>(1.00, 0.42, 0.21);
    let cool = vec3<f32>(0.16, 0.53, 0.96);
    let mid  = vec3<f32>(0.02, 0.02, 0.05);
    if (s >= 0.0) { col = mix(mid, warm, s); } else { col = mix(mid, cool, -s); }
  } else {
    let t = tanh(speed * P.param_a);
    col = mix(vec3<f32>(0.01, 0.01, 0.04), vec3<f32>(0.55, 0.92, 1.0), t);
  }

  let d = loadTex(dye, i, j);
  col = col + d.rgb;
  col = col / (1.0 + col);          // Reinhard, so bright dye rolls off rather than clipping
  col = pow(col, vec3<f32>(1.0 / 2.2));   // to sRGB

  textureStore(dst, vec2<i32>(i, j), vec4<f32>(col, 1.0));
}
`;
