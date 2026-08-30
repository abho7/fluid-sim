# fluid-sim

[![ci](https://github.com/abho7/fluid-sim/actions/workflows/pages.yml/badge.svg)](https://github.com/abho7/fluid-sim/actions/workflows/pages.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An incompressible Navier-Stokes solver running on WebGPU compute shaders, with
the physics validated against closed-form solutions rather than against how it
looks.

**[Live demo](https://abho7.github.io/fluid-sim/)** ·
**[Validation report](https://abho7.github.io/fluid-sim/report.html)**

No runtime dependencies. The solver, the FFT, the multigrid cycle, the plots on
the report and the static server are all written from scratch; Node and a
WebGPU-capable browser are the only requirements.

---

## What is actually claimed

The interesting question is not whether it looks like fluid — a shader trick
does that. It is whether the numbers are right, and by how much they are wrong.
So the report answers, with measurements:

| question | answer |
|---|---|
| How much viscosity did the scheme invent? | **ν_num = 5.79e-4** with MacCormack, 2.9% of the physical ν. With plain semi-Lagrangian it is 2.10e-2 — **105% of ν**, i.e. the simulated fluid is twice as viscous as the one requested. |
| Where does it go unstable? | Explicit diffusion at **ν·dt/h² = 0.25 stable, 0.26 unstable**. Theory says exactly 0.25, and measured amplification matches \|1 − 8d\| to better than 1e-6 at every point. |
| What order is it? | **First order in time** — Lie splitting caps it there regardless of the operators, which are individually second order. |
| Does it reproduce turbulence? | **Partly, and the difference is diagnosable.** The 2D inverse energy cascade measures **-1.525** against Kraichnan's −5/3 (r² = 0.986) — within 9%. The enstrophy cascade is **-4.529** against −3: a very clean power law (r² = 0.999) at the wrong exponent. |
| Is it really incompressible? | max \|∇·u\| ≈ **1e-13** on the CPU reference; ~1e-6 on the GPU with multigrid. |
| Does the two-way coupling conserve momentum? | **1.1e-14** relative drift per coupling step — Newton's third law holds to machine precision. |

## Method

Stam's stable-fluids scheme, implemented properly, with two deliberate
departures from the usual presentation:

- **MAC staggered grid**, not collocated. Storing u, v and p at cell centres
  admits a checkerboard pressure mode that the 5-point Laplacian cannot see, so
  it accumulates invisibly to the solver and visibly on screen. Staggering makes
  divergence and gradient exact negative adjoints, which is asserted directly in
  the tests.
- **Two advection schemes**, semi-Lagrangian (1st order) and MacCormack/BFECC
  (2nd order, limited), because the difference between them is one of the
  project's actual results rather than an implementation detail.

Pressure projection is solved four ways so the comparison is measurable:
conjugate gradient and an exact FFT solve on the CPU, and Jacobi, red-black
Gauss-Seidel and a geometric multigrid V-cycle on the GPU.

## Vorticity confinement is off in every measurement

Confinement restores small-scale detail that numerical dissipation destroyed,
and it is in the demo because it looks good. It is also a **fabricated energy
source**: the sweep on the report shows the simulation *gaining* energy from
ε = 1 upward while the exact solution is losing it.

A decay rate or a spectrum measured with confinement enabled would be measuring
the confinement parameter, not the fluid. So the solver **throws** if it is
requested in validation mode, rather than relying on anyone to remember.

## Layout

```
src/core/      grid + MAC indexing, analytic solutions, error norms, FFT, spectra
src/cpu/       f64 reference solver — the correctness oracle
src/gpu/       WGSL compute kernels, multigrid, the WebGPU backend
validate/      headless studies -> results/*.json
tools/         static server, GPU harnesses, plotting, report builder
test/          solver + primitives, and mutation tests on the validators
```

The CPU solver is portable ES modules with no DOM or GPU dependency, so the
analytic validation runs headlessly under Node and is reproducible without a
GPU, while the same code runs in the browser for a same-machine, same-moment
performance comparison.

## Running it

```bash
node --test test/*.test.js     # 76 tests
node validate/run.js           # regenerate every CPU-side number
node tools/serve.js            # WebGPU needs a secure context; localhost counts
node tools/build_report.js     # rebuild report.html from results/*.json
```

With the server up:

| page | what it does |
|---|---|
| `/` | the interactive demo |
| `/tools/gputest.html` | verifies every GPU kernel against the f64 CPU reference |
| `/tools/bench.html` | CPU vs GPU, same algorithm and at equal solution quality |
| `/tools/spectrum.html` | forced 2D turbulence, energy spectrum |

The demo's **obstacle** dropdown attaches an immersed rigid disk — free or pinned —
that blocks the flow and is pushed by it.

`node tools/collect.js` receives results from those pages and writes them to
`results/`, so every published number traces to a run.

## Bugs found

Four, all fixed, all written up on the report. Three were in the *measuring*
code rather than the solver, which is the part worth noting:

1. **Every GPU pass read the same uniform values.** `queue.writeBuffer` is
   ordered against *submitted* command buffers, not against commands being
   encoded, so a step that wrote params and encoded a pass repeatedly before one
   submit had every pass read the final value. Jacobi's relaxation factor became
   0 (the solve did nothing) and red-black's colour flag was always 1 (red cells
   never updated, residual frozen from the first sweep). Fixed with a ring of
   per-pass uniform slots.
2. **The stability sweep reported the scheme stable at 2.4× its real limit** — it
   seeded a smooth field containing none of the mode that goes unstable, and
   watched for an energy threshold a decaying field would never cross. The
   operator had been correct all along.
3. **A benchmark that flattered the GPU twice** — CG on the CPU against Jacobi on
   the GPU, and 20 GPU sweeps against an exact CPU solve. The honest numbers are
   lower, and showed the GPU *losing* at equal quality, which is what motivated
   implementing multigrid.
4. **Non-monotonic GPU timings** from insufficient warm-up: a 4× smaller grid
   measured 3× slower.

## Known limitations

- **The enstrophy range is still too steep** (-4.53 against −3), most likely
  residual numerical dissipation at the smallest scales, where f32 and a 3-cycle
  projection bite hardest. This is a measured attribution, not a guess: running
  the identical study with only the advection scheme changed moved the inverse
  cascade from +0.99 to -1.52 and the enstrophy slope from −6.08 to -4.53. So
  dissipation is demonstrably most of it — but not all, and the remainder is not
  yet explained.
- **First order in time**, from Lie splitting. Strang splitting would give second
  order for ~1.5× the cost.
- **f32 on the GPU.** The multigrid residual plateaus near 1e-7 because that is
  the precision floor. GPU/CPU agreement cannot be claimed below ~1e-5.
- **Periodic boundaries only.** The FFT solver and the spectra both require it.
- **Measured on an Intel iGPU** sharing bandwidth with the CPU — the pessimistic
  end of every performance ratio quoted.

---

Abhineeth Duddela · © 2026
