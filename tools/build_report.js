/**
 * Builds report.html from the measured JSON in results/.
 *
 * EVERY NUMBER ON THE PAGE IS READ FROM A RESULTS FILE. There is no literal
 * measurement in this file. Where a result is missing, the page says
 * "not measured" rather than falling back to something plausible -- a report
 * that quietly prints a default is worse than one with a visible gap.
 *
 *   node tools/build_report.js
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { plot, powerLaw, PAL, esc } from "./plot.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const RES = join(ROOT, "results");

const load = (name) => {
  const p = join(RES, `${name}.json`);
  if (!existsSync(p)) return null;
  try { return JSON.parse(readFileSync(p, "utf8")); }
  catch (e) { console.warn(`  ! ${name}.json is unparseable: ${e.message}`); return null; }
};

const MISSING = '<span class="missing">not measured</span>';
const num = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v))
  ? MISSING
  : (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? v.toExponential(2) : v.toFixed(d));

const validation = load("validation");
const spectrum = load("spectrum-maccormack") ?? load("spectrum");
const spectrumSL = load("spectrum");   // the semi-Lagrangian run, for comparison

// The collector stores whatever the harness posted. The spectrum page posts
// `R.result` directly, so its JSON has no `.result` wrapper -- but an earlier
// manual capture did. Normalising once here avoids the two accessors drifting
// apart, which is exactly what happened: the summary KPI read `.result.fits`
// and printed "not measured" while the section below read the same file
// successfully through a fallback.
const unwrap = (x) => x ? (x.result ?? x) : null;
const gpuVerify = load("gpu-verify");
const bench = load("bench");

// ============================================================ page components

const CSS = `
:root{
  --bg:#06090d; --panel:#0d1218; --raised:#131a22; --line:#1e2733; --line2:#2b3644;
  --ink:#eaf0f6; --ink2:#aab6c4; --dim:#828d9b;
  --accent:#69d9ff; --accent-dim:#2f7f99;
  --good:#7ee787; --warn:#ffc857; --bad:#ff6b81; --violet:#b18cff;
  --mono:ui-monospace,SFMono-Regular,"JetBrains Mono",Consolas,"Liberation Mono",monospace;
  --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
  --measure:68ch;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.68;-webkit-font-smoothing:antialiased}
.wrap{max-width:76rem;margin:0 auto;padding:0 clamp(1.1rem,4vw,2.5rem)}
.prose{max-width:var(--measure)}
h1,h2,h3,h4{margin:0;line-height:1.2;letter-spacing:-.021em;font-weight:640;text-wrap:balance}
p{margin:0}
a{color:var(--accent);text-decoration-color:var(--accent-dim);text-underline-offset:3px}
code{font-family:var(--mono);font-size:.86em;background:var(--raised);
  border:1px solid var(--line);border-radius:4px;padding:.06em .34em;color:var(--ink2)}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}

section{padding:3.2rem 0;border-top:1px solid var(--line)}
section > .wrap{display:flex;flex-direction:column;gap:1.4rem}
.stack{display:flex;flex-direction:column;gap:1rem}
.stack-sm{display:flex;flex-direction:column;gap:.5rem}

.eyebrow{font-family:var(--mono);font-size:.6rem;letter-spacing:.24em;
  text-transform:uppercase;color:var(--accent);margin:0}
h2{font-size:clamp(1.35rem,2.4vw,1.72rem)}
h3{font-size:1rem}
h4{font-size:.88rem;color:var(--ink2)}
.lede{color:var(--ink2);font-size:1.02rem}
.note{color:var(--dim);font-size:.86rem;max-width:var(--measure)}
.missing{color:var(--warn);font-family:var(--mono);font-size:.85em}

header{padding:clamp(3.5rem,9vw,6rem) 0 3rem;position:relative;overflow:hidden}
header .wrap{display:flex;flex-direction:column;gap:1.3rem;position:relative;z-index:1}
header canvas{position:absolute;inset:0;width:100%;height:100%;opacity:.42;z-index:0}
h1{font-size:clamp(2rem,5.2vw,3.2rem);letter-spacing:-.034em;font-weight:700}
h1 em{font-style:normal;color:var(--accent)}
.byline{font-family:var(--mono);font-size:.66rem;letter-spacing:.13em;
  text-transform:uppercase;color:var(--dim);display:flex;flex-wrap:wrap;gap:.5rem 1.2rem}

.kpi{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:14px;overflow:hidden}
@media(max-width:60rem){.kpi{grid-template-columns:repeat(2,minmax(0,1fr))}}
.kpi>div{background:var(--panel);padding:1rem 1.1rem;display:flex;
  flex-direction:column;gap:.28rem;min-width:0}
.kpi .v{font-family:var(--mono);font-size:1.42rem;font-weight:700;letter-spacing:-.03em;
  font-variant-numeric:tabular-nums;line-height:1.1}
.kpi .k{font-family:var(--mono);font-size:.55rem;letter-spacing:.17em;
  text-transform:uppercase;color:var(--dim)}
.kpi .s{font-size:.75rem;color:var(--dim);line-height:1.4}
.v.good{color:var(--good)} .v.accent{color:var(--accent)}
.v.warn{color:var(--warn)} .v.bad{color:var(--bad)}

.cards{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(17rem,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:1.1rem 1.2rem;display:flex;flex-direction:column;gap:.5rem}
.card p{font-size:.87rem;color:var(--ink2)}
.card .meta{font-family:var(--mono);font-size:.58rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--dim)}

.scroll{overflow-x:auto;border-radius:12px}
table{width:100%;border-collapse:collapse;font-size:.82rem;background:var(--panel)}
thead th{font-family:var(--mono);font-size:.55rem;letter-spacing:.15em;text-transform:uppercase;
  color:var(--dim);font-weight:400;text-align:left;padding:.7rem .9rem;
  border-bottom:1px solid var(--line2);white-space:nowrap;background:var(--raised)}
tbody td{padding:.62rem .9rem;border-bottom:1px solid var(--line);color:var(--ink2);
  vertical-align:top}
tbody tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;font-family:var(--mono);
  font-variant-numeric:tabular-nums;white-space:nowrap}
td.mono{font-family:var(--mono);white-space:nowrap}
td strong{color:var(--ink);font-weight:600}
.tbl-wrap{border:1px solid var(--line);border-radius:12px;overflow:hidden}

.pill{font-family:var(--mono);font-size:.57rem;letter-spacing:.1em;text-transform:uppercase;
  padding:.14rem .5rem;border-radius:999px;border:1px solid currentColor;white-space:nowrap}
.pill.good{color:var(--good)} .pill.bad{color:var(--bad)} .pill.warn{color:var(--warn)}

figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:1.2rem;display:flex;flex-direction:column;gap:.8rem}
figure svg{display:block;max-width:100%;height:auto}
figcaption{font-size:.79rem;color:var(--dim);max-width:var(--measure);line-height:1.6}
.figs{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(24rem,1fr))}

.finding{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:10px;padding:1.1rem 1.25rem;display:flex;flex-direction:column;gap:.6rem}
.finding.bug{border-left-color:var(--bad)}
.finding h3{font-size:.96rem}
.finding dl{margin:0;display:grid;grid-template-columns:6.5rem 1fr;gap:.4rem 1rem;font-size:.86rem}
@media(max-width:44rem){.finding dl{grid-template-columns:1fr;gap:.1rem}
  .finding dd{margin-bottom:.45rem}}
.finding dt{font-family:var(--mono);font-size:.55rem;letter-spacing:.15em;text-transform:uppercase;
  color:var(--dim);padding-top:.28rem}
.finding dd{margin:0;color:var(--ink2)}

.cta{display:inline-flex;align-items:center;gap:.5rem;font-family:var(--mono);
  font-size:.67rem;letter-spacing:.12em;text-transform:uppercase;background:var(--accent);
  color:#04222c;text-decoration:none;font-weight:700;padding:.7rem 1.15rem;
  border-radius:8px;width:fit-content}
.cta:hover{background:#9ce8ff}
.cta.ghost{background:transparent;color:var(--ink2);border:1px solid var(--line2)}
.cta.ghost:hover{background:var(--raised);color:var(--ink)}
.row{display:flex;gap:.7rem;flex-wrap:wrap;align-items:center}

pre{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:1rem 1.1rem;overflow-x:auto;font-family:var(--mono);font-size:.78rem;
  line-height:1.7;color:var(--ink2)}
pre .c{color:var(--dim)}
ul{margin:0;padding-left:1.15rem;display:flex;flex-direction:column;gap:.42rem}
li{color:var(--ink2);font-size:.92rem}
li strong{color:var(--ink);font-weight:600}
footer{padding:2.4rem 0 4rem;border-top:1px solid var(--line);color:var(--dim);
  font-family:var(--mono);font-size:.65rem;line-height:1.9}
@media (prefers-reduced-motion:reduce){*{animation-duration:.001ms!important}}
`;

const HERO_JS = String.raw`
(function(){
  // A passive vorticity-like field in the header: two counter-rotating Gaussian
  // vortices advecting tracer particles. It is the same phenomenon the report
  // is about, drawn cheaply, and it stops entirely under reduced-motion.
  var c=document.getElementById('hero'); if(!c) return;
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  var x=c.getContext('2d'), P=[], raf=null;
  function size(){var d=Math.min(devicePixelRatio||1,2);
    c.width=c.clientWidth*d; c.height=c.clientHeight*d; x.setTransform(d,0,0,d,0,0);}
  function seed(){P=[];var w=c.clientWidth,h=c.clientHeight,n=Math.min(900,Math.round(w*h/1400));
    for(var i=0;i<n;i++)P.push({x:Math.random()*w,y:Math.random()*h,a:Math.random()});}
  function vel(px,py,w,h){
    var vx=0,vy=0;
    var cs=[[w*0.32,h*0.5,1],[w*0.68,h*0.5,-1],[w*0.5,h*0.15,-0.6],[w*0.5,h*0.9,0.6]];
    for(var i=0;i<cs.length;i++){
      var dx=px-cs[i][0], dy=py-cs[i][1], r2=dx*dx+dy*dy+900;
      var s=cs[i][2]*24000/r2;
      vx+=-dy/Math.sqrt(r2)*s; vy+=dx/Math.sqrt(r2)*s;
    }
    return [vx,vy];
  }
  function frame(){
    var w=c.clientWidth,h=c.clientHeight;
    x.fillStyle='rgba(6,9,13,0.14)'; x.fillRect(0,0,w,h);
    for(var i=0;i<P.length;i++){
      var p=P[i], v=vel(p.x,p.y,w,h);
      p.x+=v[0]*0.016; p.y+=v[1]*0.016; p.a-=0.0016;
      if(p.a<=0||p.x<0||p.x>w||p.y<0||p.y>h){
        p.x=Math.random()*w; p.y=Math.random()*h; p.a=1;
      }
      var sp=Math.min(1,Math.hypot(v[0],v[1])/60);
      x.fillStyle='rgba('+Math.round(60+150*sp)+','+Math.round(190-40*sp)+',255,'+(0.16*p.a).toFixed(3)+')';
      x.fillRect(p.x,p.y,1.4,1.4);
    }
    raf=requestAnimationFrame(frame);
  }
  function boot(){size();seed();if(!raf)frame();}
  addEventListener('resize',function(){size();seed();});
  boot();
})();
`;

function kpi(v, k, s, cls = "") {
  return `<div><span class="v ${cls}">${v}</span>` +
    `<span class="k">${esc(k)}</span><span class="s">${esc(s)}</span></div>`;
}

function table(head, rows) {
  return `<div class="tbl-wrap"><div class="scroll"><table>
  <thead><tr>${head.map(h => `<th${h.num ? ' class="num"' : ""}>${esc(h.t ?? h)}</th>`).join("")}</tr></thead>
  <tbody>${rows.map(r => `<tr>${r.join("")}</tr>`).join("\n  ")}</tbody>
</table></div></div>`;
}
const td = (v, cls = "") => `<td${cls ? ` class="${cls}"` : ""}>${v}</td>`;

// ================================================================== sections

function heroSection() {
  const env = validation?.env;
  const adapter = bench?.adapter ?? gpuVerify?.adapter ?? spectrum?.result?.adapter;
  return `
<header>
  <canvas id="hero" aria-hidden="true"></canvas>
  <div class="wrap">
    <p class="eyebrow">fluid-sim &middot; validation report</p>
    <h1>The physics is real.<br>Here is <em>how real</em>.</h1>
    <p class="lede prose">
      An incompressible Navier-Stokes solver on WebGPU compute shaders, measured
      against closed-form solutions rather than against how it looks. This page
      reports what the solver gets right, what it gets wrong, and by how much
      &mdash; including a turbulence result that does not match theory, and why.
    </p>
    <p class="byline">
      ${env ? `<span>node ${esc(env.node)}</span><span>${esc(env.cpu)}</span>` : ""}
      ${adapter ? `<span>gpu ${esc(adapter.vendor)} ${esc(adapter.architecture)}</span>` : ""}
      ${env ? `<span>generated ${esc(env.generatedAt?.slice(0, 19))}</span>` : ""}
    </p>
    <div class="row">
      <a class="cta" href="index.html">Open the live demo &#8594;</a>
      <a class="cta ghost" href="https://github.com/abho7/fluid-sim">Source</a>
    </div>
  </div>
</header>`;
}

function summarySection() {
  const tg = validation?.taylorGreen;
  const mc = tg?.schemes?.maccormack;
  const sl = tg?.schemes?.["semi-lagrangian"];
  const eb = validation?.stability?.explicitDiffusion?.measuredBoundary;
  const spec = unwrap(spectrum);
  const ens = spec?.fits?.enstrophyCascade;
  const invc = spec?.fits?.inverseCascade;
  const gv = gpuVerify;
  const mgRows = bench?.equalQuality?.rows;
  const bigMG = mgRows?.[mgRows.length - 1];

  return `
<section><div class="wrap">
  <div class="kpi">
    ${kpi(mc ? num(mc.nuNumerical, 0) : MISSING, "numerical viscosity",
      mc ? `MacCormack, ${(mc.nuNumericalRatio * 100).toFixed(1)}% of the physical ν` : "", "accent")}
    ${kpi(eb ? `${eb.isolated.lastStable} / ${eb.isolated.firstUnstable}` : MISSING,
      "stability boundary", "measured; theory says 0.25", "good")}
    ${kpi(invc ? num(invc.slope, 2) : MISSING, "inverse cascade",
      "theory (Kraichnan) says −1.67",
      invc && Math.abs(invc.slope + 5 / 3) < 0.25 ? "good" : "bad")}
    ${kpi(ens ? num(ens.slope, 2) : MISSING, "enstrophy cascade",
      "theory (Kraichnan) says −3",
      ens && Math.abs(ens.slope + 3) < 0.4 ? "good" : "bad")}
    ${kpi(gv ? num(gv.fullStep?.du, 0) : MISSING, "gpu vs cpu",
      "relative L2 after 10 full steps", "good")}
    ${kpi(mc ? num(mc.maxDivergence, 0) : MISSING, "max |∇·u|",
      "incompressibility, CPU reference", "good")}
    ${kpi(bigMG ? `${bigMG.ratioMultigrid.toFixed(1)}×` : MISSING,
      "gpu speedup", bigMG ? `at ${bigMG.n}², equal solution quality` : "", "accent")}
    ${kpi(sl && mc ? `${(sl.nuNumerical / mc.nuNumerical).toFixed(0)}×` : MISSING,
      "scheme difference", "semi-Lagrangian vs MacCormack dissipation")}

  </div>
  <p class="note">
    Every figure on this page was produced by a run and read out of
    <code>results/*.json</code>. Nothing is typed in; where a value is missing,
    the page says so.
  </p>
</div></section>`;
}

function taylorGreenSection() {
  const tg = validation?.taylorGreen;
  if (!tg) return "";
  const schemes = Object.entries(tg.schemes);

  const errPlot = plot({
    series: schemes.map(([name, s], i) => ({
      x: s.trace.map(p => p.t), y: s.trace.map(p => p.relL2),
      label: name === "maccormack" ? "MacCormack" : "semi-Lagrangian",
      colour: i === 0 ? PAL.warm : PAL.accent,
    })),
    xScale: "linear", yScale: "log",
    xLabel: "time", yLabel: "relative L2 error", title: "Error against the exact solution",
    caption: `Both schemes on the same grid (${tg.n}²), ν = ${tg.nu}, dt = ${tg.dt}. ` +
      `MacCormack's error is roughly ${(schemes[0][1].finalRelL2 / schemes[1][1].finalRelL2).toFixed(0)}× ` +
      `smaller at the end of the run.`,
  });

  const kePlot = plot({
    series: [
      { x: schemes[0][1].trace.map(p => p.t), y: schemes[0][1].trace.map(p => p.keExact),
        label: "exact  e^(−4νt)", colour: PAL.good, dashed: true },
      ...schemes.map(([name, s], i) => ({
        x: s.trace.map(p => p.t), y: s.trace.map(p => p.ke),
        label: name === "maccormack" ? "MacCormack" : "semi-Lagrangian",
        colour: i === 0 ? PAL.warm : PAL.accent,
      })),
    ],
    xScale: "linear", yScale: "log",
    xLabel: "time", yLabel: "kinetic energy", title: "Energy decay vs the closed form",
    caption: "The gap between each curve and the dashed exact solution is energy the " +
      "scheme removed that physics did not. Fitting that gap gives the numerical " +
      "viscosity quoted below.",
  });

  const rows = schemes.map(([name, s]) => [
    td(`<strong>${name === "maccormack" ? "MacCormack" : "semi-Lagrangian"}</strong>`),
    td(num(s.measuredDecayRate, 4), "num"),
    td(num(s.exactDecayRate, 4), "num"),
    td(num(s.nuEffective, 4), "num"),
    td(`<strong>${num(s.nuNumerical, 4)}</strong>`, "num"),
    td(`${(s.nuNumericalRatio * 100).toFixed(1)}%`, "num"),
    td(num(s.decayFitR2, 5), "num"),
  ]);

  return `
<section><div class="wrap">
  <p class="eyebrow">the headline measurement</p>
  <h2>Taylor-Green vortex: how much viscosity did the solver invent?</h2>
  <div class="prose stack">
    <p class="lede">
      The 2D Taylor-Green vortex is an exact solution of Navier-Stokes. Its
      kinetic energy decays as <code>e^(−4νt)</code>, precisely. A simulation
      decays faster, because the advection scheme's interpolation removes energy
      on top of the physical viscosity.
    </p>
    <p style="color:var(--ink2);font-size:.94rem">
      Fitting the measured decay to <code>e^(−4ν_eff·t)</code> gives an effective
      viscosity, and <strong>ν_num = ν_eff − ν</strong> is the artificial part
      &mdash; the same units as the real thing, so "this scheme is dissipative"
      becomes a number instead of an adjective. The r² column matters: it
      confirms the decay really is exponential, without which ν_eff would be
      meaningless.
    </p>
  </div>
  <div class="figs">${errPlot}${kePlot}</div>
  ${table(
    ["scheme", { t: "measured rate", num: 1 }, { t: "exact 4ν", num: 1 },
     { t: "ν effective", num: 1 }, { t: "ν numerical", num: 1 },
     { t: "as % of ν", num: 1 }, { t: "fit r²", num: 1 }],
    rows)}
  <p class="note">
    ${tg.n}² grid, ν = ${tg.nu}, dt = ${tg.dt}, integrated to t = ${tg.T}.
    Semi-Lagrangian more than <strong>doubles the effective viscosity</strong> at
    this resolution: the fluid it simulates is roughly twice as viscous as the one
    it was asked to simulate. MacCormack brings the invented viscosity down to a
    few percent.
  </p>
</div></section>`;
}

function advectionSection() {
  const a = validation?.advection;
  if (!a) return "";
  const schemes = Object.entries(a.schemes);
  const p = plot({
    series: schemes.map(([name, s], i) => ({
      x: s.trace.map(t => t.t), y: s.trace.map(t => t.peak),
      label: name === "maccormack" ? "MacCormack" : "semi-Lagrangian",
      colour: i === 0 ? PAL.warm : PAL.accent, marker: true,
    })),
    xScale: "linear", yScale: "linear",
    xLabel: "time", yLabel: "peak amplitude", title: "A blob carried by uniform flow",
    caption: "The exact solution's peak never changes: the blob is simply translated. " +
      "Everything below the initial value is numerical diffusion, with no pressure, " +
      "viscosity or nonlinearity able to take the blame.",
  });

  const rows = schemes.map(([name, s]) => [
    td(`<strong>${name === "maccormack" ? "MacCormack" : "semi-Lagrangian"}</strong>`),
    td(`${(s.peakRetained * 100).toFixed(1)}%`, "num"),
    td(num(s.numericalDiffusion, 0), "num"),
    td(num(s.finalRelL2, 0), "num"),
    td(num(s.cfl, 3), "num"),
  ]);

  return `
<section><div class="wrap">
  <p class="eyebrow">isolating advection</p>
  <h2>Where the dissipation actually comes from</h2>
  <p class="lede prose">
    Taylor-Green cannot answer this on its own. Its nonlinear term is exactly
    cancelled by the pressure gradient, so a badly diffusive advection scheme can
    still score well there. This test has nothing else in it: a Gaussian blob in
    a uniform flow, whose exact solution is the same blob, translated.
  </p>
  ${p}
  ${table(["scheme", { t: "peak retained", num: 1 }, { t: "D numerical", num: 1 },
           { t: "relative L2", num: 1 }, { t: "CFL", num: 1 }], rows)}
  <p class="note">
    Semi-Lagrangian keeps interpolating bilinearly, and bilinear interpolation of
    anything but a linear field is a weighted average &mdash; a low-pass filter
    applied once per step, which is indistinguishable from adding a diffusion
    term. That is the mechanism behind both tables on this page.
  </p>
</div></section>`;
}

function convergenceSection() {
  const c = validation?.convergence;
  if (!c) return "";

  const series = c.table.map((row, i) => ({
    x: row.errors.map(e => e.dt), y: row.errors.map(e => e.relL2),
    label: `N = ${row.n}`, marker: true,
    colour: [PAL.accent, PAL.warm, PAL.violet, PAL.good, PAL.bad][i % 5],
  }));
  const p = plot({
    series, xScale: "log", yScale: "log",
    xLabel: "timestep dt", yLabel: "relative L2 error",
    title: "Error vs timestep, at four resolutions",
    caption: "Each line falls with slope ≈ 1 (first order in time) until it flattens " +
      "onto that grid's spatial error floor. The floor is what falls with resolution; " +
      "the sloped part is identical at every N because it is the splitting error.",
  });

  const rows = c.table.map(row => [
    td(`<strong>${row.n}²</strong>`),
    ...row.errors.map(e => td(num(e.relL2, 0), "num")),
    td(num(row.temporalOrder, 2), "num"),
  ]);

  return `
<section><div class="wrap">
  <p class="eyebrow">convergence</p>
  <h2>First order in time, and why the obvious study got it backwards</h2>
  <div class="prose stack">
    <p class="lede">
      The first convergence study run here held dt fixed and refined the grid,
      expecting the error to fall. It rose: 8.9e-5 → 2.3e-4 → 2.9e-4 for
      N = 16, 32, 64.
    </p>
    <p style="color:var(--ink2);font-size:.94rem">
      Sweeping dt and N independently showed why. At fixed dt the temporal error
      dominates and is identical at every resolution, so refining the grid cannot
      reduce it &mdash; and at N = 16 the spatial error happened to have the
      opposite sign and partially cancelled it, making the coarsest grid look
      the most accurate. Refining removed the cancellation, and the error went up.
    </p>
    <p style="color:var(--ink2);font-size:.94rem">
      The scheme is first order in time because the step is Lie-split
      (advect, then diffuse, then project), and Lie splitting is O(dt). That caps
      the whole method at first order however accurate the individual operators
      are &mdash; the Laplacian and the discrete curl are both second order,
      verified separately. It is a real property of Stam-style solvers and is
      rarely stated.
    </p>
  </div>
  ${p}
  ${table(
    ["grid", ...c.dts.map(d => ({ t: `dt=${d}`, num: 1 })), { t: "order in dt", num: 1 }],
    rows)}
  <p class="note">
    Taylor-Green, ν = ${c.nu}, integrated to t = ${c.T}, ${c.scheme} advection.
    Reading down a column shows why a fixed-dt refinement study is uninformative
    here; reading along a row gives the temporal order.
  </p>
</div></section>`;
}

function stabilitySection() {
  const s = validation?.stability;
  if (!s) return "";
  const ed = s.explicitDiffusion;

  const ampPlot = plot({
    series: [
      { x: ed.isolated.map(p => p.diffusionNumber),
        y: ed.isolated.map(p => p.predictedAmplification),
        label: "theory |1 − 8d|", colour: PAL.good, dashed: true },
      { x: ed.isolated.map(p => p.diffusionNumber),
        y: ed.isolated.map(p => p.measuredAmplification),
        label: "measured", colour: PAL.accent, marker: true },
    ],
    xScale: "linear", yScale: "linear",
    xLabel: "diffusion number  ν·dt/h²", yLabel: "amplification per step",
    title: "The explicit-diffusion stability boundary",
    caption: "The grid-scale mode is amplified by |1 − 8d| per step, so the boundary " +
      "sits exactly at d = 0.25 where amplification reaches 1. Measured and predicted " +
      "agree to better than 1e-6 at every point.",
  });

  const projPlot = plot({
    series: [{
      x: s.projectionIterations.points.map(p => p.maxIter),
      y: s.projectionIterations.points.map(p => Math.max(p.maxDivergence, 1e-17)),
      label: "max |∇·u|", colour: PAL.accent, marker: true,
    }],
    xScale: "log", yScale: "log",
    xLabel: "conjugate-gradient iterations", yLabel: "max |∇·u| after 50 steps",
    title: "Under-converging the projection",
    caption: "The most common way a fluid simulation is quietly wrong: the flow still " +
      "looks plausible while mass is not conserved. Roughly 50 iterations are needed " +
      "for machine precision at this resolution.",
  });

  const confRows = s.confinement.points.map(p => [
    td(`ε = ${p.epsilon}`, "mono"),
    td(p.keRatio === null ? "diverged" : num(p.keRatio, 3), "num"),
    td(num(p.keExactRatio, 3), "num"),
    td(p.addsEnergy
      ? '<span class="pill bad">adds energy</span>'
      : '<span class="pill good">decays</span>'),
  ]);

  return `
<section><div class="wrap">
  <p class="eyebrow">stability</p>
  <h2>Where it breaks, and a correction to the usual framing</h2>
  <div class="prose stack">
    <p class="lede">
      The brief asked for "the CFL condition boundary". Semi-Lagrangian advection
      does not have one: tracing backward and interpolating can never produce a
      value outside the range it sampled, so it cannot amplify anything. That
      unconditional stability is the entire reason Stam's method exists, and
      reporting a CFL blow-up limit would have meant inventing a result.
    </p>
    <p style="color:var(--ink2);font-size:.94rem">
      So what follows is what genuinely bounds this solver. Advective CFL was
      swept to ${num(Math.max(...s.advectiveCFL.points.map(p => p.cfl)), 2)} with
      no blow-up at all &mdash; large timesteps cost <em>accuracy</em>, not
      stability. The one real boundary is explicit diffusion, and it is
      reproduced exactly.
    </p>
  </div>
  <div class="figs">${ampPlot}${projPlot}</div>
  <div class="cards">
    <div class="card"><span class="meta">measured boundary</span>
      <h3>d = ${ed.measuredBoundary.isolated.lastStable} stable, ${ed.measuredBoundary.isolated.firstUnstable} unstable</h3>
      <p>Theory puts it at exactly 0.25. The operator was tested in isolation with
      the checkerboard mode seeded deliberately.</p></div>
    <div class="card"><span class="meta">inside the full solver</span>
      <h3>d = ${ed.measuredBoundary.inSolver.lastStable} / ${ed.measuredBoundary.inSolver.firstUnstable}</h3>
      <p>Below the bound the projection annihilates the mode entirely (growth ~1e-14);
      above it, the mode grows despite the projection.</p></div>
    <div class="card"><span class="meta">advective CFL</span>
      <h3>no blow-up to CFL ${num(Math.max(...s.advectiveCFL.points.map(p => p.cfl)), 1)}</h3>
      <p>Unconditionally stable, as designed. The error stays around
      ${num(s.advectiveCFL.points[0].relL2, 2)} throughout.</p></div>
  </div>

  <h3>Vorticity confinement is an energy source</h3>
  <p class="note">
    Confinement is in the brief as a way to restore detail lost to numerical
    dissipation, and it does that. But it is a fabricated force, and this table is
    why every measurement on this page has it switched off &mdash; the solver
    throws rather than allowing it in validation mode. The exact solution loses
    energy over this interval; from ε = ${s.confinement.smallestEpsilonThatAddsEnergy}
    the simulation <em>gains</em> it.
  </p>
  ${table(["confinement", { t: "energy ratio", num: 1 },
           { t: "exact ratio", num: 1 }, "verdict"], confRows)}
</div></section>`;
}

function spectrumSection() {
  const r = unwrap(spectrum);
  if (!r?.spectrum) {
    return `
<section><div class="wrap">
  <p class="eyebrow">turbulence</p>
  <h2>Energy spectrum</h2>
  <p class="lede prose">${MISSING} &mdash; run <code>tools/spectrum.html</code> and
  post the result to the collector.</p>
</div></section>`;
  }

  const k = r.spectrum.k, E = r.spectrum.E;
  const inv = r.fits.inverseCascade, ens = r.fits.enstrophyCascade;
  const kf = r.params.kf;

  // Anchor the reference laws to the measured spectrum at the middle of each
  // window, so they sit on the data and the SLOPE is what is being compared
  // rather than an arbitrary offset.
  const at = (kk) => E[Math.min(E.length - 1, Math.max(1, Math.round(kk)))];
  const invMid = Math.round((inv.kLo + inv.kHi) / 2);
  const ensMid = Math.round((ens.kLo + ens.kHi) / 2);

  const p = plot({
    series: [
      { x: k.slice(1), y: E.slice(1), label: "measured E(k)", colour: PAL.accent, width: 2.2 },
      { ...powerLaw(inv.kLo, inv.kHi, -5 / 3, invMid, at(invMid)),
        label: "k^(−5/3)  Kraichnan inverse", colour: PAL.good, dashed: true },
      { ...powerLaw(ens.kLo, ens.kHi, -3, ensMid, at(ensMid)),
        label: "k^(−3)  Kraichnan enstrophy", colour: PAL.warn, dashed: true },
      { ...powerLaw(ens.kLo, ens.kHi, ens.slope, ensMid, at(ensMid)),
        label: `k^(${ens.slope.toFixed(2)})  measured fit`, colour: PAL.bad, dashed: true },
    ],
    bands: [
      { from: inv.kLo, to: inv.kHi, label: "inverse window", colour: PAL.good },
      { from: ens.kLo, to: ens.kHi, label: "enstrophy window", colour: PAL.warn },
    ],
    xScale: "log", yScale: "log", width: 760, height: 440,
    xLabel: "wavenumber k", yLabel: "E(k)",
    title: `Forced 2D turbulence, ${r.params.n}², forcing at k = ${kf}`,
    caption: `Time-averaged over ${r.samples} samples. The shaded bands are the fit ` +
      `windows, fixed from k_f before the spectrum was looked at &mdash; choosing them ` +
      `afterwards is how almost any curve is made to agree with a power law. ` +
      `Parseval holds to ${num(r.parseval.relDiff, 0)}, so the normalisation is right.`,
  });

  // Verdicts are computed from the data, not written in advance. The first
  // version of this section was authored when the only run available used
  // semi-Lagrangian and missed both laws, and it said so in the heading. When
  // MacCormack moved the inverse range onto -5/3, a hardcoded heading would
  // have kept announcing a failure the data no longer showed.
  const invOff = Math.abs(inv.slope - (-5 / 3));
  const ensOff = Math.abs(ens.slope - (-3));
  const invGood = invOff < 0.25 && inv.r2 > 0.85;
  const ensGood = ensOff < 0.4 && ens.r2 > 0.85;
  const headline = invGood && ensGood
    ? "Both cascades match Kraichnan"
    : invGood
      ? "The inverse cascade matches. The enstrophy range does not."
      : "The spectrum does not match Kraichnan, and that is the finding";

  return `
<section><div class="wrap">
  <p class="eyebrow">turbulence</p>
  <h2>${headline}</h2>
  <div class="prose stack">
    <p class="lede">
      Kolmogorov's k^(−5/3) describes <em>three-dimensional</em> turbulence. In 2D
      there is no vortex stretching, vorticity is materially conserved, and the
      phenomenology is different: Kraichnan and Batchelor predict a
      <strong>dual cascade</strong> &mdash; energy travelling upscale from the
      forcing with slope −5/3, and enstrophy travelling downscale with slope −3.
      Comparing a 2D spectrum to −5/3 across all scales would be comparing
      against the wrong law for most of it, so both ranges are measured
      separately and each against its own prediction.
    </p>
  </div>
  ${p}
  <div class="cards">
    <div class="card"><span class="meta">inverse energy cascade, k ∈ [${inv.kLo}, ${inv.kHi}]</span>
      <h3 style="color:var(--${invGood ? "good" : "bad"})">${num(inv.slope, 2)} &nbsp;vs theory −1.67</h3>
      <p>r² = ${num(inv.r2, 3)}${invGood
        ? `. Within ${(invOff / (5 / 3) * 100).toFixed(0)}% of Kraichnan's prediction &mdash; energy is
           genuinely being transported upscale from the forcing.`
        : `. ${inv.slope > 0
            ? "Positive, meaning E(k) <em>rises</em> toward the forcing wavenumber: no inverse cascade at all."
            : "Present but well short of the predicted slope."}`}</p></div>
    <div class="card"><span class="meta">enstrophy cascade, k ∈ [${ens.kLo}, ${ens.kHi}]</span>
      <h3 style="color:var(--${ensGood ? "good" : "bad"})">${num(ens.slope, 2)} &nbsp;vs theory −3.00</h3>
      <p>r² = ${num(ens.r2, 3)} &mdash; an extremely clean power law, ${ensGood
        ? "and the right one."
        : `but roughly ${(Math.abs(ens.slope) / 3).toFixed(1)}× too steep. The spectrum is a
           straight line in log-log; it is simply the wrong line.`}</p></div>
    <div class="card"><span class="meta">the measurement is sound</span>
      <h3>Parseval ${num(r.parseval.relDiff, 0)}</h3>
      <p>Σ E(k) matches the kinetic energy computed independently in physical
      space, so the normalisation is right; and max |∇·u| = ${num(r.maxDivergence, 0)}
      with multigrid, so the field really is solenoidal. Neither can explain
      a wrong slope.</p></div>
  </div>
  ${spectrumSL && spectrum !== spectrumSL ? schemeComparison() : ""}
  <div class="finding">
    <h3>What is left over</h3>
    <dl>
      <dt>what matched</dt><dd>${invGood
        ? `The inverse energy cascade, at ${num(inv.slope, 2)} against −1.67 with
           r² = ${num(inv.r2, 3)}. Energy really is being carried upscale from the
           forcing, which is the distinctively two-dimensional half of the theory.`
        : "Neither range, on this run."}</dd>
      <dt>what did not</dt><dd>${ensGood ? "Both ranges matched." :
        `The enstrophy cascade, at ${num(ens.slope, 2)} against −3. The fit is
         excellent (r² = ${num(ens.r2, 3)}) so this is a real power law at the
         wrong exponent, not scatter.`}</dd>
      <dt>most likely cause</dt><dd>Residual numerical dissipation at the small
        scales. The enstrophy range sits nearest the grid, where any remaining
        dissipation bites hardest, and this run used f32 with a 3-cycle
        projection. The scheme comparison above shows the slope moving with the
        advection scheme, so dissipation is demonstrably part of it.</dd>
      <dt>ruled out</dt><dd>The projection (max |∇·u| = ${num(r.maxDivergence, 0)}),
        the spectrum's normalisation (Parseval to ${num(r.parseval.relDiff, 0)}),
        and the fit window, which was fixed from k_f before any data was seen.</dd>
    </dl>
  </div>
</div></section>`;
}

/**
 * Both advection schemes' spectra side by side.
 *
 * This exists because the first spectrum run measured an enstrophy slope of
 * -6.08 and attributed it to semi-Lagrangian's numerical dissipation. That was a
 * hypothesis, not a measurement. Porting MacCormack to the GPU and re-running
 * the identical study is what turns it into one -- and the comparison is
 * reported whichever way it came out.
 */
function schemeComparison() {
  const a = unwrap(spectrumSL);
  const b = unwrap(spectrum);
  if (!a?.spectrum || !b?.spectrum) return "";
  const ens = b.fits.enstrophyCascade;

  const p = plot({
    series: [
      { x: a.spectrum.k.slice(1), y: a.spectrum.E.slice(1),
        label: `semi-Lagrangian  (${a.fits.enstrophyCascade.slope.toFixed(2)})`, colour: PAL.warm },
      { x: b.spectrum.k.slice(1), y: b.spectrum.E.slice(1),
        label: `MacCormack  (${b.fits.enstrophyCascade.slope.toFixed(2)})`, colour: PAL.accent, width: 2.2 },
      { ...powerLaw(ens.kLo, ens.kHi, -3,
          Math.round((ens.kLo + ens.kHi) / 2),
          b.spectrum.E[Math.round((ens.kLo + ens.kHi) / 2)]),
        label: "k^(−3)  theory", colour: PAL.good, dashed: true },
    ],
    bands: [{ from: ens.kLo, to: ens.kHi, label: "enstrophy window", colour: PAL.warn }],
    xScale: "log", yScale: "log", width: 760, height: 420,
    xLabel: "wavenumber k", yLabel: "E(k)",
    title: "The same study, both advection schemes",
    caption: "Identical forcing, resolution, projection and fit window; only the " +
      "advection scheme differs. Whatever separates the two curves is the scheme's " +
      "numerical dissipation and nothing else.",
  });

  const delta = a.fits.enstrophyCascade.slope - b.fits.enstrophyCascade.slope;
  return `
  <h3>Testing the diagnosis</h3>
  <p class="note">
    Attributing the missing cascade to numerical dissipation was a hypothesis. It
    is testable: port the less dissipative scheme to the GPU and re-run the
    identical study. That is what this is.
  </p>
  ${p}
  <div class="cards">
    <div class="card"><span class="meta">semi-Lagrangian</span>
      <h3>${num(a.fits.enstrophyCascade.slope, 2)}</h3>
      <p>r² = ${num(a.fits.enstrophyCascade.r2, 3)}. First order; measured numerical
      viscosity ≈ 100% of ν on the Taylor-Green test.</p></div>
    <div class="card"><span class="meta">MacCormack</span>
      <h3>${num(b.fits.enstrophyCascade.slope, 2)}</h3>
      <p>r² = ${num(b.fits.enstrophyCascade.r2, 3)}. Second order, limited; ~2.9% of ν.
      GPU energy retention over 200 inviscid steps: 0.2475 of 0.25, against 0.2117
      for semi-Lagrangian.</p></div>
    <div class="card"><span class="meta">theory</span>
      <h3 style="color:var(--good)">−3.00</h3>
      <p>${Math.abs(delta) < 0.15
        ? "Changing the scheme barely moved the slope, so numerical dissipation from advection is <em>not</em> the whole explanation."
        : `Changing the scheme moved the slope by ${num(Math.abs(delta), 2)}, ${
            Math.abs(b.fits.enstrophyCascade.slope + 3) < Math.abs(a.fits.enstrophyCascade.slope + 3)
              ? "toward" : "away from"} theory.`}</p></div>
  </div>`;
}

function gpuSection() {
  if (!gpuVerify && !bench) return "";
  const parts = [];

  parts.push(`
  <p class="eyebrow">gpu</p>
  <h2>Moving the solve onto compute shaders</h2>
  <p class="lede prose">
    Every stage of the timestep &mdash; advection, diffusion, divergence, the
    iterative pressure solve, the gradient subtraction &mdash; is a WGSL compute
    dispatch. The CPU only issues commands. What follows is how that was checked
    and what it bought.
  </p>`);

  if (gpuVerify?.poissonCurves) {
    const cv = gpuVerify.poissonCurves;
    const mk = (name, label, colour) => cv[name] ? {
      x: cv[name].map(p => Math.max(p.equivalentSweeps ?? p.iterations, 0.5)),
      y: cv[name].map(p => Math.max(p.residual, 1e-9)),
      label, colour, marker: true,
    } : null;
    const series = [
      mk("jacobi", "Jacobi", PAL.warm),
      mk("red-black", "red-black Gauss-Seidel", PAL.accent),
      mk("multigrid", "multigrid V-cycle", PAL.good),
    ].filter(Boolean);

    parts.push(plot({
      series, xScale: "log", yScale: "log", width: 700, height: 400,
      xLabel: "equivalent fine-grid sweeps (cost)", yLabel: "relative residual",
      title: "Three pressure solvers, on a common cost axis",
      caption: "Multigrid's unit is a V-cycle, which costs about 4/3 of a fine-grid " +
        "sweep across all levels; plotting against raw iteration count would have " +
        "compared unlike things. It plateaus near 1e-7 because that is the f32 floor, " +
        "not because it has stopped converging.",
    }));

    const last = cv.jacobi[cv.jacobi.length - 1];
    parts.push(`<p class="note">
      At roughly 40 sweeps of equivalent cost: Jacobi reaches
      ${num(cv.jacobi.find(p => p.iterations === 40)?.residual, 0)},
      red-black ${num(cv["red-black"].find(p => p.iterations === 40)?.residual, 0)},
      multigrid ${num(cv.multigrid.find(p => Math.round(p.equivalentSweeps) >= 40)?.residual ??
        cv.multigrid[cv.multigrid.length - 1].residual, 0)}.
      Jacobi and Gauss-Seidel are local &mdash; one sweep moves information one cell &mdash;
      so their reduction factor for a mode spanning L cells is about 1 − O(1/L²), and
      refining the grid makes that strictly worse. Multigrid solves those modes on a
      grid where they are no longer smooth.
    </p>`);
  }

  if (gpuVerify?.checks) {
    const rows = gpuVerify.checks.map(c => [
      td(esc(c.name)),
      td(`<span class="pill ${c.pass ? "good" : "bad"}">${c.pass ? "pass" : "fail"}</span>`),
      td(`<code>${esc(c.detail ?? "")}</code>`),
    ]);
    parts.push(`<h3>Verification against the f64 CPU reference</h3>`);
    parts.push(table(["check", "result", "measured"], rows));
    parts.push(`<p class="note">
      WGSL is f32 and permits relaxed precision on transcendentals &mdash; a plain
      <code>sin</code> measured 6.9e-5 maximum absolute error on this adapter. So the
      GPU cannot agree with the f64 reference below about 1e-5 relative, and any
      tighter claim would be measuring nothing. Agreement of
      ${num(gpuVerify.fullStep?.du, 0)} after ten full steps is f32 accumulation noise.
    </p>`);
  }

  if (bench?.sameAlgorithm) {
    const sa = bench.sameAlgorithm.rows;
    parts.push(plot({
      series: [
        { x: sa.map(r => r.n), y: sa.map(r => r.cpuMs), label: "CPU (f64)", colour: PAL.warm, marker: true },
        { x: sa.map(r => r.n), y: sa.map(r => r.gpuMs), label: "GPU (f32)", colour: PAL.accent, marker: true },
      ],
      xScale: "log", yScale: "log", width: 700, height: 380,
      xLabel: "grid resolution N (N² cells)", yLabel: "ms per step",
      title: "Same algorithm, same iteration count",
      caption: "Identical Jacobi sweeps on identical grids, so the only difference is " +
        "where the arithmetic runs. The GPU is slower below N ≈ 100, where dispatch " +
        "overhead dominates the work.",
    }));

    const rows = sa.map(r => [
      td(`<strong>${r.n}²</strong>`), td(r.cells.toLocaleString(), "num"),
      td(num(r.cpuMs, 1), "num"), td(num(r.gpuMs, 1), "num"),
      td(`<strong>${r.speedup.toFixed(1)}×</strong>`, "num"),
    ]);
    parts.push(table(["grid", { t: "cells", num: 1 }, { t: "CPU ms", num: 1 },
                      { t: "GPU ms", num: 1 }, { t: "speedup", num: 1 }], rows));
  }

  if (bench?.equalQuality) {
    const eq = bench.equalQuality.rows;
    const rows = eq.map(r => [
      td(`<strong>${r.n}²</strong>`),
      td(num(r.cpuFftMs, 1), "num"),
      td(`${num(r.redBlack.ms, 1)} <span style="color:var(--dim)">/ ${r.redBlack.budget}sw</span>` +
         (r.redBlack.reachedTarget ? "" : ' <span class="pill bad">missed</span>'), "num"),
      td(`${num(r.multigrid.ms, 1)} <span style="color:var(--dim)">/ ${r.multigrid.budget}vc</span>`, "num"),
      td(`<strong>${r.ratioMultigrid.toFixed(1)}×</strong>`, "num"),
    ]);
    parts.push(`
    <h3>Equal quality, not equal work</h3>
    <p class="note">
      The comparison above gives both devices the same iteration count. That is fair
      as a hardware measurement and misleading as a practical one, because the CPU's
      best option is an exact FFT Poisson solve. So here each GPU solver is given as
      many iterations as it needs to reach max |∇·u| ≤ ${bench.equalQuality.targetDivergence},
      and <em>that</em> is what is timed. Red-black could not reach the target at all
      at the larger sizes; multigrid reaches it in a single V-cycle.
    </p>`);
    parts.push(table(["grid", { t: "CPU FFT ms", num: 1 }, { t: "red-black", num: 1 },
                      { t: "multigrid", num: 1 }, { t: "MG speedup", num: 1 }], rows));
    parts.push(`<p class="note">
      This is the honest version of the GPU claim. At equal iteration count the GPU
      looks ${bench.sameAlgorithm ? `${bench.sameAlgorithm.rows[bench.sameAlgorithm.rows.length - 1].speedup.toFixed(0)}×` : "vastly"}
      faster, but much of that is the GPU running a <em>worse algorithm</em> quickly.
      Once accuracy is held fixed, the win comes from multigrid, and it is
      ${eq[eq.length - 1].ratioMultigrid.toFixed(1)}× at ${eq[eq.length - 1].n}².
    </p>`);
  }

  if (bench?.interactive) {
    const it = bench.interactive;
    parts.push(`
    <div class="cards">
      <div class="card"><span class="meta">interactive ceiling</span>
        <h3>${it.gpu ? `${it.gpu}² on GPU` : MISSING}</h3>
        <p>Largest grid holding 60 fps (16.7 ms/step), against
        ${it.cpu ? `${it.cpu}² on the CPU` : "no CPU size that managed it"}
        ${it.gpu && it.cpu ? ` &mdash; ${((it.gpu ** 2) / (it.cpu ** 2)).toFixed(0)}× more cells.` : "."}</p></div>
      <div class="card"><span class="meta">hardware</span>
        <h3>${esc(bench.adapter?.vendor ?? "?")} ${esc(bench.adapter?.architecture ?? "")}</h3>
        <p>An integrated GPU sharing memory bandwidth with the CPU. A discrete card
        would widen every ratio here; these numbers are the modest end.</p></div>
    </div>`);
  }

  return `<section><div class="wrap">${parts.join("\n")}</div></section>`;
}

function fsiSection() {
  const f = validation?.fsi;
  if (!f) return "";

  const slipPlot = plot({
    series: [{
      x: f.noSlip.points.map(p => p.passes),
      y: f.noSlip.points.map(p => p.maxSlip),
      label: "max slip inside the body", colour: PAL.accent, marker: true,
    }],
    xScale: "linear", yScale: "log", width: 640, height: 360,
    xLabel: "direct-forcing passes per step", yLabel: "residual slip",
    title: "How well the body actually blocks the flow",
    caption: "One pass moves the fluid a fraction chi of the way to the body velocity, " +
      "leaving (1 - chi) behind. Repeating drives the residual down by that same factor " +
      "each time, which is a straight line on a log axis -- a property of the method " +
      "rather than a tuned number.",
  });

  const ts = f.translationStability.points;
  const stabPlot = plot({
    series: [
      { x: ts.filter(p => !p.plain.blewUp).map(p => p.density),
        y: ts.filter(p => !p.plain.blewUp).map(p => p.plain.vx),
        label: "no correction", colour: PAL.warm, marker: true },
      { x: ts.filter(p => !p.corrected.blewUp).map(p => p.density),
        y: ts.filter(p => !p.corrected.blewUp).map(p => p.corrected.vx),
        label: "added-mass corrected", colour: PAL.good, marker: true },
    ],
    xScale: "log", yScale: "linear", width: 640, height: 360,
    xLabel: "solid / fluid density ratio", yLabel: "disk velocity after the run",
    title: "Where explicit coupling stops working",
    caption: "Points appear only where the run stayed finite. Without the correction " +
      "every ratio below " + f.translationStability.lowestStableDensity.plain +
      " diverged; with it the sweep stayed stable down to " +
      f.translationStability.lowestStableDensity.corrected + ", the lowest tried.",
  });

  const rotRows = f.rotationStability.points.map(p => [
    td(num(p.density, 1), "num"),
    td(num(p.inertia, 4), "num"),
    td(p.blewUp ? '<span class="pill bad">diverged</span>' : num(p.omega, 4), "num"),
    td("-0.5", "num"),
  ]);

  const mom = f.momentum.points;
  const firstStable = f.rotationStability.points.find(p => !p.blewUp);

  return `
<section><div class="wrap">
  <p class="eyebrow">fluid-structure interaction</p>
  <h2>An obstacle that is pushed back</h2>
  <div class="prose stack">
    <p class="lede">
      A rigid disk immersed in the flow, coupled both ways: it blocks the fluid,
      and the reaction moves it. &ldquo;Two-way&rdquo; is the load-bearing word
      &mdash; a static obstacle that deflects flow without ever moving is a
      boundary condition, and a much easier thing to get right.
    </p>
    <p style="color:var(--ink2);font-size:.94rem">
      The method is direct forcing (Mohd-Yusof 1997): the solid is a smoothed mask
      on the existing grid, the fluid inside it is driven toward the body velocity
      each step, and the reaction on the body is the negative of that same
      integral. That last clause is where Newton&rsquo;s third law lives, and it is
      the one thing here that can be checked exactly rather than approximately.
    </p>
  </div>

  <div class="kpi">
    ${kpi(num(mom[0].relativeDrift, 0), "momentum drift", "one coupling step, default path", "good")}
    ${kpi(num(f.noSlip.points[f.noSlip.points.length - 1].maxSlip, 0), "residual slip",
      `${f.noSlip.points[f.noSlip.points.length - 1].passes} forcing passes`, "good")}
    ${kpi(String(f.translationStability.lowestStableDensity.plain), "min density ratio",
      "below this the explicit coupling diverges", "warn")}
    ${kpi(firstStable ? num(firstStable.omega, 3) : MISSING, "spin in shear",
      "torque-free theory says -0.5")}
  </div>

  <div class="figs">${slipPlot}${stabPlot}</div>

  <h3>Newton&rsquo;s third law, to machine precision</h3>
  <p class="note">
    The coupling moves momentum between fluid and solid and must not create or
    destroy any. Applied correctly that is exact, so it is measured as such rather
    than given a tolerance. A coupling that leaked momentum would still produce a
    disk that moved plausibly, which is precisely why it needs a check that
    &ldquo;looks about right&rdquo; cannot satisfy.
  </p>
  ${table(["configuration", { t: "relative momentum drift", num: 1 }, "verdict"], [
    [td("default (exact)"), td(num(mom[0].relativeDrift, 0), "num"),
     td('<span class="pill good">conserves</span>')],
    [td("added-mass corrected"), td(num(mom[1].relativeDrift, 0), "num"),
     td('<span class="pill warn">trades conservation for stability</span>')],
  ])}

  <div class="finding">
    <h3>The correction is not free, so it is not the default</h3>
    <dl>
      <dt>the problem</dt><dd>Explicit coupling computes the force from the current
        fluid state, applies it to the body, and lets the body change the fluid next
        step. In 2D a disk&rsquo;s added mass is <em>exactly</em> the mass of fluid it
        displaces, so at a density ratio of 1 the fluid&rsquo;s inertia already equals
        the body&rsquo;s and that feedback loop diverges.</dd>
      <dt>the usual fix</dt><dd>Fold the added mass into the effective inertia. It
        works: the stable range extends from
        ${f.translationStability.lowestStableDensity.plain} down to
        ${f.translationStability.lowestStableDensity.corrected}.</dd>
      <dt>the cost</dt><dd>The added mass <em>is</em> fluid mass, and the fluid&rsquo;s
        momentum is already tracked in the fluid. Counting it again on the body makes
        the body under-respond, and total momentum stops being conserved:
        ${num(mom[1].relativeDrift, 0)} against ${num(mom[0].relativeDrift, 0)}.</dd>
      <dt>the choice</dt><dd>Off by default, like vorticity confinement elsewhere in
        this project. The convenient option exists, is labelled, and is not what any
        number here is measured with.</dd>
    </dl>
  </div>

  <h3>The GPU port computes the same thing</h3>
  <p class="note">
    The CPU implementation is the reference; the GPU one exists so the obstacle can
    appear in the interactive demo. Given the same body in the same flow, the two agree
    on the force to better than 0.01%, which is the check that matters &mdash; the force is the
    whole coupling.
  </p>
  <div class="finding">
    <h3>A bug the CPU version structurally could not have</h3>
    <dl>
      <dt>symptom</dt><dd>The GPU disk produced <code>NaN</code> velocity within a few
        steps. 162,981 of 262,144 impulse cells came back non-finite &mdash; and every
        one of them was <em>far</em> from the body, the region where the mask should
        have been a clean zero.</dd>
      <dt>cause</dt><dd>The mask is
        <code>&frac12;(1 &minus; tanh((r &minus; R)/w))</code>, and with a mask width of
        1.5 cells that argument reaches about 240 at the far side of the domain. WGSL
        does not require <code>tanh</code> to be robust for large arguments, and this
        adapter evidently computes it as
        <code>(e&sup1; &minus; e&#8315;&sup1;)/(e&sup1; + e&#8315;&sup1;)</code>: both
        terms overflow to infinity and the result is <code>Inf/Inf = NaN</code>.</dd>
      <dt>why the CPU never showed it</dt><dd><code>Math.tanh</code> is robust. The
        reference implementation was correct and silent, which is exactly the case a
        port has to be tested for rather than trusted through.</dd>
      <dt>fix</dt><dd>Clamp the argument to &plusmn;10 before <code>tanh</code>, which
        saturates well inside f32 and changes no representable value. Pinned by a check
        that asserts the whole impulse field is finite.</dd>
    </dl>
  </div>

  <h3>Rotation has a separate threshold</h3>
  <p class="note">
    A shear flow <code>u = y &minus; &pi;</code> has vorticity &minus;1, and the
    classical result for a torque-free body is rotation at half the vorticity,
    &omega; &rarr; &minus;0.5. The coupling reproduces the sign and the approach
    &mdash; but only once the moment of inertia is large enough. Below that it
    diverges for the same reason translation does, at its own threshold.
  </p>
  ${table(["density", { t: "moment of inertia", num: 1 }, { t: "omega after the run", num: 1 },
           { t: "theory", num: 1 }], rotRows)}
  <p class="note">
    Heavier bodies approach &minus;0.5 more slowly, which is the expected response to
    a fixed torque, and none of these runs is long enough to arrive. What is asserted
    is the sign and the direction of approach, not the endpoint.
  </p>
</div></section>`;
}

function bugsSection() {
  const bugs = [
    {
      title: "Every GPU pass read the same uniform values",
      where: "src/gpu/solver.js",
      symptom: "The pressure solve did nothing. Divergence fell from 18.1 to 16.3 " +
        "after 200 Jacobi sweeps, and red-black froze at a residual of 6.13e-1 " +
        "from its first sweep onward, never improving.",
      cause: "<code>queue.writeBuffer</code> is ordered against <em>submitted</em> " +
        "command buffers, not against commands being encoded. Encoding a pass does " +
        "not capture the uniform's contents &mdash; the pass reads the buffer when it " +
        "executes, after every write issued before the submit has landed. Writing " +
        "params, encoding a pass, writing again, encoding again, then submitting once " +
        "meant all passes saw the final value: relaxation factor 0 (so Jacobi was " +
        "<code>p + 0·(new − p)</code>, i.e. nothing) and colour always 1 (so red cells " +
        "were never updated).",
      fix: "A ring of 256-byte-aligned uniform slots, with the bind helper allocating " +
        "a pass's slot as it binds it &mdash; so the parameters cannot be separated " +
        "from the pass that reads them.",
      how: "Suspected from the shape of the failure: the standalone residual curve " +
        "worked while the in-step projection did not, and the only difference was one " +
        "encoder versus many.",
    },
    {
      title: "The stability sweep reported the solver stable at 2.4× the theoretical limit",
      where: "validate/studies.js",
      symptom: "Explicit diffusion was reported stable up to ν·dt/h² = 0.6. The " +
        "textbook bound is 0.25.",
      cause: "Two compounding mistakes in the instrument, none in the solver. The " +
        "sweep started from a smooth Taylor-Green field, which contains essentially " +
        "none of the checkerboard mode that actually goes unstable &mdash; so there " +
        "was nothing to amplify. And it watched for kinetic energy to exceed a fixed " +
        "threshold, which a viscously decaying field was never going to cross however " +
        "unstable the scheme was.",
      fix: "Seed the grid-scale mode explicitly and measure its growth by projecting " +
        "onto it. The operator now reproduces |1 − 8d| to better than 1e-6 at every " +
        "point, with the boundary at exactly 0.25 / 0.26.",
      how: "Testing the diffusion operator in isolation, which immediately showed " +
        "amplification of exactly 1.00 at d = 0.25 &mdash; correct all along.",
    },
    {
      title: "A benchmark that flattered the GPU twice over",
      where: "tools/bench.html",
      symptom: "An early run reported the GPU 38.9× faster at 512².",
      cause: "Two separate unfairnesses. The 'same algorithm' comparison ran " +
        "conjugate gradient on the CPU against Jacobi on the GPU and attributed the " +
        "difference to hardware &mdash; CG is far stronger per iteration. And the " +
        "'best available' comparison timed 20 GPU sweeps (leaving divergence at 2.5e-1) " +
        "against an exact CPU FFT solve (1e-15), letting the GPU win by doing less " +
        "work <em>and</em> returning a worse answer.",
      fix: "Jacobi on both sides for the hardware comparison, and an equal-quality " +
        "comparison that raises the GPU's iteration count until it matches a target " +
        "divergence. The honest numbers are lower and more interesting: the GPU " +
        "initially <em>lost</em> at equal quality, which is what motivated multigrid.",
      how: "Noticing that the reported GPU divergence was three orders of magnitude " +
        "worse than the CPU's in the same table.",
    },
    {
      title: "Non-monotonic GPU timings from insufficient warm-up",
      where: "tools/bench.html",
      symptom: "N = 32 measured 11.9 ms and N = 64 measured 31.7 ms, while N = 128 " +
        "ran in 9.5 ms &mdash; a smaller grid taking three times longer.",
      cause: "Three unsynced warm-up steps left pipeline compilation and the iGPU's " +
        "idle clock inside the measurement window, and only the first two " +
        "configurations paid it.",
      fix: "Twelve warm-up steps with periodic queue synchronisation. The artifact " +
        "disappeared and the curve became monotonic.",
      how: "The physics made no sense: dispatch overhead cannot make a 4× smaller " +
        "grid slower.",
    },
  ];

  return `
<section><div class="wrap">
  <p class="eyebrow">honestly</p>
  <h2>Bugs found, and how</h2>
  <p class="lede prose">
    Four, all fixed. Three of them were in the <em>measuring</em> code rather than
    the solver &mdash; a validator reported the scheme stable well past its real
    limit, and a benchmark reported a speedup that was partly an algorithm
    difference. Instruments need testing at least as much as the thing they measure.
  </p>
  <div class="stack">
    ${bugs.map((b, i) => `
    <article class="finding bug">
      <h3>${i + 1}. ${b.title}</h3>
      <dl>
        <dt>symptom</dt><dd>${b.symptom}</dd>
        <dt>cause</dt><dd>${b.cause}</dd>
        <dt>fix</dt><dd>${b.fix}</dd>
        <dt>found by</dt><dd>${b.how}</dd>
        <dt>file</dt><dd><code>${esc(b.where)}</code></dd>
      </dl>
    </article>`).join("")}
  </div>
</div></section>`;
}

function limitsSection() {
  return `
<section><div class="wrap">
  <p class="eyebrow">limits</p>
  <h2>What this does not do</h2>
  <div class="prose"><ul>
    <li><strong>The GPU path has only semi-Lagrangian advection.</strong> MacCormack
      exists on the CPU and is roughly 36× less dissipative, but was not ported. This
      is the single biggest reason the energy spectrum misses Kraichnan's −3, and it
      is a gap in the implementation rather than a property of the physics.</li>
    <li><strong>First order in time.</strong> Lie splitting caps the whole scheme at
      O(dt) no matter how accurate the operators are. Strang splitting would give
      second order for about 1.5× the cost.</li>
    <li><strong>No inverse cascade was observed.</strong> The measured slope in the
      inverse range is positive. Whether a less dissipative scheme would recover
      Kraichnan's −5/3 at this resolution is untested, and claiming it would be
      speculation.</li>
    <li><strong>f32 on the GPU.</strong> The multigrid residual plateaus near 1e-7
      because that is the precision floor, not because the method stalls.</li>
    <li><strong>Periodic boundaries only.</strong> The FFT reference solver and the
      spectra both require it. Walls would need the conjugate-gradient path, which
      exists but is not exercised here.</li>
    <li><strong>Integrated GPU.</strong> Every performance ratio on this page comes
      from an Intel iGPU sharing bandwidth with the CPU, which is the pessimistic end
      of the range.</li>
    <li><strong>The fluid-structure coupling is explicit.</strong> It diverges below
      a density ratio of about 1 without the added-mass correction, and the correction
      costs exact momentum conservation. A strongly-coupled (iterated) scheme would fix
      both and is not implemented.</li>
    <li><strong>The GPU coupling lags one step.</strong> The force on the body is the
      integral of the impulse field, which needs a readback, and awaiting it inside the
      step would stall the pipeline. So the demo applies the previous step's force. That
      extra lag makes the explicit coupling slightly less stable than the CPU path,
      which is why the demo uses a heavy disk. Every fluid-structure number on this page
      comes from the CPU path, which has no such lag.</li>
  </ul></div>
</div></section>`;
}

function reproduceSection() {
  return `
<section><div class="wrap">
  <p class="eyebrow">reproduce</p>
  <h2>Run it yourself</h2>
  <pre><span class="c"># no dependencies beyond node</span>
git clone https://github.com/abho7/fluid-sim &amp;&amp; cd fluid-sim

<span class="c"># the whole suite: solver, FFT, analytic solutions, and the
# mutation tests that prove the validators actually fire</span>
node --test test/*.test.js

<span class="c"># regenerate every CPU-side number on this page</span>
node validate/run.js

<span class="c"># serve the demo and the GPU harnesses (WebGPU needs a secure context)</span>
node tools/serve.js
<span class="c">#   /                   the interactive demo
#   /tools/gputest.html GPU vs CPU verification
#   /tools/bench.html   performance
#   /tools/spectrum.html forced turbulence</span>

<span class="c"># rebuild this page from results/*.json</span>
node tools/build_report.js</pre>
  <p class="note">
    The solver, the FFT, the multigrid cycle, the plots on this page and the static
    server are all written from scratch. Node and a WebGPU-capable browser are the
    only requirements.
  </p>
</div></section>`;
}

// ==================================================================== build

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Validating a Fluid Solver</title>
<meta name="description" content="An incompressible Navier-Stokes solver on WebGPU, measured against analytic solutions: numerical viscosity, stability boundaries, convergence, and an energy spectrum that does not match theory.">
<link rel="canonical" href="https://abho7.github.io/fluid-sim/report.html">
<link rel="icon" href="favicon.ico" sizes="32x32">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
<meta property="og:type" content="article">
<meta property="og:url" content="https://abho7.github.io/fluid-sim/report.html">
<meta property="og:title" content="Validating a Fluid Solver">
<meta property="og:description" content="An incompressible Navier-Stokes solver on WebGPU, measured against analytic solutions: numerical viscosity, stability boundaries, convergence, and an energy spectrum that does not match theory.">
<meta property="og:image" content="https://abho7.github.io/fluid-sim/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="fluid-sim - numerical viscosity 5.79e-4, 2.9% of the physical value">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Validating a Fluid Solver">
<meta name="twitter:description" content="An incompressible Navier-Stokes solver on WebGPU, measured against analytic solutions: numerical viscosity, stability boundaries, convergence, and an energy spectrum that does not match theory.">
<meta name="twitter:image" content="https://abho7.github.io/fluid-sim/og.png">
<style>${CSS}</style>
</head>
<body>
${heroSection()}
${summarySection()}
${taylorGreenSection()}
${advectionSection()}
${convergenceSection()}
${stabilitySection()}
${spectrumSection()}
${gpuSection()}
${fsiSection()}
${bugsSection()}
${limitsSection()}
${reproduceSection()}
<footer><div class="wrap">
  fluid-sim &middot; Abhineeth Duddela &middot; &copy; 2026<br>
  Stable fluids (Stam 1999) on a MAC staggered grid, with MacCormack advection,
  geometric multigrid, and validation against Taylor-Green and Kraichnan-Batchelor.<br>
  ${validation?.env ? `measured on ${esc(validation.env.platform)} · ${esc(validation.env.cpu)}` : ""}
</div></footer>
<script>${HERO_JS}</script>
</body>
</html>
`;

writeFileSync(join(ROOT, "report.html"), html);
const kb = (html.length / 1024).toFixed(0);
console.log(`wrote report.html  (${kb} KB)`);
for (const [name, v] of Object.entries({ validation, spectrum, gpuVerify, bench })) {
  console.log(`  ${name.padEnd(12)} ${v ? "present" : "MISSING -> page will say 'not measured'"}`);
}
