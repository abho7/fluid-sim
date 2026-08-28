/**
 * Hand-authored SVG plotting.
 *
 * No charting library, for the same reason there are no other dependencies
 * here: the FFT, the solver and the multigrid cycle are all from scratch, and a
 * plotting dependency would be the one vendored abstraction in the stack.
 *
 * These are scientific plots, so the defaults are chosen accordingly: log axes
 * where the data spans decades, reference lines for the theory being compared
 * against, explicit fit windows drawn as shaded bands (because a slope quoted
 * without its window is not checkable), and no smoothing of any kind.
 */

const PAL = {
  ink: "#e8edf3", dim: "#7d8794", line: "#232a35", grid: "#1a2029",
  accent: "#69d9ff", warm: "#ff9d5c", good: "#7ee787", bad: "#ff6b81",
  violet: "#b18cff",
};

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/** Nice log-decade ticks covering [lo, hi]. */
function logTicks(lo, hi) {
  const out = [];
  const a = Math.floor(Math.log10(lo)), b = Math.ceil(Math.log10(hi));
  for (let e = a; e <= b; e++) {
    const v = Math.pow(10, e);
    if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
  }
  return out;
}

function linTicks(lo, hi, n = 5) {
  const out = [];
  for (let i = 0; i <= n; i++) out.push(lo + (hi - lo) * i / n);
  return out;
}

const fmtLog = v => {
  const e = Math.round(Math.log10(v));
  return `10${sup(e)}`;
};
const SUPS = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³",
  4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const sup = n => String(n).split("").map(c => SUPS[c] ?? c).join("");

const fmtNum = v => {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) return v.toExponential(1);
  if (a >= 10) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  return v.toFixed(2);
};

/**
 * A general 2D plot.
 *
 * @param {object} o
 * @param {Array<{x:number[],y:number[],label:string,colour?:string,dashed?:boolean,
 *                 marker?:boolean}>} o.series
 * @param {"log"|"linear"} o.xScale
 * @param {"log"|"linear"} o.yScale
 * @param {Array<{from:number,to:number,label:string,colour?:string}>} [o.bands]
 *   Shaded fit windows, drawn behind the data.
 */
export function plot({
  series, xScale = "linear", yScale = "linear",
  xLabel = "", yLabel = "", title = "", caption = "",
  width = 640, height = 380, bands = [], legend = true,
}) {
  const m = { l: 62, r: 18, t: title ? 30 : 14, b: 46 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;

  const all = series.filter(s => s.x.length);
  if (!all.length) return `<figure><figcaption>${esc(caption)}</figcaption></figure>`;

  const positive = v => v > 0;
  const xs = all.flatMap(s => s.x).filter(v => xScale === "log" ? positive(v) : Number.isFinite(v));
  const ys = all.flatMap(s => s.y).filter(v => yScale === "log" ? positive(v) : Number.isFinite(v));
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);

  if (xScale === "linear") { const p = (x1 - x0) * 0.04 || 1; x0 -= p; x1 += p; }
  if (yScale === "linear") { const p = (y1 - y0) * 0.08 || 1; y0 -= p; y1 += p; }
  else { y0 *= 0.6; y1 *= 1.6; }

  const sx = v => xScale === "log"
    ? m.l + iw * (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0))
    : m.l + iw * (v - x0) / (x1 - x0);
  const sy = v => yScale === "log"
    ? m.t + ih * (1 - (Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0)))
    : m.t + ih * (1 - (v - y0) / (y1 - y0));

  const parts = [];

  // fit-window bands, behind everything
  for (const b of bands) {
    const bx0 = sx(Math.max(b.from, x0)), bx1 = sx(Math.min(b.to, x1));
    parts.push(`<rect x="${bx0.toFixed(1)}" y="${m.t}" width="${(bx1 - bx0).toFixed(1)}" ` +
      `height="${ih}" fill="${b.colour ?? PAL.accent}" opacity=".07"/>`);
    if (b.label) {
      parts.push(`<text x="${((bx0 + bx1) / 2).toFixed(1)}" y="${m.t + 12}" ` +
        `text-anchor="middle" font-size="9" fill="${b.colour ?? PAL.accent}" ` +
        `opacity=".85">${esc(b.label)}</text>`);
    }
  }

  // grid + axes
  const xt = xScale === "log" ? logTicks(x0, x1) : linTicks(x0, x1);
  const yt = yScale === "log" ? logTicks(y0, y1) : linTicks(y0, y1);
  for (const t of xt) {
    const X = sx(t);
    parts.push(`<line x1="${X.toFixed(1)}" y1="${m.t}" x2="${X.toFixed(1)}" y2="${m.t + ih}" stroke="${PAL.grid}"/>`);
    parts.push(`<text x="${X.toFixed(1)}" y="${m.t + ih + 16}" text-anchor="middle" ` +
      `font-size="10" fill="${PAL.dim}">${xScale === "log" ? fmtLog(t) : fmtNum(t)}</text>`);
  }
  for (const t of yt) {
    const Y = sy(t);
    parts.push(`<line x1="${m.l}" y1="${Y.toFixed(1)}" x2="${m.l + iw}" y2="${Y.toFixed(1)}" stroke="${PAL.grid}"/>`);
    parts.push(`<text x="${m.l - 8}" y="${(Y + 3.5).toFixed(1)}" text-anchor="end" ` +
      `font-size="10" fill="${PAL.dim}">${yScale === "log" ? fmtLog(t) : fmtNum(t)}</text>`);
  }
  parts.push(`<rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="none" stroke="${PAL.line}"/>`);

  // series
  const colours = [PAL.accent, PAL.warm, PAL.violet, PAL.good, PAL.bad];
  series.forEach((s, i) => {
    const col = s.colour ?? colours[i % colours.length];
    const pts = [];
    for (let k = 0; k < s.x.length; k++) {
      const X = s.x[k], Y = s.y[k];
      if (!Number.isFinite(X) || !Number.isFinite(Y)) continue;
      if (xScale === "log" && X <= 0) continue;
      if (yScale === "log" && Y <= 0) continue;
      pts.push(`${sx(X).toFixed(1)},${sy(Y).toFixed(1)}`);
    }
    if (!pts.length) return;
    parts.push(`<polyline points="${pts.join(" ")}" fill="none" stroke="${col}" ` +
      `stroke-width="${s.width ?? 1.8}"${s.dashed ? ' stroke-dasharray="5 4"' : ""}/>`);
    if (s.marker) {
      for (const p of pts) {
        const [px, py] = p.split(",");
        parts.push(`<circle cx="${px}" cy="${py}" r="2.6" fill="${col}"/>`);
      }
    }
  });

  // Legend, on an opaque chip.
  //
  // The chip is not decoration: without it the labels sit directly on top of
  // whichever curve happens to reach the top-right, and which curve that is
  // changes with the data. A backing rectangle keeps the legend legible whatever
  // the plot turns out to look like, rather than requiring the placement to be
  // hand-tuned per figure and re-tuned whenever the numbers change.
  if (legend && series.some(s => s.label)) {
    const labelled = series.filter(s => s.label);
    const chipW = 12 + 26 + Math.max(...labelled.map(s => s.label.length)) * 5.6;
    const chipH = labelled.length * 15 + 8;
    const cx = m.l + iw - chipW - 6, cy = m.t + 6;
    parts.push(`<rect x="${cx.toFixed(1)}" y="${cy.toFixed(1)}" width="${chipW.toFixed(1)}" ` +
      `height="${chipH}" rx="5" fill="#0d1218" fill-opacity=".88" stroke="${PAL.line}"/>`);
    let ly = cy + 13;
    series.forEach((s, i) => {
      if (!s.label) return;
      const col = s.colour ?? colours[i % colours.length];
      parts.push(`<line x1="${(cx + 8).toFixed(1)}" y1="${ly}" x2="${(cx + 24).toFixed(1)}" y2="${ly}" ` +
        `stroke="${col}" stroke-width="2"${s.dashed ? ' stroke-dasharray="4 3"' : ""}/>`);
      parts.push(`<text x="${(cx + 30).toFixed(1)}" y="${ly + 3.5}" font-size="10" ` +
        `fill="${PAL.ink}">${esc(s.label)}</text>`);
      ly += 15;
    });
  }

  if (title) {
    parts.push(`<text x="${m.l}" y="18" font-size="11" fill="${PAL.ink}" ` +
      `font-weight="600">${esc(title)}</text>`);
  }
  parts.push(`<text x="${m.l + iw / 2}" y="${height - 6}" text-anchor="middle" ` +
    `font-size="10" fill="${PAL.dim}">${esc(xLabel)}</text>`);
  parts.push(`<text x="14" y="${m.t + ih / 2}" text-anchor="middle" font-size="10" ` +
    `fill="${PAL.dim}" transform="rotate(-90 14 ${m.t + ih / 2})">${esc(yLabel)}</text>`);

  return `<figure>
  <div class="scroll"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title || caption)}">
    ${parts.join("\n    ")}
  </svg></div>
  ${caption ? `<figcaption>${caption}</figcaption>` : ""}
</figure>`;
}

/** A reference power law y = C·x^p over [x0, x1], for overlaying on a spectrum. */
export function powerLaw(x0, x1, p, anchorX, anchorY, n = 40) {
  const C = anchorY / Math.pow(anchorX, p);
  const x = [], y = [];
  for (let i = 0; i <= n; i++) {
    const v = x0 * Math.pow(x1 / x0, i / n);
    x.push(v); y.push(C * Math.pow(v, p));
  }
  return { x, y };
}

export { PAL, esc };
