/**
 * Radix-2 Cooley-Tukey FFT and the 2D energy spectrum built on it.
 *
 * Written from scratch because there is no scipy here and no runtime
 * dependencies in this project. That makes correctness this file's own
 * problem, so it is checked three ways in the tests:
 *
 *   1. against a naive O(N²) DFT, which is transcribed straight from the
 *      definition and is slow but obviously right
 *   2. round-trip: ifft(fft(x)) == x
 *   3. Parseval's theorem, which the spectrum code below then re-uses as a
 *      permanent self-check -- Σ E(k) must equal the kinetic energy computed
 *      independently in physical space
 *
 * Check 3 is the valuable one. Nearly every way of getting a spectrum wrong is
 * a normalisation error, and a normalisation error shows up as Σ E(k) missing
 * the physical-space energy by a factor of N, N², or 2. A spectrum that
 * satisfies Parseval has the right units, and a spectrum with the right units
 * is one whose SLOPE can be trusted -- which is the only thing the Kraichnan
 * comparison actually reads off it.
 */

/** True if n is a power of two and at least 1. */
export function isPow2(n) {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

/**
 * In-place complex FFT of length n (power of two).
 * `re` and `im` are Float64Arrays of length n.
 * @param {number} sign -1 for forward, +1 for inverse (unnormalised)
 */
export function fftInPlace(re, im, sign = -1) {
  const n = re.length;
  if (!isPow2(n)) throw new Error(`fft length must be a power of two, got ${n}`);
  if (im.length !== n) throw new Error("re/im length mismatch");

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }

  // Butterflies.
  for (let len = 2; len <= n; len <<= 1) {
    const ang = sign * 2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k], ai = im[i + k];
        const br = re[i + k + len / 2], bi = im[i + k + len / 2];
        const dr = br * cr - bi * ci;
        const di = br * ci + bi * cr;
        re[i + k] = ar + dr;      im[i + k] = ai + di;
        re[i + k + len / 2] = ar - dr;
        im[i + k + len / 2] = ai - di;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/** Forward FFT, returns new arrays. */
export function fft(reIn, imIn) {
  const re = Float64Array.from(reIn);
  const im = imIn ? Float64Array.from(imIn) : new Float64Array(re.length);
  fftInPlace(re, im, -1);
  return [re, im];
}

/** Inverse FFT with 1/n normalisation, returns new arrays. */
export function ifft(reIn, imIn) {
  const re = Float64Array.from(reIn);
  const im = Float64Array.from(imIn);
  fftInPlace(re, im, +1);
  const n = re.length;
  for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  return [re, im];
}

/** Naive DFT straight from the definition. Reference for testing only. */
export function dftNaive(reIn, imIn, sign = -1) {
  const n = reIn.length;
  const im0 = imIn || new Float64Array(n);
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let sr = 0, si = 0;
    for (let t = 0; t < n; t++) {
      const a = sign * 2 * Math.PI * k * t / n;
      const c = Math.cos(a), s = Math.sin(a);
      sr += re0(t) * c - im0[t] * s;
      si += re0(t) * s + im0[t] * c;
    }
    re[k] = sr; im[k] = si;
  }
  function re0(t) { return reIn[t]; }
  return [re, im];
}

/**
 * 2D forward FFT of a real field stored row-major (nx fastest).
 * Returns [re, im] of length nx*ny. Both dimensions must be powers of two.
 */
export function fft2(field, nx, ny) {
  if (!isPow2(nx) || !isPow2(ny)) {
    throw new Error(`fft2 needs power-of-two dimensions, got ${nx}x${ny}`);
  }
  const re = Float64Array.from(field);
  const im = new Float64Array(nx * ny);

  const rowR = new Float64Array(nx), rowI = new Float64Array(nx);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) { rowR[i] = re[j * nx + i]; rowI[i] = im[j * nx + i]; }
    fftInPlace(rowR, rowI, -1);
    for (let i = 0; i < nx; i++) { re[j * nx + i] = rowR[i]; im[j * nx + i] = rowI[i]; }
  }

  const colR = new Float64Array(ny), colI = new Float64Array(ny);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) { colR[j] = re[j * nx + i]; colI[j] = im[j * nx + i]; }
    fftInPlace(colR, colI, -1);
    for (let j = 0; j < ny; j++) { re[j * nx + i] = colR[j]; im[j * nx + i] = colI[j]; }
  }
  return [re, im];
}

/**
 * 2D inverse FFT. Takes [re, im] and returns [re, im] with 1/(nx·ny) applied.
 */
export function ifft2(re0, im0, nx, ny) {
  const re = Float64Array.from(re0);
  const im = Float64Array.from(im0);

  const rowR = new Float64Array(nx), rowI = new Float64Array(nx);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) { rowR[i] = re[j * nx + i]; rowI[i] = im[j * nx + i]; }
    fftInPlace(rowR, rowI, +1);
    for (let i = 0; i < nx; i++) { re[j * nx + i] = rowR[i]; im[j * nx + i] = rowI[i]; }
  }
  const colR = new Float64Array(ny), colI = new Float64Array(ny);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) { colR[j] = re[j * nx + i]; colI[j] = im[j * nx + i]; }
    fftInPlace(colR, colI, +1);
    for (let j = 0; j < ny; j++) { re[j * nx + i] = colR[j]; im[j * nx + i] = colI[j]; }
  }
  const N = nx * ny;
  for (let i = 0; i < N; i++) { re[i] /= N; im[i] /= N; }
  return [re, im];
}

/**
 * Signed wavenumber for FFT bin index i of length n.
 * Bins above n/2 represent negative frequencies; treating them as large
 * positive wavenumbers would fold the whole upper half of the spectrum onto
 * wavenumbers that do not exist and produce a spurious high-k tail.
 */
export function freqIndex(i, n) {
  return i <= n / 2 ? i : i - n;
}

// ============================================================ energy spectrum

/**
 * Shell-averaged kinetic energy spectrum E(k) of a 2D velocity field.
 *
 * The velocity must be COLLOCATED (both components at cell centres) -- pass MAC
 * fields through `velocityAtCenters` first. Mixing staggered samples into one
 * transform would introduce a half-cell phase shift between the components,
 * which is invisible in E(k) magnitude but corrupts anything phase-sensitive
 * built on it later.
 *
 * DEFINITION AND NORMALISATION. With û the discrete transform normalised by
 * 1/N (N = nx·ny), the energy in mode k is ½(|û|² + |v̂|²). Summing over all
 * modes gives ½⟨u²+v²⟩, i.e. the physical-space kinetic energy -- that is
 * Parseval, and it is what `parsevalCheck` verifies.
 *
 * E(k) then bins those modes into integer shells k ≤ |k| < k+1. Two properties
 * follow, both of which matter for reading the slope:
 *
 *   - Σ_k E(k) = total KE exactly (no energy is lost or double counted)
 *   - E(k) is a DENSITY over shells of unit width, so it is directly the
 *     quantity that Kraichnan's k^(-5/3) and k^(-3) predictions describe
 *
 * A note on what is NOT done here: no windowing. The domain is periodic, so
 * there is no spectral leakage to window against, and applying one anyway
 * would tilt the spectrum and change the fitted slope.
 */
export function energySpectrum(cu, cv, nx, ny) {
  const N = nx * ny;
  const [ur, ui] = fft2(cu, nx, ny);
  const [vr, vi] = fft2(cv, nx, ny);

  const kmax = Math.floor(Math.min(nx, ny) / 2);
  const E = new Float64Array(kmax + 1);
  const counts = new Int32Array(kmax + 1);

  for (let j = 0; j < ny; j++) {
    const ky = freqIndex(j, ny);
    for (let i = 0; i < nx; i++) {
      const kx = freqIndex(i, nx);
      const idx = j * nx + i;

      // 1/N normalisation applied to the amplitude, so |û|² has units of u².
      const pu = (ur[idx] * ur[idx] + ui[idx] * ui[idx]) / (N * N);
      const pv = (vr[idx] * vr[idx] + vi[idx] * vi[idx]) / (N * N);
      const e = 0.5 * (pu + pv);

      const kmag = Math.sqrt(kx * kx + ky * ky);
      const shell = Math.round(kmag);
      if (shell <= kmax) {
        E[shell] += e;
        counts[shell]++;
      }
      // Modes beyond kmax (the corners of the square k-grid, |k| up to
      // kmax·√2) are DROPPED, which is why Σ E(k) can fall slightly short of
      // the physical-space energy. `parsevalCheck` reports the shortfall
      // rather than hiding it: those corner modes are anisotropically sampled
      // and including them would bend the tail of an otherwise isotropic
      // spectrum.
    }
  }

  const k = new Float64Array(kmax + 1);
  for (let i = 0; i <= kmax; i++) k[i] = i;
  return { k, E, counts, kmax };
}

/**
 * Verify Parseval: Σ E(k) against the physical-space kinetic energy.
 * Returns both and their relative difference.
 */
export function parsevalCheck(cu, cv, nx, ny) {
  const { E } = energySpectrum(cu, cv, nx, ny);
  let spectral = 0;
  for (let i = 0; i < E.length; i++) spectral += E[i];

  let physical = 0;
  for (let i = 0; i < cu.length; i++) physical += cu[i] * cu[i] + cv[i] * cv[i];
  physical = 0.5 * physical / cu.length;

  return {
    spectral,
    physical,
    relDiff: physical > 0 ? Math.abs(spectral - physical) / physical : 0,
  };
}

/**
 * Enstrophy spectrum, Z(k) = k²·E(k).
 *
 * In 2D this is the quantity whose cascade is forward, and the -3 slope of E(k)
 * in the enstrophy range is equivalent to a flat (k^-1) Z(k). Reporting both
 * makes it much easier to see which cascade a given range of scales is in.
 */
export function enstrophySpectrum(cu, cv, nx, ny) {
  const s = energySpectrum(cu, cv, nx, ny);
  const Z = new Float64Array(s.E.length);
  for (let i = 0; i < s.E.length; i++) Z[i] = s.k[i] * s.k[i] * s.E[i];
  return { ...s, Z };
}

// ============================================================== slope fitting

/**
 * Least-squares slope of log E vs log k over [kLo, kHi] inclusive.
 *
 * The fit window is a PARAMETER and is reported alongside every slope this
 * produces. Choosing the window after seeing the data is the standard way to
 * manufacture agreement with a power law: almost any curve contains some
 * sub-range that fits -5/3 if you are allowed to hunt for it. The windows used
 * in this project are fixed from physical reasoning (relative to the forcing
 * wavenumber and the dissipation scale) before the spectrum is looked at, and
 * both the window and the full curve are shown.
 *
 * Also returns r², so a slope quoted from a range that is not actually a power
 * law can be recognised as such rather than reported as a clean match.
 */
export function fitSlope(k, E, kLo, kHi) {
  const xs = [], ys = [];
  for (let i = 0; i < k.length; i++) {
    if (k[i] >= kLo && k[i] <= kHi && E[i] > 0 && k[i] > 0) {
      xs.push(Math.log(k[i]));
      ys.push(Math.log(E[i]));
    }
  }
  const n = xs.length;
  if (n < 3) {
    return { slope: NaN, intercept: NaN, r2: NaN, n, kLo, kHi };
  }
  let sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; }
  const mx = sx / n, my = sy / n;

  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  const slope = sxy / sxx;
  const r2 = syy > 0 ? (sxy * sxy) / (sxx * syy) : NaN;
  return { slope, intercept: my - slope * mx, r2, n, kLo, kHi };
}

/**
 * Observed order of accuracy from an error-vs-resolution study.
 *
 * Fits log(error) against log(h); the slope is the convergence order. Feeding
 * it errors that have hit a floor (round-off, or an exact-solution error)
 * produces a meaninglessly low order, so the caller is expected to check that
 * the errors are still decreasing -- `convergenceOrder` reports the per-pair
 * orders too, which makes a flattening tail obvious.
 */
export function convergenceOrder(hs, errors) {
  if (hs.length !== errors.length || hs.length < 2) {
    throw new Error("need at least two (h, error) pairs");
  }
  const k = Float64Array.from(hs);
  const E = Float64Array.from(errors);
  const fit = fitSlope(k, E, Math.min(...hs) * 0.999, Math.max(...hs) * 1.001);

  // Pairwise orders: log(e1/e2)/log(h1/h2). A clean study has these all equal;
  // a tail that flattens is how a floor announces itself.
  const pairwise = [];
  for (let i = 0; i + 1 < hs.length; i++) {
    pairwise.push(Math.log(errors[i] / errors[i + 1]) / Math.log(hs[i] / hs[i + 1]));
  }
  return { order: fit.slope, r2: fit.r2, pairwise };
}
