/**
 * Receives results measured in the browser and writes them to results/*.json.
 *
 * WHY THIS EXISTS. The GPU results -- kernel verification, the CPU/GPU
 * benchmark, the turbulence spectrum -- can only be produced where WebGPU
 * exists, which is a browser. But the report must be built from committed JSON
 * so that every published number traces to a run rather than to a screenshot of
 * one. This is the bridge: the harness pages POST their `window.__RESULTS__`
 * here, and it lands on disk.
 *
 * Run alongside tools/serve.js:  node tools/collect.js [port]
 */

import { createServer } from "node:http";
import { writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "results");
const PORT = Number(process.argv[2] || 8932);

mkdirSync(OUT, { recursive: true });

createServer((req, res) => {
  // The harness pages are served from a different port, so this is a
  // cross-origin POST. Permissive CORS is correct here: it listens on loopback
  // only and exists for the duration of a measurement run.
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "content-type");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  if (req.method === "OPTIONS") { res.writeHead(204).end(); return; }
  if (req.method !== "POST") { res.writeHead(405).end("post only"); return; }

  const name = decodeURIComponent(new URL(req.url, "http://x").pathname.slice(1))
    .replace(/[^a-z0-9._-]/gi, "");
  if (!name) { res.writeHead(400).end("name required"); return; }

  const chunks = [];
  req.on("data", c => chunks.push(c));
  req.on("end", () => {
    const body = Buffer.concat(chunks).toString("utf8");
    try {
      JSON.parse(body);                       // reject anything unparseable
      const path = join(OUT, `${name}.json`);
      writeFileSync(path, body);
      console.log(`wrote results/${name}.json  (${(body.length / 1024).toFixed(1)} KB)`);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true, bytes: body.length }));
    } catch (e) {
      res.writeHead(400).end(String(e));
    }
  });
}).listen(PORT, "127.0.0.1", () => {
  console.log(`collector listening on http://127.0.0.1:${PORT}/<name>`);
});
