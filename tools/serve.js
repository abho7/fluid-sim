/**
 * A minimal static server for local development.
 *
 * WebGPU requires a secure context, which for local work means http://localhost
 * (browsers treat loopback as secure). Opening the pages as file:// URLs will
 * not work: ES module imports are blocked by CORS on file://, and
 * navigator.gpu is unavailable there.
 *
 * Written rather than pulled in because the project has no runtime
 * dependencies, and a dozen lines of http.createServer does the job.
 *
 *   node tools/serve.js [port]
 */

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { join, extname, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(join(fileURLToPath(import.meta.url), "..", ".."));
const PORT = Number(process.argv[2] || 8931);

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wgsl": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

createServer(async (req, res) => {
  try {
    let path = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (path.endsWith("/")) path += "index.html";

    // Contain the served tree. `normalize` collapses ".." before the prefix
    // check, so a request for /../../etc/passwd resolves outside ROOT and is
    // rejected rather than escaping.
    const full = resolve(join(ROOT, normalize(path)));
    if (!full.startsWith(ROOT)) {
      res.writeHead(403).end("forbidden");
      return;
    }

    const s = await stat(full).catch(() => null);
    if (!s || !s.isFile()) {
      res.writeHead(404).end("not found");
      return;
    }

    const body = await readFile(full);
    res.writeHead(200, {
      "Content-Type": TYPES[extname(full).toLowerCase()] ?? "application/octet-stream",
      "Content-Length": body.length,
      // No caching: the whole point of this server is iterating on the files.
      "Cache-Control": "no-store",
    });
    res.end(body);
  } catch (e) {
    res.writeHead(500).end(String(e));
  }
}).listen(PORT, "127.0.0.1", () => {
  console.log(`serving ${ROOT} at http://127.0.0.1:${PORT}/`);
});
