// Rewrites links in the docs for the site and fails the build on a broken one.
//
// The docs are written so the same Markdown also reads on GitHub:
//   models.md#parity                  another doc       -> <base>models/#parity
//   results.md, progress.md           a guided page     -> <base>results/, <base> (the Timeline)
//   catalog.md#model-<id>             a model's page    -> <base>model-catalog/<id>/
//   catalog.md#dataset-<id>           a dataset's page  -> <base>dataset-catalog/<id>/
//   /configs/eval/default.yaml        a file in the repo -> its page on GitHub
//   https://colombo-hci-lab.github.io/deeptrace-bench/...  the live site -> the same page here
// Anything else relative (../x) is refused: it would break once the doc moves. Astro only logs
// an error thrown while rendering a doc, so the problems are collected and checkLinks() fails
// the build at the end.
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { defineMdastPlugin } from "satteri";

// From the site's root, where astro runs: the guided pages load this module from a bundle, where
// import.meta.url no longer points here.
const REPO = resolve(process.cwd(), "..") + sep;
const DOCS = resolve(process.cwd(), "src/content/docs") + sep;
const GITHUB = "https://github.com/Colombo-HCI-Lab/deeptrace-bench";
const LIVE = "https://colombo-hci-lab.github.io/deeptrace-bench/";

// Every page has one address. Results, Progress and How it works are guided pages built from
// their docs, so links to those docs land there; Progress is the Timeline, the home page. Every
// other doc is a page at the site root. The docs/<slug>/ addresses they had until 2026-10-29
// forward to these (pages/docs/[slug].astro).
export const GUIDED = { results: "results/", progress: "", method: "method/" };
export const docPath = (slug) => (slug in GUIDED ? GUIDED[slug] : `${slug}/`);
// catalog.md is generated for reading on GitHub; on the site its two halves are the Model and
// Dataset catalogs, with a page per entry.
function catalogTarget(frag) {
  const entry = frag.match(/^#(model|dataset)-(.+)$/);
  if (entry) return `${entry[1]}-catalog/${entry[2]}/`;
  return /^#(datasets?|all-datasets)\b/.test(frag) ? "dataset-catalog/" : "model-catalog/";
}
const target = (slug, frag) => (slug === "catalog" ? catalogTarget(frag) : docPath(slug) + frag);

// Docs merged into others on 2026-10-19, and where their content went. Old links, old URLs and
// a stub page at each old address land there.
export const MOVED = {
  overview: "method/",
  smoke_test: "getting_started/",
  experiments: "results/#experiments",
  cluster: "spark/#curnagl",
};
// An old doc's new address, keeping the #anchor unless the move names its own.
const movedTo = (slug, frag) => MOVED[slug] + (MOVED[slug].includes("#") ? "" : frag);

// Anchors of the old single status page, and where each now lives.
export const LEGACY = {
  runs: "results/#all-runs",
  coverage: "coverage/",
  models: "model-catalog/",
  datasets: "dataset-catalog/",
  compute: "compute/",
  pipeline: "method/#the-pipeline",
  start: "method/#key-terms",
};
// The tabs of the old Models & data page (inventory/#<tab>), and where each now lives.
export const INVENTORY = { coverage: "coverage/", models: LEGACY.models, datasets: LEGACY.datasets, compute: "compute/" };

export function rewrite(url, base) {
  if (/^(https?:|mailto:|#)/.test(url) && !url.startsWith(LIVE)) return url;
  if (url.startsWith(LIVE)) {
    // The old Jekyll page catalog.html is the catalog route now, and docs/<slug>/ is the doc's
    // old address.
    const rest = url.slice(LIVE.length).replace(/^([\w-]+)\.html/, "$1/");
    const [page, hash] = rest.split("#", 2);
    const slug = page.replace(/\/+$/, "");
    if (!slug) return base + (hash && LEGACY[hash] ? LEGACY[hash] : rest);
    const old = slug.replace(/^docs\//, "");
    const frag = hash === undefined ? "" : `#${hash}`;
    if (MOVED[old]) return base + movedTo(old, frag);
    if (existsSync(`${DOCS}${old}.md`)) return base + target(old, frag);
    return base + rest;
  }
  const [path, hash] = url.split("#", 2);
  const frag = hash === undefined ? "" : `#${hash}`;
  if (/^[\w-]+\.md$/.test(path)) {
    if (MOVED[path.slice(0, -3)]) throw new Error(`${path} moved to ${MOVED[path.slice(0, -3)]}; link the new doc`);
    if (!existsSync(DOCS + path)) throw new Error(`no doc ${path}`);
    return base + target(path.slice(0, -3), frag);
  }
  if (path.startsWith("/")) {
    const target = REPO + path.slice(1);
    if (!existsSync(target)) throw new Error(`no file ${path} in the repo`);
    const kind = statSync(target).isDirectory() ? "tree" : "blob";
    return `${GITHUB}/${kind}/main${path}${frag}`;
  }
  throw new Error(`link ${url}: write docs as name.md and repo files as /path`);
}

// Shared through globalThis: the config and the bundled pages each load their own copy of this
// module, and checkLinks must see the problems of both.
const problems = (globalThis.__deeptraceLinkProblems ??= []);

export const links = ({ base }) =>
  defineMdastPlugin({
    name: "deeptrace-links",
    link(node, ctx) {
      try {
        ctx.setProperty(node, "url", rewrite(node.url, base));
      } catch (err) {
        const where = ctx.fileURL ? fileURLToPath(ctx.fileURL).replace(REPO, "") : "a doc";
        problems.push(`${where}: ${err.message}`);
        console.error(`broken link in ${where}: ${err.message}`);
      }
    },
  });

// Every page has one address, so after the build a link on a page must open a page, not a
// forwarding stub, and its #anchor must exist there.
function checkBuilt(out, base) {
  const pages = new Map();
  const walk = (dir) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name.endsWith(".html")) pages.set(base + relative(out, p).split(sep).join("/").replace(/(^|\/)index\.html$/, "$1"), readFileSync(p, "utf8"));
    }
  };
  walk(out);
  const stub = (html) => html.includes('http-equiv="refresh"');
  const ids = (html) => new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
  for (const [page, html] of pages) {
    if (stub(html)) continue;
    for (const [, raw] of html.matchAll(/\shref="([^"]+)"/g)) {
      const href = raw.replaceAll("&amp;", "&");
      const local = href.startsWith("#") ? page + href : href.startsWith(base) ? href : null;
      if (!local || /\/_astro\/|\.\w+(#|$)/.test(local)) continue;
      const [path, frag] = local.split("#", 2);
      const target = pages.get(path);
      if (target === undefined) problems.push(`${page}: ${href} is no page`);
      else if (stub(target)) problems.push(`${page}: ${href} only forwards; link where it goes`);
      else if (frag && !ids(target).has(decodeURIComponent(frag))) problems.push(`${page}: ${href} has no #${frag}`);
    }
  }
}

export const checkLinks = ({ base }) => ({
  name: "deeptrace-check-links",
  hooks: {
    "astro:build:done": ({ dir }) => {
      checkBuilt(fileURLToPath(dir), base);
      if (problems.length) throw new Error(`broken links:\n  ${[...new Set(problems)].join("\n  ")}`);
    },
  },
});
