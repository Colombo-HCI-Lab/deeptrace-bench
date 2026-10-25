// Rewrites links in the docs for the site and fails the build on a broken one.
//
// The docs are written so the same Markdown also reads on GitHub:
//   models.md#parity                  another doc       -> <base>docs/models/#parity
//   results.md, progress.md           a guided page     -> <base>results/, <base> (the Timeline)
//   catalog.md#dataset-<id>           a dataset entry   -> <base>docs/datasets/#dataset-<id>
//   /configs/eval/default.yaml        a file in the repo -> its page on GitHub
//   https://colombo-hci-lab.github.io/deeptrace-bench/...  the live site -> the same page here
// Anything else relative (../x) is refused: it would break once the doc moves. Astro only logs
// an error thrown while rendering a doc, so the problems are collected and checkLinks() fails
// the build at the end.
import { existsSync, statSync } from "node:fs";
import { resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { defineMdastPlugin } from "satteri";

// From the site's root, where astro runs: the guided pages load this module from a bundle, where
// import.meta.url no longer points here.
const REPO = resolve(process.cwd(), "..") + sep;
const DOCS = resolve(process.cwd(), "src/content/docs") + sep;
const GITHUB = "https://github.com/Colombo-HCI-Lab/deeptrace-bench";
const LIVE = "https://colombo-hci-lab.github.io/deeptrace-bench/";

// Results, Progress and How it works have guided pages of their own, built from their docs;
// links to those docs land there. Progress is the Timeline, the home page. Every other doc lives
// under docs/.
export const GUIDED = { results: "results/", progress: "", method: "method/" };
export const docPath = (slug) => (slug in GUIDED ? GUIDED[slug] : `docs/${slug}/`);
// catalog.md holds the dataset entries too, but they show on the Dataset catalog page.
const catalogFrag = (frag) => /^#datasets?\b|^#dataset-/.test(frag);
const target = (slug, frag) => (slug === "catalog" && catalogFrag(frag) ? docPath("datasets") : docPath(slug)) + frag;

// Docs merged into others on 2026-10-19, and where their content went. Old links, old URLs and
// a stub page at each old address land there.
export const MOVED = {
  overview: "method/",
  smoke_test: "docs/getting_started/",
  experiments: "results/#experiments",
  cluster: "docs/spark/#curnagl",
};
// An old doc's new address, keeping the #anchor unless the move names its own.
const movedTo = (slug, frag) => MOVED[slug] + (MOVED[slug].includes("#") ? "" : frag);

// Anchors of the old single status page, and where each now lives.
export const LEGACY = {
  runs: "results/#all-runs",
  coverage: "coverage/",
  models: "docs/catalog/#all-models",
  datasets: "docs/datasets/#all-datasets",
  compute: "compute/",
  pipeline: "method/#pipeline",
  start: "method/#key-terms",
};
// The tabs of the old Models & data page (inventory/#<tab>), and where each now lives.
export const INVENTORY = { coverage: "coverage/", models: LEGACY.models, datasets: LEGACY.datasets, compute: "compute/" };

export function rewrite(url, base) {
  if (/^(https?:|mailto:|#)/.test(url) && !url.startsWith(LIVE)) return url;
  if (url.startsWith(LIVE)) {
    // The old Jekyll page catalog.html is the catalog route now, and docs moved under docs/.
    const rest = url.slice(LIVE.length).replace(/^([\w-]+)\.html/, "$1/");
    const [page, hash] = rest.split("#", 2);
    const slug = page.replace(/\/+$/, "");
    if (!slug) return base + (hash && LEGACY[hash] ? LEGACY[hash] : rest);
    const old = slug.replace(/^docs\//, "");
    if (MOVED[old]) return base + movedTo(old, hash === undefined ? "" : `#${hash}`);
    if (existsSync(`${DOCS}${slug}.md`)) return base + target(slug, hash === undefined ? "" : `#${hash}`);
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

export const checkLinks = () => ({
  name: "deeptrace-check-links",
  hooks: {
    "astro:build:done": () => {
      if (problems.length) throw new Error(`broken links:\n  ${problems.join("\n  ")}`);
    },
  },
});
