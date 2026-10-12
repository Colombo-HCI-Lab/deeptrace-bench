// Rewrites links in the docs for the site and fails the build on a broken one.
//
// The docs are written so the same Markdown also reads on GitHub:
//   models.md#parity                  another doc       -> <base>docs/models/#parity
//   results.md, progress.md           a guided page     -> <base>results/, <base>progress/
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

// Results and Progress have guided pages of their own, built from their docs; links to those
// docs land there. Every other doc lives under docs/.
export const GUIDED = new Set(["results", "progress"]);
export const docPath = (slug) => (GUIDED.has(slug) ? `${slug}/` : `docs/${slug}/`);

// Anchors of the old single status page, and where each now lives.
export const LEGACY = {
  runs: "results/#all-runs",
  coverage: "inventory/#coverage",
  models: "inventory/#models",
  datasets: "inventory/#datasets",
  compute: "inventory/#compute",
  pipeline: "method/#pipeline",
  start: "method/#terms",
};

export function rewrite(url, base) {
  if (/^(https?:|mailto:|#)/.test(url) && !url.startsWith(LIVE)) return url;
  if (url.startsWith(LIVE)) {
    // The old Jekyll page catalog.html is the catalog route now, and docs moved under docs/.
    const rest = url.slice(LIVE.length).replace(/^([\w-]+)\.html/, "$1/");
    const [page, hash] = rest.split("#", 2);
    const slug = page.replace(/\/+$/, "");
    if (!slug) return base + (hash && LEGACY[hash] ? LEGACY[hash] : rest);
    if (existsSync(`${DOCS}${slug}.md`)) return base + docPath(slug) + (hash === undefined ? "" : `#${hash}`);
    return base + rest;
  }
  const [path, hash] = url.split("#", 2);
  const frag = hash === undefined ? "" : `#${hash}`;
  if (/^[\w-]+\.md$/.test(path)) {
    if (!existsSync(DOCS + path)) throw new Error(`no doc ${path}`);
    return base + docPath(path.slice(0, -3)) + frag;
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
