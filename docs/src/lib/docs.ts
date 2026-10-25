// One doc as the site shows it: its rendered HTML cut into top-level blocks, so figures can sit
// between them (FIGS in figures.ts), and its sections. The two catalogs are composed from the
// generated catalog.md: models on the Model catalog page, the dataset entries after datasets.md
// on the Dataset catalog page.
import { getEntry } from "astro:content";
import { PAGES, RAIL_GROUP, RENAME, groupOf, mdLabels } from "./nav";

export interface Block {
  html: string;
  // A heading's id, so a figure can be placed after it.
  id: string;
}
export interface Section {
  href: string;
  text: string;
}

const VOID = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"]);

// The top-level elements of well-formed HTML, as rendered from Markdown.
export function blocksOf(html: string): Block[] {
  const out: Block[] = [];
  const tag = /<!--[\s\S]*?-->|<(\/?)([a-zA-Z][\w:-]*)\b[^>]*?(\/?)>/g;
  let depth = 0;
  let start = -1;
  let m;
  while ((m = tag.exec(html))) {
    if (!m[2]) continue;
    const name = m[2].toLowerCase();
    const closing = m[1] === "/";
    const selfClosing = m[3] === "/" || VOID.has(name);
    if (!closing && depth === 0) start = m.index;
    if (closing) depth--;
    else if (!selfClosing) depth++;
    if (depth === 0 && start >= 0) {
      const h = html.slice(start, m.index + m[0].length);
      const id = /^<h[1-6]\b/i.test(h) ? (h.match(/^<h[1-6][^>]*\bid="([^"]+)"/i)?.[1] ?? "") : "";
      out.push({ html: h, id });
      start = -1;
    }
  }
  return out;
}

const text = (html: string) =>
  html
    .replace(/<[^>]+>/g, "")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .trim();
// Headings one level up: the catalog's entries sit two levels deep in catalog.md.
const promote = (b: Block): Block => ({ ...b, html: b.html.replace(/<(\/?)h([3-6])\b/g, (_, s, n) => `<${s}h${Number(n) - 1}`) });
const anchor = (id: string): Block => ({ html: `<span id="${id}"></span>`, id: "" });

async function rendered(slug: string) {
  const entry = await getEntry("docs", slug);
  if (!entry?.rendered) throw new Error(`no rendered doc ${slug}`);
  return { entry, blocks: blocksOf(mdLabels(entry.rendered.html)) };
}

// catalog.md's two halves: the models (without their "## Models" heading) and the datasets.
async function catalogParts() {
  const { blocks } = await rendered("catalog");
  const at = (id: string) => blocks.findIndex((b) => b.id === id);
  const m = at("models");
  const d = at("datasets");
  // The intro paragraph; the list after it points at the two halves, which are pages now.
  const intro = blocks.slice(0, m).filter((b) => !/^<ul\b/.test(b.html));
  return {
    intro,
    models: blocks.slice(m + 1, d).map(promote),
    datasets: blocks.slice(d + 1).map(promote),
  };
}

export async function docView(slug: string) {
  const { entry, blocks: own } = await rendered(slug);
  let blocks = own;
  if (slug === "catalog") {
    const c = await catalogParts();
    blocks = [...c.intro, anchor("models"), ...c.models];
  } else if (slug === "datasets") {
    const c = await catalogParts();
    blocks = [...own, anchor("datasets"), ...c.datasets];
  }
  const sections: Section[] = blocks.filter((b) => /^<h2\b/.test(b.html) && b.id).map((b) => ({ href: `#${b.id}`, text: text(b.html) }));
  return {
    slug,
    title: RENAME[slug] ?? entry.data.title,
    // catalog.md describes both halves; its page shows the models.
    description: slug === "catalog" ? "Every detector: what it trained on, the authors' numbers beside ours, size and cost" : entry.data.description,
    group: groupOf(slug),
    blocks,
    sections,
  };
}

// The summary tables that open the two catalogs; the sidebar lists them first.
export const TABLES: Record<string, Section> = {
  catalog: { href: "#all-models", text: "All models" },
  datasets: { href: "#all-datasets", text: "All datasets" },
};

// What the sidebar lists under a doc in the Models and datasets group, always open.
export async function railSections(slug: string): Promise<Section[]> {
  if (groupOf(slug) !== RAIL_GROUP || PAGES.includes(slug)) return [];
  const v = await docView(slug);
  return [...(TABLES[slug] ? [TABLES[slug]] : []), ...v.sections];
}
