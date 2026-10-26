// One doc as the site shows it: its rendered HTML cut into top-level blocks, so figures can sit
// between them (FIGS in figures.ts), and its sections.
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
async function rendered(slug: string) {
  const entry = await getEntry("docs", slug);
  if (!entry?.rendered) throw new Error(`no rendered doc ${slug}`);
  return { entry, blocks: blocksOf(mdLabels(entry.rendered.html)) };
}

export async function docView(slug: string) {
  const { entry, blocks } = await rendered(slug);
  const sections: Section[] = blocks.filter((b) => /^<h2\b/.test(b.html) && b.id).map((b) => ({ href: `#${b.id}`, text: text(b.html) }));
  return {
    slug,
    title: RENAME[slug] ?? entry.data.title,
    description: entry.data.description,
    group: groupOf(slug),
    blocks,
    sections,
  };
}

// What the sidebar lists under a doc in the Models and datasets group, always open.
export async function railSections(slug: string): Promise<Section[]> {
  if (groupOf(slug) !== RAIL_GROUP || PAGES.includes(slug)) return [];
  const v = await docView(slug);
  return v.sections;
}
