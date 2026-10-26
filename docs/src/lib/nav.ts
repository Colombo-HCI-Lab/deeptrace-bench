// The one site nav: every page and doc in its group and order. The sidebar, the menu on narrow
// screens, the previous and next links at the foot of each page and each page's kicker all read
// it. Ids are guided pages (PAGES) or doc slugs; "timeline" is the home page.
import { docPath } from "./links.mjs";

export type Leaf = { id: string; label: string };
export type Item = Leaf | { heading: string; items: Leaf[] };
export type Group = { label: string; items: Item[] };

const leaf = (id: string, label: string): Leaf => ({ id, label });

export const NAV: Group[] = [
  { label: "Findings", items: [leaf("timeline", "Timeline"), leaf("results", "Results")] },
  {
    label: "Method",
    items: [leaf("method", "How it works"), leaf("evaluation", "Evaluation rules"), leaf("data_policy", "Data policy")],
  },
  {
    label: "Models and datasets",
    items: [
      leaf("coverage", "Coverage"),
      { heading: "Models", items: [leaf("model-catalog", "Model catalog"), leaf("compute", "Compute cost"), leaf("models", "Porting and parity")] },
      { heading: "Datasets", items: [leaf("dataset-catalog", "Dataset catalog"), leaf("datasets", "Access and requests")] },
    ],
  },
  {
    label: "Run it yourself",
    items: [
      leaf("getting_started", "Getting started"),
      leaf("adding_a_model", "Adding a model"),
      leaf("adding_a_dataset", "Adding a dataset"),
      leaf("store_layout", "The store"),
      leaf("spark", "Where to run"),
    ],
  },
];

// Pages built by hand rather than from one doc. Results, Progress (the Timeline) and How it works
// are laid out from their docs; the full write-ups stay under docs/. The two catalogs are built
// from status.json, with a page per model and per dataset under each.
export const PAGES = ["timeline", "results", "method", "coverage", "compute", "model-catalog", "dataset-catalog"];
// The docs a guided page stands in for. catalog.md, generated for reading on GitHub, is the two
// catalogs' source text there.
export const STANDS_IN: Record<string, string> = { progress: "timeline", results: "results", method: "method", catalog: "model-catalog" };
// Docs in this group list their headings in the sidebar, always open; the rest show them in a bar
// above the text.
export const RAIL_GROUP = "Models and datasets";
// Docs whose nav label differs from their title: the page heading follows the nav.
export const RENAME: Record<string, string> = {
  evaluation: "Evaluation rules",
  models: "Porting and parity",
  datasets: "Access and requests",
};

const isLeaf = (i: Item): i is Leaf => "id" in i;
export const leaves = (items: Item[]) => items.flatMap((i) => (isLeaf(i) ? [i] : i.items));
export const ORDER: Leaf[] = NAV.flatMap((g) => leaves(g.items));
export { isLeaf };

const base = import.meta.env.BASE_URL;
export const href = (id: string) => (id === "timeline" ? base : PAGES.includes(id) ? `${base}${id}/` : `${base}${docPath(id)}`);
export const labelOf = (id: string) => ORDER.find((l) => l.id === id)?.label ?? "";
export const groupOf = (id: string) => NAV.find((g) => leaves(g.items).some((l) => l.id === (STANDS_IN[id] ?? id)))?.label ?? "";

// The pages before and after this one, in nav order.
export function pager(id: string) {
  const at = ORDER.findIndex((l) => l.id === (STANDS_IN[id] ?? id));
  const at2 = (k: number) => (at >= 0 && ORDER[at + k] ? { label: ORDER[at + k].label, href: href(ORDER[at + k].id) } : null);
  return { prev: at2(-1), next: at2(1) };
}

// Link text written as a file name ("models.md", as GitHub shows it) reads as the page it opens.
export function mdLabels(html: string) {
  return html.replace(/(<a [^>]*>)(?:<code>)?([\w-]+)\.md(?:<\/code>)?<\/a>/g, (m, open: string, slug: string) => {
    const l = labelOf(STANDS_IN[slug] ?? slug);
    return l ? `${open}${l}</a>` : m;
  });
}
