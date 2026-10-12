// The groups the docs fall into, in the order the list shows them, and where each doc lives.
import { docPath } from "./links.mjs";

export const GROUPS = {
  start: "Start here",
  findings: "Experiments and results",
  benchmark: "Reference",
  working: "Working on it",
} as const;

// Results and Progress link to their guided pages; every other doc is under docs/.
export const docHref = (slug: string) => `${import.meta.env.BASE_URL}${docPath(slug)}`;
