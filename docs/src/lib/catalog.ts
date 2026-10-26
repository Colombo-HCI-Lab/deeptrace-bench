// The two catalogs: an index page each (model-catalog/, dataset-catalog/) and a page per model
// and per dataset, all from status.json. The descriptions behind them live in
// src/reference/catalog.yaml.
import { D, evalsets, type Dataset, type Model, type Run } from "./status";

const base = import.meta.env.BASE_URL;

export const MODALITIES = ["audio", "video", "audio_video"] as const;
export const ROLES = ["south_asian", "reproduction", "real_reference", "in_the_wild", "pipeline_test"] as const;
export const ROLE_LABEL: Record<string, string> = {
  south_asian: "South Asian",
  reproduction: "Reproduction",
  real_reference: "Real reference",
  in_the_wild: "In the wild",
  pipeline_test: "Pipeline tests only",
};
// What each role is for, under its heading in the dataset index.
export const ROLE_ABOUT: Record<string, string> = {
  south_asian: "The target: South Asian languages, faces or both.",
  reproduction: "Not South Asian: they check our adapters against published numbers and settle score direction.",
  real_reference: "Real speech or faces only, as the real half for fake-only sets or to count false alarms.",
  in_the_wild: "Deepfakes collected where they circulated.",
  pipeline_test: "Prove the code runs; never reported.",
};
export const LEVEL: Record<string, string> = { pretrain: "pretrained on", train: "trained on", finetune: "fine-tuned on" };
// South Asian languages in the order the coverage matrix shows them: most-spoken first, then
// the two the project is after (Nepali, Sinhala), then the rest.
export const SA_LANGS = ["hi", "bn", "ur", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ne", "si", "sa", "mni", "doi", "brx"];
// The manifest features the dataset index tabulates, as reference.py names them.
export const FEATURES = ["generator", "language", "speaker or face id", "source identity of each fake", "gender", "age", "race", "sub-corpus", "official split"];

export const byName = <T extends { name: string }>(a: T, b: T) => a.name.localeCompare(b.name, "en", { sensitivity: "base" });

export const MODELS: Model[] = MODALITIES.flatMap((mod) => D.models.filter((m) => m.modality === mod).sort(byName));
export const DATASETS: Dataset[] = ROLES.flatMap((role) => D.datasets.filter((d) => d.role === role).sort(byName));

export const modelHref = (id: string) => `${base}model-catalog/${id}/`;
export const datasetHref = (id: string) => `${base}dataset-catalog/${id}/`;

export const modelRuns = (id: string): Run[] =>
  D.runs.filter((r) => r.model === id).sort((a, b) => order(a.evalset) - order(b.evalset) || a.evalset.localeCompare(b.evalset));
// Reproduction sets first, then the South Asian ones.
const order = (evalset: string) => (evalsets.get(evalset)?.role === "reproduction" ? 0 : 1);
export const southAsianRuns = (id: string) => modelRuns(id).filter((r) => evalsets.get(r.evalset)?.role === "south_asian");
export const datasetRuns = (d: Dataset): Run[] =>
  D.runs.filter((r) => d.evalsets.includes(r.evalset)).sort((a, b) => a.evalset.localeCompare(b.evalset) || b.auc - a.auc);

// What a dataset holds today: a full build, a sample, or nothing yet, and how to get the rest.
export function holding(d: Dataset): { key: "full" | "sample" | "open" | "request" | "dead"; label: string } {
  if (d.recorded?.[0] === "full") {
    const held = d.languages_held?.filter((l) => d.languages.includes(l)).length ?? d.languages.length;
    return { key: "full", label: held < d.languages.length ? `in hand: ${held} of ${d.languages.length} languages` : "in hand" };
  }
  if (d.status === "dead") return { key: "dead", label: "unavailable" };
  if (d.recorded?.[0] === "sample") return { key: "sample", label: "sample only" };
  if (d.status === "request") return { key: "request", label: "on request" };
  return { key: "open", label: d.status === "deferred" ? "deferred" : d.status === "click_through" ? "accept terms" : "open" };
}

// What we hold of one language of a dataset: a full build may cover only some of its languages.
export function holdingIn(d: Dataset, lang: string) {
  const h = holding(d);
  if (h.key !== "full" || !d.languages_held || d.languages_held.includes(lang)) return h;
  return { key: "open" as const, label: "not downloaded" };
}

// The previous and next entries of a catalog, for the foot of each entry's page.
export function siblings<T extends { id: string; name: string }>(list: T[], id: string, href: (id: string) => string) {
  const i = list.findIndex((x) => x.id === id);
  const at = (k: number) => (list[i + k] ? { label: list[i + k].name, href: href(list[i + k].id) } : null);
  return { prev: at(-1), next: at(1) };
}
