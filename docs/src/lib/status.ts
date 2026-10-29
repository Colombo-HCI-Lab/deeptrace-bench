// The status data build_site.py writes (src/reference/status.json, from status.collect), its
// types, and the lookups and formatting every status table shares.
import raw from "../reference/status.json";

export type Modality = "audio" | "video" | "audio_video" | "image";

export interface Paper {
  title: string | null;
  year: number | null;
  citations: number | null;
  venue: string | null;
  url: string;
}

export interface Profile {
  params: number;
  gflops_per_unit: number | null;
  ms_per_unit: number;
  ms_per_item: number;
  units_per_item: number;
  unit: string;
  peak_memory_gb: number | null;
  gpu: string;
  precision: string;
}

// A headline number the authors report for the released weights; evalset when we score that set too.
export interface Published {
  test: string;
  metric: string;
  value: string;
  source: string;
  cite: string;
  evalset?: string;
}

export interface Model {
  id: string;
  name: string;
  modality: Modality;
  adapter: boolean;
  blocked: boolean;
  summary: string | null;
  why: string | null;
  cut: boolean;
  // Kinds of fake in its training data (FakeType ids) and what it looks at.
  catches: string[];
  cue: string | null;
  paper: Paper | null;
  published: Published[];
  trained_on: string[];
  parity: { evalset: string; items: number; max_abs_diff: number } | null;
  profile: Profile | null;
  licence: string;
  architecture: string | null;
  description: string | null;
  input: string | null;
  training: { name: string; level: "pretrain" | "train" | "finetune"; split: string | null }[];
}

export interface Dataset {
  id: string;
  name: string;
  modality: Modality;
  role: string;
  status: string;
  languages: string[];
  summary: string | null;
  why: string | null;
  cut: boolean;
  licence: string;
  contains: string[];
  // Kinds of fake it holds (FakeType ids); [] for real only, null when untagged.
  fakes: string[] | null;
  generators: string[] | null;
  annotations: string[] | null;
  format: string | null;
  notes: string | null;
  // Features the manifest fills in, from the full build or a sample; null before any build.
  recorded: ["full" | "sample", string[]] | null;
  // Languages the full build holds, when it records them: a partial download shows here.
  languages_held: string[] | null;
  scale: string | null;
  paper: Paper | null;
  evalsets: string[];
}

export interface Evalset {
  id: string;
  modality: Modality;
  role: string;
  pairing: "same_corpus" | "cross_corpus";
  status: string;
  size: { items: number; windows?: number } | null;
}

export interface Run {
  model: string;
  evalset: string;
  run_id: string;
  n: number;
  n_real: number;
  n_fake: number;
  n_failed: number;
  auc: number;
  auc_lo: number;
  auc_hi: number;
  eer: number;
  tpr_at_fpr05: number | null;
  reasons: string[];
  flags: string[];
  wall_s: number | null;
  // Measured while scoring (telemetry, since 2026-10-27); null for runs before it.
  compute: RunCompute | null;
}

export interface RunCompute {
  items_per_s: number | null;
  windows_per_s: number | null;
  realtime_factor: number | null;
  cuda_max_allocated_gb: number | null;
  gpu_util_mean: number | null;
  energy_wh: number | null;
  energy_wh_per_1k_items: number | null;
  gpu: string | null;
}

// A kind of fake, defined once in catalog.yaml: what it changes and what it leaves behind.
export interface FakeType {
  id: string;
  label: string;
  modality: "video" | "audio" | "any";
  changed: string;
  trace: string;
}

export interface Status {
  built: string;
  eval_status: string;
  citations_checked: string | null;
  fake_types: FakeType[];
  models: Model[];
  datasets: Dataset[];
  evalsets: Evalset[];
  cells: Record<string, "na" | "guard">;
  runs: Run[];
}

export const D = raw as unknown as Status;

export const runs = new Map(D.runs.map((r) => [`${r.model}|${r.evalset}`, r]));
export const models = new Map(D.models.map((m) => [m.id, m]));
export const evalsets = new Map(D.evalsets.map((e) => [e.id, e]));
export const nRuns = (m: Model) => D.runs.filter((r) => r.model === m.id).length;

export const GROUPS: [string, string, string][] = [
  ["reproduction", "same_corpus", "Reproduction"],
  ["south_asian", "same_corpus", "South Asian · same corpus"],
  ["south_asian", "cross_corpus", "South Asian · cross corpus"],
];

// AUC in bands of 0.1 for the coverage grid, each its own hue so neighbouring bands read apart at
// a glance: red below chance, grey near it, then yellow, green, teal and indigo. The round's
// dashboard uses the same bands.
export const AUC_BANDS = [
  { min: 0.9, label: "0.9 and up", bg: "#3b4cc0", fg: "#ffffff" },
  { min: 0.8, label: "0.8 to 0.9", bg: "#1f8a8a", fg: "#ffffff" },
  { min: 0.7, label: "0.7 to 0.8", bg: "#5cc863", fg: "#10241a" },
  { min: 0.6, label: "0.6 to 0.7", bg: "#f2d544", fg: "#2a2306" },
  { min: 0.5, label: "0.5 to 0.6, near chance", bg: "#c9c9c9", fg: "#222222" },
  { min: -Infinity, label: "below 0.5", bg: "#d6453d", fg: "#ffffff" },
];
export const aucBand = (auc: number) => AUC_BANDS.find((b) => auc >= b.min)!;

// The coverage grid's columns and row groups: the models of one modality, and the evalsets of the
// kinds they read, without the quick subsets (stratified samples of sets shown in full).
export const gridModels = (mod: Modality) =>
  D.models
    .filter((m) => m.modality === mod)
    .sort((a, b) => nRuns(b) - nRuns(a) || Number(!!b.parity) - Number(!!a.parity) || a.name.localeCompare(b.name));
export const gridGroups = (mods: Modality[]) =>
  GROUPS.map(([role, pairing, label]) => ({
    label,
    evs: D.evalsets
      .filter((e) => mods.includes(e.modality) && !e.id.endsWith("_quick") && e.role === role && e.pairing === pairing)
      .sort((a, b) => Number(a.status !== "ready") - Number(b.status !== "ready") || a.id.localeCompare(b.id)),
  })).filter((g) => g.evs.length);

// What one grid cell holds: a run, a guard refusal, nothing to run (na), a pair not scored yet,
// one waiting for data access, or one whose model has no adapter.
export type CellState = "run" | "guard" | "na" | "empty" | "pending" | "off";
export function cellState(m: Model, e: Evalset): CellState {
  const key = `${m.id}|${e.id}`;
  const state = D.cells[key];
  if (state === "na" || state === "guard") return state;
  if (runs.has(key)) return "run";
  if (!m.adapter) return "off";
  return e.status === "ready" ? "empty" : "pending";
}

// One audio window is AASIST's 64,600 samples at 16 kHz, about 4 s.
export const WINDOW_S = 64600 / 16000;

// Model-only time for a full evalset: audio from its 4 s windows, video from its clips.
export function estimate(m: Model, e: Evalset): number | null {
  const p = m.profile;
  const size = e.size;
  if (!p || !size) return null;
  if (m.modality === "audio") return size.windows ? (size.windows * p.ms_per_unit) / 1000 : null;
  return (size.items * p.ms_per_item) / 1000;
}

export const fmt = (x: number, d = 3) => x.toFixed(d);
export const pct = (x: number) => (100 * x).toFixed(1) + "%";
export const int = (x: number) => Math.round(x).toLocaleString("en-US");
export const params = (n: number) =>
  n >= 1e9
    ? (n / 1e9).toFixed(2) + " B"
    : n >= 1e7
      ? (n / 1e6).toFixed(0) + " M"
      : n >= 1e6
        ? (n / 1e6).toFixed(1) + " M"
        : (n / 1e3).toFixed(0) + " K";
export const dur = (s: number) =>
  s < 90
    ? Math.max(1, Math.round(s)) + " s"
    : s < 5400
      ? Math.round(s / 60) + " min"
      : s < 172800
        ? (s / 3600).toFixed(1) + " h"
        : (s / 86400).toFixed(1) + " days";
export const ms = (x: number) => (x < 10 ? x.toFixed(1) + " ms" : Math.round(x) + " ms");

export const LANG: Record<string, string> = {
  hi: "Hindi", en: "English", bn: "Bengali", ta: "Tamil", mr: "Marathi", ml: "Malayalam",
  te: "Telugu", ur: "Urdu", or: "Odia", kn: "Kannada", gu: "Gujarati", sa: "Sanskrit",
  si: "Sinhala", pa: "Punjabi", ne: "Nepali", as: "Assamese", mni: "Manipuri", doi: "Dogri",
  brx: "Bodo",
};
export const ACCESS: Record<string, string> = {
  open: "Open download", click_through: "Accept terms online", request: "On request",
  deferred: "Deferred", dead: "Unavailable",
};
export const ROLE: Record<string, string> = {
  south_asian: "South Asian", reproduction: "Reproduction", real_reference: "Real reference",
  in_the_wild: "In the wild",
};
export const MOD: Record<string, string> = {
  audio: "Audio", video: "Video", audio_video: "Audio-visual", image: "Image",
};
