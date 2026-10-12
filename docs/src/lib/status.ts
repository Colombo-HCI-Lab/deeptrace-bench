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

export interface Model {
  id: string;
  name: string;
  modality: Modality;
  adapter: boolean;
  blocked: boolean;
  summary: string | null;
  paper: Paper | null;
  trained_on: string[];
  parity: { evalset: string; items: number; max_abs_diff: number } | null;
  profile: Profile | null;
}

export interface Dataset {
  id: string;
  name: string;
  modality: Modality;
  role: string;
  status: string;
  languages: string[];
  summary: string | null;
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
  reasons: string[];
  flags: string[];
  wall_s: number | null;
}

export interface Status {
  built: string;
  eval_status: string;
  citations_checked: string | null;
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
