// The guided pages (the Timeline, Results, How it works): findings sliced out of progress.md,
// results.md and method.md by heading, joined with status.json, and the diverging AUC colour
// scale. The Markdown stays the source; these pages only lay it out.
import { markdownToHtml } from "satteri";
import { links } from "./links.mjs";
import { mdLabels } from "./nav";
import { D, dur, int, type Run } from "./status";

const base = import.meta.env.BASE_URL;

// Kept in step with site.css (--paper, --tint, --good, --bad).
export const PAPER = "#fdfcfa";
export const TINT = "#f5f1ec";
export const GOOD = "oklch(0.47 0.07 220)";
export const BAD = "oklch(0.47 0.14 45)";
// The stepped scale: neutral tint near 0.5, then four bins each way, petrol above chance and
// sienna below. Lightness and chroma per bin.
const BIN_L = [0, 0.9, 0.79, 0.57, 0.47];
const BIN_C = [0, 0.35, 0.6, 0.9, 1];
const POS = [0.07, 220];
const NEG = [0.14, 45];

// Short names for the evalsets the findings talk about, with what each holds.
export const EV: Record<string, [string, string]> = {
  urdu_csalt: ["Urdu CSALT", "Urdu"],
  banglafake: ["BanglaFake", "Bengali"],
  bangla_voices: ["Bangla voices", "Bengali"],
  indictts_challenge: ["IndicTTS", "16 languages"],
  indicsynth_hi: ["IndicSynth Hindi", "Hindi"],
  asvspoof2019_la_eval: ["ASVspoof 2019 LA", "English"],
  in_the_wild: ["In-the-Wild", "English"],
  hidf_video: ["HiDF", "Faces"],
};
const SA = ["urdu_csalt", "banglafake", "bangla_voices", "indictts_challenge", "indicsynth_hi"];

type Kind = "bad" | "mid" | "good";
// The three tiers results.md describes, by what each detector was trained on.
const TIERS: { label: string; title: string; kind: Kind; models: string[] }[] = [
  { label: "Fails", title: "Trained only on English spoofing data (ASVspoof 2019)", kind: "bad", models: ["aasist", "aasist_l"] },
  { label: "Uneven", title: "Same training data, multilingual front end", kind: "mid", models: ["xlsr_aasist"] },
  {
    label: "Near ceiling",
    title: "Post-trained on large multilingual collections of real and fake speech",
    kind: "good",
    models: ["antideepfake_w2v_small", "antideepfake_mms_300m", "df_arena_500m", "df_arena_1b"],
  },
];

const fmt3 = (x: number) => x.toFixed(3);
const pct = (x: number) => {
  const v = 100 * x;
  return (v < 10 ? v.toFixed(1) : String(Math.round(v))) + "%";
};
const pos = (v: number, lo = 0, hi = 1) => (100 * Math.max(0, Math.min(1, (v - lo) / (hi - lo)))).toFixed(2) + "%";
const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const plain = (s: string) =>
  String(s || "").replace(/\*\*/g, "").replace(/`/g, "").replace(/\[([^\]]+)\]\([^)]+\)/g, "$1").trim();
export const evName = (id: string) => (EV[id] ? EV[id][0] : id);
const runs = new Map(D.runs.map((r) => [`${r.model}|${r.evalset}`, r]));
const name = (id: string) => D.models.find((m) => m.id === id)?.name ?? id;

export interface Tone {
  bg: string;
  fg: string;
  solid: string;
}

export function tone(v: number): Tone {
  const d = Math.max(-1, Math.min(1, (v - 0.5) / 0.5));
  const k = Math.abs(d);
  const b = k < 0.2 ? 0 : k < 0.5 ? 1 : k < 0.8 ? 2 : k < 0.94 ? 3 : 4;
  const solid = d >= 0 ? GOOD : BAD;
  if (!b) return { bg: TINT, fg: "var(--ink)", solid };
  const [c, h] = d >= 0 ? POS : NEG;
  return { bg: `oklch(${BIN_L[b]} ${(c * BIN_C[b]).toFixed(3)} ${h})`, fg: b >= 3 ? PAPER : "var(--ink)", solid };
}

// Snippets of a doc, rendered with the docs' own link rules.
const md = (text: string) => (text.trim() ? mdLabels(markdownToHtml(text, { mdastPlugins: [links({ base })] }).html) : "");
// The same, for a table cell or a phrase: no paragraph around it.
const mdi = (text: string) => md(text).replace(/^\s*<p>([\s\S]*)<\/p>\s*$/, "$1");

// ---------- Markdown slicing ----------
interface Part {
  title: string;
  text: string;
}
function split(body: string, level: number): Part[] {
  const re = new RegExp(`^${"#".repeat(level)} (.+)$`, "gm");
  const heads: { title: string; start: number; end: number }[] = [];
  let m;
  while ((m = re.exec(body))) heads.push({ title: m[1].trim(), start: m.index, end: m.index + m[0].length });
  return heads.map((h, i) => ({
    title: h.title,
    text: body.slice(h.end, i + 1 < heads.length ? heads[i + 1].start : body.length).trim(),
  }));
}
const section = (parts: Part[], re: RegExp) => parts.find((s) => re.test(s.title)) ?? { title: "", text: "" };
const paras = (t: string) =>
  t
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);
// The paragraphs of a section, without its lists and tables.
const prose = (t: string) =>
  paras(t)
    .filter((p) => !/^(\||[-*] |\d+\. )/.test(p))
    .join("\n\n");
// The items of the lists in a section, continuation lines joined.
function items(text: string): string[] {
  const out: string[][] = [];
  let cur: string[] | null = null;
  for (const l of text.split("\n")) {
    const m = l.match(/^(?:[-*]|\d+\.)\s+(.*)$/);
    if (m) {
      cur = [m[1]];
      out.push(cur);
    } else if (cur && /^\s+\S/.test(l)) cur.push(l.trim());
    else if (l.trim()) cur = null;
  }
  return out.map((a) => a.join(" "));
}

// "**Label.** Body" -> { label, rest }. Run-on bold phrases keep their grammar: a parenthetical
// or a lower-case continuation joins the label, and "**X**, the thing, is ..." becomes "The thing
// is ...".
function lead(s: string) {
  const m = s.match(/^\*\*(.+?)\*\*\s*([\s\S]*)$/);
  if (!m) return { label: "", rest: s, code: false };
  let label = plain(m[1]).replace(/[.:]$/, "");
  let rest = m[2];
  const code = /^`[^`]+`$/.test(m[1].trim());
  const par = rest.match(/^(\([^)]*\))\s*[,.:;]?\s*/);
  if (par) {
    label += " " + plain(par[1]);
    rest = rest.slice(par[0].length);
  } else if (/^[a-z]/.test(rest)) {
    const e = rest.match(/^([^.;:]*?)[.;:](\s+|$)/);
    if (e) {
      label += " " + plain(e[1]);
      rest = rest.slice(e[0].length);
    }
  }
  const comma = rest.startsWith(",");
  rest = rest.replace(/^[,.:;]\s*/, "");
  if (comma) rest = rest.replace(/^([^,.;()]+),\s+(?=(?:is|are|was|were|has|have|still|can|will)\b)/, "$1 ");
  if (/^[a-z]/.test(rest)) rest = cap(rest);
  return { label, rest, code };
}

function tables(text: string) {
  const out: string[][] = [];
  let block: string[] | null = null;
  for (const l of text.split("\n")) {
    if (/^\s*\|/.test(l)) {
      if (!block) out.push((block = []));
      block.push(l);
    } else block = null;
  }
  const cells = (l: string) =>
    l
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((c) => c.trim());
  return out.map((b) => ({ head: cells(b[0]), rows: b.slice(2).map(cells) }));
}

export interface HeatCell {
  text: string;
  bg: string;
  fg: string;
  title: string;
}
// A table of AUCs from the doc, each cell on the shared scale.
function heat(t: { head: string[]; rows: string[][] } | undefined) {
  if (!t) return { first: "", cols: [] as { name: string; range: string }[], rows: [] as { name: string; cells: HeatCell[] }[] };
  const names = t.head.slice(1).map(plain);
  const vals = t.rows.map((r) => r.slice(1).map((c) => parseFloat(plain(c))));
  return {
    first: plain(t.head[0]),
    cols: names.map((n, j) => {
      const col = vals.map((v) => v[j]).filter(Number.isFinite);
      return { name: n, range: col.length ? `${Math.min(...col).toFixed(2)}–${Math.max(...col).toFixed(2)}` : "" };
    }),
    rows: t.rows.map((r, i) => ({
      name: plain(r[0]),
      cells: vals[i].map((v, j) => {
        const ok = Number.isFinite(v);
        const c = ok ? tone(v) : { bg: "transparent", fg: "var(--muted)" };
        return { text: ok ? v.toFixed(3) : "·", bg: c.bg, fg: c.fg, title: `${plain(r[0])} · ${names[j]}: AUC ${ok ? v.toFixed(3) : "n/a"}` };
      }),
    })),
  };
}

const badge = (kind: Kind) => (kind === "mid" ? "badge mid" : `badge ${kind}`);

// ---------- Progress ----------
export function buildProgress(body: string) {
  const secs = split(body, 2);
  const standT = section(secs, /^Where things stand/).text;
  const timeline = split(section(secs, /^Timeline/).text, 3).map((s, i) => {
    const m = s.title.match(/^(\d{4}-\d{2}-\d{2}):\s*(.*)$/) ?? [null, "", s.title];
    const ps = paras(s.text);
    const ci = ps.findIndex((p) => /^Commits?\b/.test(p));
    const commits = ci >= 0 ? ps.splice(ci, 1)[0] : "";
    return { date: m[1], title: cap(m[2]), latest: i === 0, commits: md(commits), body: md(ps.join("\n\n")) };
  });
  return {
    asOf: standT.match(/As of (\d{4}-\d{2}-\d{2})/)?.[1] ?? "",
    stand: items(standT)
      .map(lead)
      .map((b) => ({ label: b.label, body: md(b.rest) })),
    timeline,
    next: items(section(secs, /^Next/).text)
      .map(lead)
      .map((b, i) => ({ n: i + 1, label: b.label, body: md(b.rest), now: i === 0 })),
    decisions: items(section(secs, /^Open decisions/).text)
      .map(lead)
      .map((b) => ({ label: b.label, code: b.code, body: md(b.rest) })),
    adding: md(section(secs, /^Adding an entry/).text),
  };
}
export type Progress = ReturnType<typeof buildProgress>;

// ---------- At a glance: every detector on every dataset ----------
// The grid's columns: reproduction sets, the South Asian sets, then video, ending with FF++,
// which every video detector trained on and nobody has scored yet.
const GRID_COLS = ["asvspoof2019_la_eval", "in_the_wild", ...SA, "hidf_video", "ffpp_test"];
const GRID_NAMES: Record<string, [string, string]> = { ...EV, ffpp_test: ["FF++", "Faces · training set"] };
const VIDEO_GROUP = { label: "Faces", title: "Video detectors, all trained on FF++", models: ["xception", "sbi", "gend"] };

export interface GridCell {
  kind: "run" | "trained" | "na" | "todo";
  text: string;
  eer?: string;
  style?: string;
  title: string;
  overlap?: boolean;
  home?: boolean;
}

function buildGrid(detectors: { head: string[]; rows: string[][] } | undefined) {
  const facts = new Map((detectors?.rows ?? []).map((r) => [plain(r[0]), r.slice(1).map(plain)]));
  const trainedOn = (modelId: string, ev: string) => {
    const m = D.models.find((x) => x.id === modelId);
    const names = D.datasets.filter((d) => d.evalsets.includes(ev)).map((d) => d.name);
    return !!m && names.some((n) => m.trained_on.includes(n));
  };
  const cell = (id: string, ev: string): GridCell => {
    const m = D.models.find((x) => x.id === id);
    const e = D.evalsets.find((x) => x.id === ev);
    const r = runs.get(`${id}|${ev}`);
    const trained = trainedOn(id, ev);
    if (r) {
      const t = tone(r.auc);
      // EER on the same scale, lower is better: 0 colours like a perfect AUC.
      const e = tone(1 - r.eer);
      return {
        kind: "run",
        text: fmt3(r.auc),
        eer: pct(r.eer),
        style: `--bg-auc: ${t.bg}; --fg-auc: ${t.fg}; --bg-eer: ${e.bg}; --fg-eer: ${e.fg}`,
        overlap: r.flags.includes("source overlap"),
        home: trained,
        title:
          `AUC ${fmt3(r.auc)} (${fmt3(r.auc_lo)}–${fmt3(r.auc_hi)}), EER ${pct(r.eer)}, ${int(r.n)} clips` +
          (trained ? " · trained on its train split, scored on the held-out eval split" : "") +
          (r.flags.length ? " · " + r.flags.join(", ") : ""),
      };
    }
    if (trained || D.cells[`${id}|${ev}`] === "guard")
      return { kind: "trained", text: "trained", title: "Trained on this dataset, so it isn't scored here" };
    const audio = (x?: string) => x === "audio";
    if (m && e && audio(m.modality) !== audio(e.modality)) return { kind: "na", text: "–", title: "Different modality" };
    return { kind: "todo", text: "·", title: "Not run yet" };
  };
  const groups = [...TIERS.map((t) => ({ label: t.label, title: t.title, badge: badge(t.kind), models: t.models })), {
    ...VIDEO_GROUP,
    badge: badge("mid"),
  }];
  return {
    cols: GRID_COLS.map((ev) => ({ id: ev, name: (GRID_NAMES[ev] ?? [ev, ""])[0], sub: (GRID_NAMES[ev] ?? [ev, ""])[1] })),
    groups: groups.map((g) => ({
      label: g.label,
      title: g.title,
      badge: g.badge,
      rows: g.models
        .filter((id) => D.models.some((m) => m.id === id))
        .map((id) => {
          const [type = "", built = "", trained = ""] = facts.get(id) ?? [];
          return { name: name(id), type, built, trained, cells: GRID_COLS.map((ev) => cell(id, ev)) };
        }),
    })),
    span: GRID_COLS.length + 1,
  };
}

// ---------- Results ----------
export function buildResults(body: string) {
  const secs = split(body, 2);
  const get = (re: RegExp) => section(secs, re).text;

  const dsT = get(/^The datasets/);
  const dsTable = tables(dsT)[0];
  const datasets = {
    head: (dsTable?.head ?? []).map(plain),
    rows: (dsTable?.rows ?? []).map((r) => r.map(plain)),
    note: md(prose(dsT)),
  };
  const grid = buildGrid(tables(get(/^The detectors/))[0]);
  const insights = items(get(/^Insights/))
    .map(lead)
    .map((b, i) => ({ n: i + 1, label: b.label, body: md(b.rest) }));

  const firstT = get(/^Read this first/);
  const first = items(firstT)
    .map(lead)
    .map((b) => ({ label: b.label, body: md(b.rest) }));

  const audio = split(get(/^Audio: South Asian/), 3);
  const sub = (re: RegExp) => section(audio, re);
  const langS = sub(/^Language matters/);
  const langP = paras(langS.text).filter((p) => !p.startsWith("|"));
  const preS = sub(/^Multilingual pretraining/);
  const notes = audio
    .filter((s) => !/^(Three tiers|Multilingual pretraining|Language matters)/.test(s.title))
    .map((s) => ({ title: plain(s.title), body: md(s.text) }));

  const repT = get(/^Reproductions/);
  const rt = tables(repT)[0];
  const num = (s: string) => {
    const m = s.match(/(\d+(?:\.\d+)?)%/);
    return m ? parseFloat(m[1]) : null;
  };
  const repro = (rt ? rt.rows : []).map((r) => {
    const ours = num(r[2]);
    const pub = num(r[4]);
    const d = ours != null && pub != null ? ours - pub : null;
    return {
      model: plain(r[0]),
      evalset: plain(r[1]),
      eer: plain(r[2]),
      auc: plain(r[3]),
      published: plain(r[4]),
      diff: d == null ? "·" : `${d > 0.005 ? "+" : d < -0.005 ? "−" : "±"}${Math.abs(d).toFixed(2)}`,
    };
  });

  const vidT = get(/^Video: HiDF/);
  const raceT = tables(vidT).find((t) => /group/i.test(t.head[0]));
  const vP = paras(vidT).filter((p) => !/^(\||[-*] )/.test(p));
  const video = {
    intro: md(vP[0] ?? ""),
    raceIntro: md(vP[1] ?? ""),
    points: items(vidT)
      .map(lead)
      .map((b) => ({ label: b.label, body: md(b.rest) })),
    rows: D.runs
      .filter((r) => r.evalset === "hidf_video")
      .sort((a, b) => b.auc - a.auc)
      .map((r) => ({
        name: name(r.model),
        left: pos(r.auc, 0.5, 1),
        ciLeft: pos(r.auc_lo, 0.5, 1),
        ciWidth: (200 * (r.auc_hi - r.auc_lo)).toFixed(2) + "%",
        auc: fmt3(r.auc),
        ci: `${fmt3(r.auc_lo)}–${fmt3(r.auc_hi)}`,
        eer: pct(r.eer),
        dot: tone(r.auc).solid,
      })),
    // A bold cell in the doc's race table is the gap it calls out.
    race: raceT
      ? {
          head: raceT.head.map(plain),
          rows: raceT.rows.map((r) => r.map((c) => ({ text: plain(c), hot: /^\*\*.*\*\*$/.test(c) }))),
        }
      : { head: [] as string[], rows: [] as { text: string; hot: boolean }[][] },
  };

  const evOf = (r: Run) => D.evalsets.find((x) => x.id === r.evalset);
  const grp = (r: Run) => {
    const e = evOf(r);
    return e?.role === "reproduction" ? "repro" : e?.modality !== "audio" ? "video" : "sa";
  };
  const ord: Record<string, number> = { repro: 0, sa: 1, video: 2 };
  const all = D.runs
    .slice()
    .sort((a, b) => ord[grp(a)] - ord[grp(b)] || a.evalset.localeCompare(b.evalset) || b.auc - a.auc);
  const filters = (
    [
      ["all", "All"],
      ["repro", "Reproductions"],
      ["sa", "South Asian audio"],
      ["video", "Video"],
    ] as const
  ).map(([id, label]) => ({ id, label, n: id === "all" ? all.length : all.filter((r) => grp(r) === id).length }));

  // Each row of the experiments table becomes a card: solid once it has runs, dashed while planned.
  const expT = get(/^Experiments/);
  const expRows = tables(expT)[0]?.rows ?? [];
  const experiments = {
    intro: md(paras(expT)[0] ?? ""),
    cards: expRows.map((r) => {
      const [id, title = ""] = plain(r[0]).split(/:\s*/);
      const status = plain(r[4]);
      const kind = /provisional/i.test(status) ? "prov" : /\bdone\b/i.test(status) ? "run" : "plan";
      const chip = kind === "prov" ? "Provisional" : kind === "plan" ? "Planned" : /^done\b/i.test(status) ? "Done" : "Partly done";
      return { id, title: cap(title), q: plain(r[1]), status, kind, chip };
    }),
    after: md(paras(expT).slice(1).filter((p) => !p.startsWith("|")).join("\n\n")),
  };

  return {
    experiments,
    asOf: firstT.match(/as of (\d{4}-\d{2}-\d{2})/)?.[1] ?? D.built,
    first,
    datasets,
    grid,
    insights,
    lang: { title: plain(langS.title), ...heat(tables(langS.text)[0]), intro: md(langP[0] ?? ""), after: md(langP.slice(1).join("\n\n")) },
    pre: { title: plain(preS.title), ...heat(tables(preS.text)[0]), body: md(prose(preS.text)) },
    notes,
    repro,
    reproBody: md(prose(repT)),
    video,
    runs: {
      filters,
      rows: all.map((r) => {
        const t = tone(r.auc);
        return {
          group: grp(r),
          evalset: evName(r.evalset),
          evalsetId: r.evalset,
          model: name(r.model),
          realFake: `${int(r.n_real)} / ${int(r.n_fake)}`,
          failed: int(r.n_failed),
          auc: fmt3(r.auc),
          bg: t.bg,
          fg: t.fg,
          ci: `${fmt3(r.auc_lo)}–${fmt3(r.auc_hi)}`,
          eer: pct(r.eer),
          wall: r.wall_s ? dur(r.wall_s) : "·",
          flags: r.flags,
          title: [r.run_id, ...r.reasons].join("\n"),
        };
      }),
    },
    gaps: md(get(/^What the results can/)),
  };
}

// ---------- How it works ----------
export function buildMethod(body: string) {
  const secs = split(body, 2);
  const first = body.search(/^## /m);
  const rows = (re: RegExp, i = 0) => tables(section(secs, re).text)[i]?.rows ?? [];
  const repoLink = (cell: string) => {
    const path = (cell.match(/`([^`]+)`/)?.[1] ?? "").replace(/\/$/, "");
    return /^[\w./-]+$/.test(path) && path !== "DTB_ROOT" ? path : "";
  };
  return {
    lede: mdi(prose(first > 0 ? body.slice(0, first) : body)),
    plain: md(section(secs, /^In plain words/).text),
    steps: rows(/^The pipeline/).map((r, i) => ({
      kicker: `${i + 1} · ${plain(r[1])}`,
      title: plain(r[0]),
      body: mdi(r[2]),
      writes: mdi(r[3]),
    })),
    scripts: rows(/^The pipeline/, 1).map((r) => ({ name: mdi(r[0]), body: mdi(r[1]) })),
    after: md(prose(section(secs, /^The pipeline/).text)),
    checks: rows(/^The checks/).map((r) => ({ title: plain(r[0]), body: mdi(r[1]) })),
    terms: split(section(secs, /^Key terms/).text, 3).map((g) => ({
      title: plain(g.title),
      rows: (tables(g.text)[0]?.rows ?? []).map((r) => ({ term: plain(r[0]), body: mdi(r[1]) })),
    })),
    repo: rows(/^The repo/).map((r) => ({ label: plain(r[0]), path: repoLink(r[0]), body: mdi(r[1]) })),
  };
}
