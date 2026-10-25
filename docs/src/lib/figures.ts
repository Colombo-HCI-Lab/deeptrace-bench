// The data behind the figures set between a doc's blocks (components/figures/) and the AUC primer
// on Results: illustrative distributions, waveform glyphs and the few facts each figure draws.
// FIGS says where each figure sits in its doc.
import { tone } from "./guide";

// [heading id ("" = the top of the doc), blocks after it, figure]: the figure sits after that many
// blocks. A heading that's gone drops its figure, so a doc can be rewritten without breaking one.
export const FIGS: Record<string, [string, number, string][]> = {
  evaluation: [
    ["evalsets-not-datasets", 2, "pairing"],
    ["preprocessing", 0, "windows"],
    ["preprocessing", 0, "faces"],
    ["metrics", 1, "threshold"],
    ["contamination-verdicts", 1, "verdicts"],
  ],
  store_layout: [
    ["naming-rules", 1, "runid"],
    ["namespaces", 1, "namespaces"],
  ],
  getting_started: [
    ["first-run-the-smoke-test", 2, "smokeflow"],
    ["reading-the-report", 3, "crops"],
  ],
  adding_a_model: [
    ["", 1, "recipe"],
    ["5-before-any-result-counts", 0, "parity"],
  ],
  adding_a_dataset: [
    ["", 1, "recipe"],
    ["3-sample-it-then-build-the-manifest", 2, "sampler"],
  ],
  models: [["parity", 2, "paritydots"]],
  catalog: [["", 1, "modeltable"]],
  datasets: [["", 1, "datatable"]],
  spark: [
    ["curnagl", 5, "storage"],
    ["curnagl", 8, "jobs"],
  ],
};

const REAL = "var(--real)";
const REAL_X = "var(--muted)";
const FAKE = "var(--fake)";
const FAKE_X = "var(--bad)";

const erf = (x: number) => {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
};
const Phi = (z: number) => 0.5 * (1 + erf(z / Math.SQRT2));
const pc = (x: number) => (100 * x).toFixed(2) + "%";
const SD = 0.1;

// Mirrored histogram: real clips above the axis, fakes below; bars on the wrong side of thr go dark.
function hist(mr: number, mf: number, thr: number, n = 25) {
  const g = (m: number, x: number) => Math.exp(-0.5 * ((x - m) / SD) ** 2);
  return Array.from({ length: n }, (_, i) => {
    const x = (i + 0.5) / n;
    return { r: pc(g(mr, x)), f: pc(g(mf, x)), rbg: x > thr ? REAL_X : REAL, fbg: x < thr ? FAKE_X : FAKE };
  });
}

// A deterministic speech-like envelope for waveform glyphs.
function wave(n: number, seed: number, lo: number, hi: number, env = 0) {
  return Array.from({ length: n }, (_, i) => {
    const v = Math.abs(Math.sin((i + 1) * 1.7 + seed) * Math.cos((i + 1) * 0.37 + seed * 2));
    const e = env ? 0.3 + 0.7 * Math.abs(Math.sin((i + seed * 10) / env)) : 1;
    return { h: pc(lo + (hi - lo) * v * e), c: "var(--stone)" };
  });
}

// The AUC primer on Results: four detectors, from near perfect to backwards.
export const panels = (
  [
    ["Near perfect", 0.33, "Almost every fake scores above every real clip."],
    ["Useful", 0.15, "Mostly right, with a band of overlap where the errors are."],
    ["Coin flip", 0, "Real and fake score alike: no signal at all."],
    ["Backwards", -0.15, "Fakes look more real than real clips: a bug on a reproduction set, a finding on a South Asian one."],
  ] as const
).map(([label, d, note]) => {
  const auc = Phi(d / (SD * Math.SQRT2));
  const t = tone(auc);
  return { label, note, auc: auc.toFixed(2), eer: Math.round(100 * Phi(-d / (2 * SD))) + "%", bg: t.bg, fg: t.fg, thr: "50%", bars: hist(0.5 - d / 2, 0.5 + d / 2, 0.5) };
});

// Two groups that rank equally well, one global threshold (the overall EER point) between them.
function thresholdGroups() {
  const G: [string, number, number][] = [
    ["Group A", 0.36, 0.64],
    ["Group B", 0.43, 0.71],
  ];
  const fpr = (t: number, m: number) => 1 - Phi((t - m) / SD);
  const fnr = (t: number, m: number) => Phi((t - m) / SD);
  let lo = 0;
  let hi = 1;
  for (let k = 0; k < 40; k++) {
    const t = (lo + hi) / 2;
    if (G.reduce((a, g) => a + fpr(t, g[1]) - fnr(t, g[2]), 0) > 0) lo = t;
    else hi = t;
  }
  const thr = (lo + hi) / 2;
  return G.map(([name, mr, mf], i) => ({
    name,
    auc: Phi((mf - mr) / (SD * Math.SQRT2)).toFixed(2),
    fpr: Math.round(100 * fpr(thr, mr)) + "%",
    fnr: Math.round(100 * fnr(thr, mf)) + "%",
    fprFg: i ? "var(--bad)" : "var(--ink)",
    thr: pc(thr),
    bars: hist(mr, mf, thr),
  }));
}

// Log axis 1e-7..1e-2 over 6%..100%; an exact match (0) sits at the left edge.
const PLOG = (v: number) => (v > 0 ? 6 + 94 * Math.max(0, Math.min(1, (Math.log10(v) + 7) / 5)) : 0).toFixed(2) + "%";

// One dot per model in models.md's parity table. A row lists "Name value" pairs; a bare size
// ("1B 0" after "DF Arena 500M 0") takes the family name before it.
export function parityDots(body: string) {
  const s = body.search(/^## Parity\s*$/m);
  if (s < 0) return [];
  const rest = body.slice(s + 3);
  const e = rest.search(/^## /m);
  const sec = e < 0 ? rest : rest.slice(0, e);
  const table = sec.split(/\n\s*\n/).find((b) => /^\|/.test(b.trim())) ?? "";
  const rows = table
    .split("\n")
    .filter((l) => /^\|/.test(l))
    .slice(2)
    .map((l) => l.split("|").slice(1, -1).map((c) => c.trim().replace(/`/g, "")));
  const dots: { name: string; v: number; title: string }[] = [];
  for (const [models, checked, upstream] of rows) {
    let family = "";
    for (const m of models.matchAll(/([^,]+?)\s+(\d+(?:\.\d+)?(?:e-?\d+)?)\s*(?:,|$)/g)) {
      let n = m[1].trim();
      if (!n.includes(" ") && family) n = `${family} ${n}`;
      else family = n.split(" ").slice(0, -1).join(" ");
      dots.push({ name: n, v: parseFloat(m[2]), title: `${n}: against ${upstream}, ${checked}` });
    }
  }
  return dots
    .sort((a, b) => a.v - b.v)
    .map((d) => ({ name: d.name, title: d.title, val: d.v ? d.v.toExponential(1) : "0", left: PLOG(d.v), bg: d.v ? "var(--good)" : "var(--card)" }));
}

const w15 = wave(15, 0.8, 0.12, 0.95);
const WIN = [0.91, 0.84, 0.89];

// What every figure draws, apart from the per-doc parts (the recipe, the parity dots).
export const FG = {
  panels,
  groups: thresholdGroups(),
  pw: { sameR: wave(20, 1, 0.18, 0.85), sameF: wave(20, 1.35, 0.18, 0.85), crossR: wave(20, 2.2, 0.08, 0.32), crossF: wave(20, 3.1, 0.5, 1) },
  wLong: wave(72, 0.3, 0.06, 0.95, 6),
  wShort: w15.concat(w15.slice(0, 9).map((b) => ({ ...b, c: "var(--stone-2)" }))),
  wTiny: wave(4, 2, 0.25, 0.7),
  wins: WIN.map((s, i) => ({ l: pc(i / 3), w: "calc(33.333% - 4px)", s: s.toFixed(2) })),
  wMean: (WIN.reduce((a, b) => a + b, 0) / WIN.length).toFixed(2),
  frames: ["36%", "39%", "42%", "44%", "42%", "39%", "36%", "34%"].map((l) => ({ l })),
  crops: (
    [
      ["Centred, upright, eyes level", true, "none", false],
      ["Black edge where the 1.3 margin runs past the frame", true, "none", true],
      ["Off-centre", false, "translate(-30%, -24%)", false],
      ["Rotated", false, "rotate(32deg)", false],
      ["Mostly background", false, "translate(30%, 28%) scale(0.36)", false],
    ] as const
  ).map(([label, good, tf, edge]) => ({ label, tf, edge, mark: good ? "✓" : "✗", ring: good ? "var(--good)" : "var(--bad)" })),
  sample: (
    [
      ["real/spk01_s04.wav", "real", 1],
      ["real/spk02_s11.wav", "real", 0],
      ["real/spk03_s07.wav", "real", 1],
      ["fake/spk01_s04.wav", "fake", 0],
      ["fake/spk02_s11.wav", "fake", 1],
      ["fake/spk03_s07.wav", "fake", 1],
      ["__MACOSX/._spk01_s04.wav", "None", 0],
      ["README.txt", "None", 0],
    ] as const
  ).map(([path, label, p]) => ({
    path,
    label,
    pick: p ? "●" : "",
    bg: p ? "var(--tint)" : "transparent",
    fg: label === "None" ? "var(--muted)" : label === "fake" ? FAKE_X : "var(--ink)",
  })),
  // Curnagl's storage, from spark.md's Curnagl section.
  storage: [
    { label: "Kept", rule: "var(--ink)", bs: "solid", items: [{ path: "git", meta: "the repo", body: "Code, configs, docs, weight hashes and published aggregates." }] },
    {
      label: "Lasting",
      rule: "var(--muted)",
      bs: "solid",
      items: [
        { path: "/users/<you>", meta: "50 GB", body: "The repo clone only: not the venv, not the uv cache." },
        { path: "/work/<project>/deeptrace-bench", meta: "DTB_ROOT · no backup", body: "Datasets, weights, scores and results, plus the venv and uv cache. No millions of small files." },
      ],
    },
    {
      label: "Temporary",
      rule: "var(--rule-3)",
      bs: "dashed",
      items: [
        { path: "$TMPDIR/dtb-cache", meta: "node-local · wiped after the job", body: "DTB_CACHE inside jobs: frames and crops." },
        { path: "/scratch", meta: "purged when it fills", body: "Nothing that must survive." },
      ],
    },
  ],
  pticks: (
    [
      ["0", 0],
      ["1e-6", 1e-6],
      ["1e-5", 1e-5],
      ["1e-4", 1e-4],
      ["1e-3", 1e-3],
      ["1e-2", 1e-2],
    ] as const
  ).map(([t, v]) => ({ t, left: PLOG(v), tf: v === 1e-2 ? "translateX(-100%)" : v ? "translateX(-50%)" : "none" })),
  ptol: PLOG(1e-3),
};

// The recipe at the top of the two "adding" docs: one card per numbered section.
const RECIPES: Record<string, { steps: [string, string, string][]; endTitle: string; endNote: string }> = {
  adding_a_model: {
    steps: [
      ["Config", "configs/models/<id>.yaml", "Upstream commit, weights, training data, input"],
      ["Fetch and pin", "setup_models.py <id> --record", "Every file hashed into the lock"],
      ["Adapter", "models/<module>.py", "convert_checkpoint, load, score"],
      ["Prove it runs", "smoke.py --model <id>", "Tests pass; the report and crops look right"],
      ["Before it counts", "parity.py", "Parity within 1e-3, then AUC above 0.5 on a reproduction set"],
    ],
    endTitle: "Give it a wave",
    endNote: "Its numbers can now appear in results.",
  },
  adding_a_dataset: {
    steps: [
      ["Config", "configs/datasets/<id>.yaml", "Access, licence, derived_from, role"],
      ["Builder", "datasets/<id>.py", "A label for each path, and build_manifest"],
      ["Sample, then build", "setup_datasets.py <id> --sample 2", "A few MB in the smoke namespace, then the full set"],
      ["Evalset", "configs/evalsets/<id>.yaml", "Components, pairing, group_by"],
      ["Prove it runs", "smoke.py --evalsets <id>", "Samples, scores and reports it"],
    ],
    endTitle: "Ready for real runs",
    endNote: "Add a builder test on an invented tree.",
  },
};

// Each step links to its numbered section ("## 1. Config" and on).
export function recipe(slug: string, sections: { href: string; text: string }[]) {
  const R = RECIPES[slug];
  if (!R) return null;
  const numbered = sections.filter((s) => /^\d+\./.test(s.text));
  return {
    steps: R.steps.map(([title, file, note], i) => ({ n: String(i + 1), title, file, note, href: numbered[i]?.href ?? "#" })),
    endTitle: R.endTitle,
    endNote: R.endNote,
  };
}
