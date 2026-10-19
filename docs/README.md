# The site

The public site at <https://colombo-hci-lab.github.io/deeptrace-bench/>: five guided pages and every doc, built with [Astro](https://astro.build) and published by `.github/workflows/pages.yml` on every push to `main` that touches `docs/`.

```bash
npm ci            # once, needs Node 22.12 or later
npm run dev       # http://localhost:4321/deeptrace-bench/
npm run build     # into dist/; fails on a broken doc link
```

- `src/content/docs/*.md`: the docs. Front matter carries `title`, `description`, `group` (`start`: how it works, getting started and progress; `findings`: results; `benchmark`: the reference on models, datasets and rules; `working`: how to work on it) and `order`; the layout prints the title, so a doc has no H1. Link another doc as `models.md#parity` and a repo file as `/configs/eval/default.yaml`.
- `src/reference/status.json` and `src/content/docs/catalog.md`: written by `uv run scripts/build_site.py` from the store. Don't edit them; the descriptions they draw on are in `src/reference/catalog.yaml`.
- `src/pages/`: the guided pages, `index.astro` (overview), `results.astro`, `progress.astro`, `method.astro` (how it works) and `inventory.astro` (models and data, in four tabs chosen by `#coverage`, `#models`, `#datasets` or `#compute`), rendered from `status.json` at build time; `docs/[slug].astro` for every doc at `docs/<slug>/`; `[slug].astro` leaves a redirect at each doc's old root URL.
- `src/lib/guide.ts`: what the overview, Results, Progress and How it works take from `results.md`, `progress.md` and `method.md`, sliced by heading. Keep those docs' H2 and H3 titles, the `- **Label.** body` form of their lists and `method.md`'s tables. The overview's finding cards are the first three Insights. The Results page leads with three of `results.md`'s sections: `## The datasets` (a table), `## The detectors` (a table whose first column is the model id in a code span, giving each grid row its type, backbone and training data) and `## Insights` (a numbered `**Label.** evidence` list); the grid itself is built from `status.json`, so it follows every rebuild. Links to `results.md`, `progress.md` and `method.md` land on the guided pages; their full text stays under `docs/`. `docs/index.astro` lists every doc from its front matter. Docs merged into others are in `MOVED` in `src/lib/links.mjs`: their old addresses redirect, and the build refuses a link to one.
- `src/components/Sidebar.astro`: the one sidebar every page shares, built from the docs' `group` and `order`; `src/layouts/Base.astro` places it, tracks the section in view, and puts it in a Menu on narrow screens.
- `src/styles/site.css`: the design tokens and shared styles.
