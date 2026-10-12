# The site

The public site at <https://colombo-hci-lab.github.io/deeptrace-bench/>: the status page and every doc, built with [Astro](https://astro.build) and published by `.github/workflows/pages.yml` on every push to `main` that touches `docs/`.

```bash
npm ci            # once, needs Node 22.12 or later
npm run dev       # http://localhost:4321/deeptrace-bench/
npm run build     # into dist/; fails on a broken doc link
```

- `src/content/docs/*.md`: the docs. Front matter carries `title`, `description`, `group` (`benchmark` or `working`) and `order`; the layout prints the title, so a doc has no H1. Link another doc as `models.md#parity` and a repo file as `/configs/eval/default.yaml`.
- `src/reference/status.json` and `src/content/docs/catalog.md`: written by `uv run scripts/build_site.py` from the store. Don't edit them; the descriptions they draw on are in `src/reference/catalog.yaml`.
- `src/pages/index.astro` and `src/components/`: the status page, rendered from `status.json` at build time.
- `src/styles/site.css`: the design tokens and shared styles.
