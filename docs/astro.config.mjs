// The deeptrace-bench site: the guided pages and every doc, served by GitHub Pages from the
// build of this folder (see .github/workflows/pages.yml).
import { satteri } from "@astrojs/markdown-satteri";
import { defineConfig } from "astro/config";
import { checkLinks, links } from "./src/lib/links.mjs";

const base = "/deeptrace-bench/";

// Code in the page's own ink: no second palette. Comments recede, strings and keywords get
// weight rather than hue; the flag colour is kept for caveats, so code never uses it.
const paper = {
  name: "paper",
  type: "light",
  colors: { "editor.background": "#e7e9e3", "editor.foreground": "#1b1d1a" },
  tokenColors: [
    { scope: ["comment", "punctuation.definition.comment"], settings: { foreground: "#5e625b", fontStyle: "italic" } },
    { scope: ["string", "constant.other.symbol"], settings: { foreground: "#3d5a45" } },
    { scope: ["keyword", "storage", "keyword.control", "support.function.builtin"], settings: { foreground: "#1b1d1a", fontStyle: "bold" } },
    { scope: ["constant.numeric", "constant.language"], settings: { foreground: "#3d4f6b" } },
    { scope: ["variable.parameter", "variable.other.option", "punctuation.definition.variable"], settings: { foreground: "#5e625b" } },
  ],
};

export default defineConfig({
  site: "https://colombo-hci-lab.github.io",
  base,
  trailingSlash: "always",
  integrations: [checkLinks()],
  markdown: {
    processor: satteri({ mdastPlugins: [links({ base })] }),
    shikiConfig: { theme: paper },
  },
});
