// The deeptrace-bench site: the guided pages and every doc, served by GitHub Pages from the
// build of this folder (see .github/workflows/pages.yml).
import { satteri } from "@astrojs/markdown-satteri";
import { defineConfig } from "astro/config";
import { checkLinks, links } from "./src/lib/links.mjs";

const base = "/deeptrace-bench/";

// Code in the page's own ink on the tint: no second palette. Comments recede, keywords get weight
// rather than hue, strings take the petrol end of the score scale and numbers the sienna accent.
const paper = {
  name: "paper",
  type: "light",
  colors: { "editor.background": "#f5f1ec", "editor.foreground": "#171412" },
  tokenColors: [
    { scope: ["comment", "punctuation.definition.comment"], settings: { foreground: "#69625c", fontStyle: "italic" } },
    { scope: ["string", "constant.other.symbol"], settings: { foreground: "#115667" } },
    { scope: ["keyword", "storage", "keyword.control", "support.function.builtin"], settings: { foreground: "#171412", fontStyle: "bold" } },
    { scope: ["constant.numeric", "constant.language"], settings: { foreground: "#994407" } },
    { scope: ["variable.parameter", "variable.other.option", "punctuation.definition.variable"], settings: { foreground: "#69625c" } },
  ],
};

export default defineConfig({
  site: "https://colombo-hci-lab.github.io",
  base,
  trailingSlash: "always",
  integrations: [checkLinks({ base })],
  markdown: {
    processor: satteri({ mdastPlugins: [links({ base })] }),
    shikiConfig: { theme: paper },
  },
});
