import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const docs = defineCollection({
  loader: glob({ pattern: "*.md", base: "./src/content/docs" }),
  schema: z.object({
    title: z.string(),
    // Where a doc sits in the nav, and its group, live in src/lib/nav.ts.
    description: z.string(),
  }),
});

export const collections = { docs };
