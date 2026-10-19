import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const docs = defineCollection({
  loader: glob({ pattern: "*.md", base: "./src/content/docs" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    // "start": how it works, getting started, progress; "findings": results; "benchmark": the
    // reference on models, datasets and rules; "working": how to work on it.
    group: z.enum(["start", "findings", "benchmark", "working"]),
    order: z.number(),
  }),
});

export const collections = { docs };
