import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const docs = defineCollection({
  loader: glob({ pattern: "*.md", base: "./src/content/docs" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    // "benchmark": what the benchmark is and what it found; "working": how to work on it.
    group: z.enum(["benchmark", "working"]),
    order: z.number(),
  }),
});

export const collections = { docs };
