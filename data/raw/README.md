# Raw documents for the vector index

Put each document in the folder for its type. The folder name becomes the chunk's `doc_type`:

| Folder | Use for |
|---|---|
| `brochure/` | model brochures (features, variants, highlights) |
| `specs/` | technical specification sheets |
| `faq/` | general FAQs (test drives, brochures, lineup) |
| `service/` | service schedules, maintenance, booking, roadside assistance |
| `warranty/` | warranty, extended warranty, Super Care |

**File names:** start the name with the model id from `data/seed/models.json` to tag the document
with that model, e.g. `kushaq-brochure.pdf`, `octavia-rs-specs.pdf`. Files that don't start with a
model id (e.g. `skoda-warranty.pdf`) are tagged `all` and apply to every model.

**Formats:** `.pdf` (text-based, not scanned), `.html` / `.htm`, `.md`, `.txt`.
In Markdown, use `#` headings (chunks never cross a heading) and `|` tables (a table is never split).
Text inside `<!-- comments -->` is not indexed, so use comments for source notes.

**After adding, replacing or deleting files, rebuild the index:**

```
python -m ingestion.build_index
python -m ingestion.test_retrieval      # check the search results
```

The files ending in `-sample.md` are short SAMPLE documents compiled on 2026-10-17 from public pages
(sources are listed at the top of each). Replace them with the official Skoda India brochures and
pages when you have them, and delete the samples so they don't duplicate the real content.
Documents should not contain prices: prices always come from `data/seed/` through the calculator tools.
