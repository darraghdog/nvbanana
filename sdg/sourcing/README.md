# Book Sourcing

Look for human-designed visual mechanisms that can become rules for 2D grid puzzles. Useful sources include puzzle collections, recreational mathematics, game manuals, and illustrated engineering books. Searches across editions, archives, and bibliographies help find new mechanisms and identify duplicate scans.

## Example books

These are three examples from the source collection. The archive links identify the scans.

| Author | Scanned edition | What to look for |
| --- | --- | --- |
| Professor Hoffmann | *Puzzles Old and New*, Frederick Warne, 1893 | Counters, line crossings, and geometric arrangements. Page 267 supplies the worked example in the [SDG guide](../README.md). [Boston Public Library scan](https://archive.org/details/puzzlesoldnew00hoff) |
| Henry Dudeney | *Amusements in Mathematics*, Nelson, 1917 | Dissections, constrained arrangements, and counting with figures. [Boston College Libraries scan](https://archive.org/details/amusementsinmath00dude) |
| Henry Dudeney | *The Canterbury Puzzles and Other Curious Problems*, second edition, Nelson, 1919 | Spatial arrangements, paths, and constraints from the illustrated puzzles. [University of Toronto scan](https://archive.org/details/canterburypuzzle00dudeuoft) |

## Prompts

1. **Find candidate books.** Use the [book discovery prompt](prompts/book_discovery.txt) to look for promising editions and representative pages. This prompt is reconstructed from the sourcing method. The candidate record includes the catalogue link, exact edition, rights evidence, and pages to inspect. Check those details against the archive entry.
2. **Screen pages with Gemini Flash.** Give the model one rendered page, its OCR when available, and basic source metadata. The [page scout prompt](prompts/page_scout_flash.txt) produces JSON labels for ARC fit, priority, visual dependence, and likely mechanisms.
3. **Extract mechanisms with Gemini Pro.** Give the model the shortlisted page image, OCR, metadata, and scout labels. The [extraction prompt](prompts/extract_mechanisms_pro.txt) produces up to four mechanisms in NVARC's seven-section spec format. It can return an empty list when the page does not support a suitable mechanism.

These are the original INSP4 prompts. That run used Gemini 3 Flash Preview for page screening and Gemini 3.1 Pro Preview for mechanism extraction. The Hoffmann demo attaches the same prompt files and defaults to Gemini 4 Argon for extraction. Later production rounds used their own prompt variants. [Hoffmann demo dataset](https://www.kaggle.com/datasets/darraghdog/hoffmann-1893-book-sdg-demo)

Use each source as a rule idea for newly generated grids. The demo pairs the specs with ARC-AGI-2 training-task summaries. [Book to ARC specs with Gemini](https://www.kaggle.com/code/darraghdog/book-to-arc-specs-with-gemini)
