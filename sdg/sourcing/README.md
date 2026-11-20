# Book Sourcing

Look for human-designed visual mechanisms that can become rules for 2D grid puzzles. Useful sources include puzzle collections, recreational mathematics, game manuals, and illustrated engineering books. Searches across editions, archives, and bibliographies help find new mechanisms and identify duplicate scans.

## Example books

These are three examples from the source collection. The archive links identify the scans.

| Author | Scanned edition | What to look for |
| --- | --- | --- |
| Professor Hoffmann | *Puzzles Old and New*, Frederick Warne, 1893 | Counters, line crossings, and geometric arrangements. Page 267 supplies the worked example in the [SDG guide](../README.md). [Boston Public Library scan](https://archive.org/details/puzzlesoldnew00hoff) |
| Henry Dudeney | *Amusements in Mathematics*, Nelson, 1917 | Dissections, constrained arrangements, and counting with figures. [Boston College Libraries scan](https://archive.org/details/amusementsinmath00dude) |
| Henry Dudeney | *The Canterbury Puzzles and Other Curious Problems*, second edition, Nelson, 1919 | Spatial arrangements, paths, and constraints from the illustrated puzzles. [University of Toronto scan](https://archive.org/details/canterburypuzzle00dudeuoft) |

[Examples from the book sources](examples.md) trace three documented inspirations into the pipeline.

## Example extraction outputs

[Download 240 INSP4 specs](inspiration_outputs.jsonl), with 60 each from four public-domain editions. Each record keeps the original seven-section ARC summary, the model and prompt version, a stable mechanism ID, and the source scan and page index. The file contains generated rule descriptions and citation metadata. The scan page index can differ from the printed page number.

| Source edition | Specs | Public-domain evidence |
| --- | ---: | --- |
| [Hoffmann, *Puzzles Old and New* (1893)](https://archive.org/details/puzzlesoldnew00hoff) | 60 | [Public-domain record for the same scan](https://commons.wikimedia.org/wiki/File:Puzzles_old_and_new_(IA_puzzlesoldnew00hoff).pdf) |
| [Dudeney, *Amusements in Mathematics* (1917)](https://archive.org/details/amusementsinmath00dude) | 60 | [Project Gutenberg rights record](https://www.gutenberg.org/ebooks/16713) |
| [Dudeney, *The Canterbury Puzzles*, second edition (1919)](https://archive.org/details/canterburypuzzle00dudeuoft) | 60 | [Project Gutenberg second edition](https://www.gutenberg.org/files/27635/27635-h/27635-h.htm) |
| [Laisant, *Initiation mathématique* (1906)](https://archive.org/details/initiationmathm00laisgoog) | 60 | The scan identifies the work as public domain. [Publication record](https://books.google.com/books/about/Initiation_math%C3%A9matique.html?id=S94EAAAAYAAJ) |

We kept grade-A, image-supported extractions with all seven sections, selected at most one per scan page, and sampled within each edition (seed 2026), retaining the worked Dudeney and Laisant specs. The full extraction export from that round is not part of this release. These rows are examples of the extraction output. The [mixing guide](../mixing/README.md) uses them to build generation inputs.

## Prompts

1. **Find candidate books.** Use the [book discovery prompt](prompts/book_discovery.txt) to look for promising editions and representative pages. This prompt is reconstructed from the sourcing method. The candidate record includes the catalogue link, exact edition, rights evidence, and pages to inspect. Check those details against the archive entry.
2. **Screen pages with Gemini Flash.** Give the model one rendered page, its OCR when available, and basic source metadata. The [page scout prompt](prompts/page_scout_flash.txt) produces JSON labels for ARC fit, priority, visual dependence, and likely mechanisms.
3. **Extract mechanisms with Gemini Pro.** Give the model the shortlisted page image, OCR, metadata, and scout labels. The [extraction prompt](prompts/extract_mechanisms_pro.txt) produces up to four mechanisms in NVARC's seven-section spec format. It can return an empty list when the page does not support a suitable mechanism.

The screening and extraction prompts are the original INSP4 versions. That run used Gemini 3 Flash Preview for page screening and Gemini 3.1 Pro Preview for mechanism extraction. The Hoffmann demo attaches the same prompt files and defaults to Gemini 4 Argon for extraction. Later production rounds used their own prompt variants. [Hoffmann demo dataset](https://www.kaggle.com/datasets/darraghdog/hoffmann-1893-book-sdg-demo)

Use each source as a rule idea for newly generated grids. The demo pairs the specs with ARC-AGI-2 training-task summaries. [Book to ARC specs with Gemini](https://www.kaggle.com/code/darraghdog/book-to-arc-specs-with-gemini)
