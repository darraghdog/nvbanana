# Examples from the book sources

These examples trace source ideas into the SDG pipeline. The status describes the step verified for each example.

## Hoffmann — counters in intersecting lines

In Chapter VI, No. XI, Hoffmann asks how to arrange counters so they form rows in several directions. The adapted grid rule crops to a frame, recolours its border, and marks a crossing of four lines with a yellow plus.

**Pipeline status.** The illustrated puzzle is RD1 pair `20854`. It passed the three-program agreement check on all 30 generated inputs. [Puzzles Old and New (1893), p. 267](https://archive.org/details/puzzlesoldnew00hoff) · [Generated puzzle figure](../README.md#from-book-page-to-arc-puzzle)

## Dudeney — The Six Sheep-Pens

Dudeney asks how to recover six equal enclosures with one fewer match. The extracted ARC spec counts separate hollow shapes and draws the same number of holes in a compact grid with shared walls.

**Pipeline status.** Gemini 3.1 Pro extraction `dudeney1917_visual__single__p0067__pro31_m01` was used in 32 INSP4 generation inputs with ARC training-task anchors. [Amusements in Mathematics (1917), puzzle 205](https://www.gutenberg.org/cache/epub/16713/pg16713-images.html)

## Laisant — a geometric multiplication table

Laisant illustrates products as rectangles between divisions on two axes. The extracted ARC spec extends lines from edge markers to divide a grid into rectangles, then fills only the regions that are square.

**Pipeline status.** Gemini 3.1 Pro extraction `laisant_visual__single__p0055__pro31_m01` was used in 32 INSP4 generation inputs with ARC training-task anchors. [Initiation mathématique (1906), printed p. 46](https://archive.org/details/initiationmathm00laisgoog)
