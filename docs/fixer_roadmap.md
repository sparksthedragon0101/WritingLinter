# Auto-Fixer Roadmap

The linter's core purpose is that `fix_prose` produces a technically correct
manuscript for writers who can't learn the rules themselves. Detection is
secondary to correction. This document tracks what the fixer corrects today,
what it doesn't, and how to build the missing rules safely.

## The safety rule that governs everything

**A false positive is far worse than a miss.** A missed error leaves the
manuscript as the author wrote it. A false positive silently rewrites correct
prose into incorrect prose, and the author — by definition of this tool's
audience — cannot tell that it happened. Every rule below should be built
narrow, guarded, and conservative. When a case is genuinely ambiguous, leave it
flagged for the report rather than auto-fixing it.

## Currently auto-fixed

| Rule | Where |
|---|---|
| Dialogue tag punctuation (`"Hi." he said` → `"Hi," he said`) | `fix_prose`, regex |
| Missing-apostrophe contractions (30+ forms) | `fix_mechanical_errors` |
| Modal + "of" (`could of` → `could have`) | `fix_mechanical_errors` |
| `Their is/are` → `There is/are` | `fix_mechanical_errors` |
| `your` → `you're` (narrow contexts only) | `fix_mechanical_errors` |
| a/an agreement, with sound exceptions | `fix_mechanical_errors` |
| Doubled function words | `fix_mechanical_errors` |
| Lone pronoun `i` → `I` | `fix_mechanical_errors` |
| `then` → `than` after comparatives | `refine_creative_text` |
| its / it's, both directions | `refine_creative_text` |
| Sentence capitalization (dialogue-tag aware) | `refine_creative_text` |
| Comma splices → separate sentences | `fix_comma_splices` |
| Spelling (edit-distance 1, never proper nouns) | `fix_spelling` |

Regression coverage: `tests/test_fixer.py`, `tests/test_golden_corpus.py`.

## Detected but NOT auto-fixed

- `sentence_splices` — fused sentences (missing period before a capitalized
  word). Detection exists; no fix wired. Probably the cheapest remaining win.
- `possessive_errors`, `dialogue_punctuation_errors` — overlap with rules that
  already fix them; verify before adding anything.

## Blind spots, in priority order

### 1. Subject–verb agreement (highest value)

`He don't care.` / `They was late.` / `She were happy.`

The highest-frequency error class for this audience, and fully mechanical.

- **Signal:** find `nsubj` → head verb pairs; compare `Number`/`Person` on the
  subject's `morph` against the verb's. Correct the verb, never the subject.
- **Scope it:** start with `be` (`was/were`, `is/are`) and `do` (`do/does`,
  `don't/doesn't`), which cover most real errors, then the general 3rd-person
  `-s` rule.
- **Pitfalls:**
  - Dialect and voice in dialogue (`"He don't know nothin'"`) is deliberate.
    Consider skipping text inside quotes, or gate it behind a flag.
  - Collective nouns (`The team is/are`) are legitimately either.
  - Intervening prepositional phrases (`The box of nails is heavy`) — trust the
    parse's `nsubj`, not proximity.
  - Coordinated subjects (`He and she were`) are plural.
- **Effort:** 1–2 days with a conjugation table for be/do/have.

### 2. Fused sentences / missing end punctuation

`He ran fast he was late.` / a paragraph-final line with no terminal mark.

- **Signal:** `detect_sentence_splices` already finds the capitalized-word case
  and returns offsets — wiring a fix is mostly plumbing.
- **Do the easy half first:** a paragraph's last line lacking terminal
  punctuation is near-risk-free to fix.
- **Pitfalls:** headings, list items, and deliberate fragments have no end
  punctuation on purpose. Check the line's shape before appending a period.
- **Effort:** half a day for the paragraph-end case; the fused-sentence fix
  inherits whatever precision the existing detector has — measure it first.

### 3. me / I case errors

`Me and him went to the store.` → `He and I went to the store.`

- **Signal:** accusative pronouns (`Case=Acc`) sitting in an `nsubj` slot,
  typically in a coordinated subject.
- **Pitfalls:** `Me` is correct as an object (`between you and me`) — only touch
  pronouns actually in subject position. Reordering (`Me and him` → `He and I`)
  is a second, separate transformation; case-fixing alone gives `I and he`,
  which is grammatical but stilted. Decide whether to reorder.
- **Effort:** ~half a day; high precision achievable.

### 4. Context homophones

`affect/effect`, `loose/lose`, `whose/who's`, general `your/you're`,
`their/there/they're` beyond the cases already handled.

- **Approach:** one rule at a time, each with its own POS/dep pattern and its
  own guard tests — exactly the shape of the existing `then/than` rule.
- **Pitfalls:** these need real context. `effect` is a legitimate verb ("to
  effect change"); `your` before a gerund is often correct ("your running
  shoes"). This is why the current `your` rule is deliberately narrow.
- **Effort:** a few hours each. Add them incrementally; never with a blanket
  regex.

### 5. Double negatives — recommend NOT fixing

`I don't know nothing.`

Detect if you like, but auto-correcting flattens character voice in dialogue and
dialect. Leave it in the report.

## How to build a new rule here

The process that caught four real bugs while building the comma-splice rule:

1. **Write the battery first.** Two lists: sentences that must be fixed, and
   correct sentences that must be left byte-identical. Include the adversarial
   ones (dialogue tags, lists, subordinate clauses, appositives).
2. **Inspect the parse before writing logic.** Print `pos_ / tag_ / dep_ /
   head / morph` for each token. Never guess the dependency labels — several
   assumptions were wrong (`When` attaches as `advmod`, not `mark`; `was
   raining` puts the finite verb on an `aux`, not the head).
3. **Test in paragraphs, not just sentences.** spaCy's parse of a sentence
   changes with surrounding context. One bug only appeared when a third
   sentence was added.
4. **Run it over a page of correct prose.** Zero false positives there is the
   release gate.
5. **Pin every bug you hit as a named regression test.**

## Known trap: report keys

`sentence_splices` (fused sentences) and `comma_splices` are different checks
with confusingly similar names. Adding a report key means touching: `lint_prose`,
the NLP-unavailable fallback list, `issue_keys`, `calculate_prose_score` (if it
should count), `format_errors`, `compact_report`'s mapping in the CLI, the
`--structure` filter, the vscode diagnostics list, and `docs/json_schema.md`.
The HTML report renders `report_content` wholesale, so it needs no change.
