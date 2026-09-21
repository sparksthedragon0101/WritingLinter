# Writing Linter JSON Schema (v1.0)

The Writing Linter outputs a structured JSON response when the `--json` flag is used (default). This document outlines the schema to enable predictable integrations with other tools.

## Root Object

```json
{
  "version": "1.0",
  "timestamp": "2026-05-11T12:00:00",
  "metadata": {},
  "text_content": {},
  "metrics": {},
  "diagnostics": {},
  "suppressions": []
}
```

### 1. Metadata
Contains summary information about the text and the analysis goals.
- `word_count` (integer)
- `goal_name` (string): e.g., "Creative", "Academic"
- `reading_time` (string)
- `avg_sent_len` (float)
- `dialogue_ratio` (float)
- `passive_perc` (float)

### 2. Text Content
Contains the raw input and the AI-refined output.
- `original` (string): The raw text provided to the linter.
- `fixed` (string): The refined text produced by the AI fixer.

### 3. Metrics
Scoring and quantitative analysis of the text.
- `readability`: Contains Flesch Reading Ease, Flesch-Kincaid Grade, etc.
- `tone`: Tone confidence scores.
- `rhythm`: Paragraph rhythm metrics and variation scores.

### 4. Diagnostics
Lists of specific issues found in the text, organized by category. Each issue typically includes the offending text, the issue type, and the surrounding context.
- `passive_voice`
- `pro_editing` (cliches, weak verbs)
- `advanced_features` (lexical diversity)
- `structure`:
  - `run_ons`
  - `clipped_sentences`
  - `sentence_splices` (fused sentences: a missing period before a capitalized word)
  - `comma_splices` (two independent clauses joined by a bare comma)
  - `dangling_participles`
  - `tense_consistency`
- `creative`:
  - `nominalizations`
  - `dialogue_beats`
  - `redundancy`
  - `show_dont_tell`
  - `adverbial_tags`

### 5. Suppressions (Optional)
If `--show-suppressions` is used, this array contains the list of suppressed issues parsed from inline comments like `<!-- lint-ignore -->`.
- `marker` (string): The suppression comment.
- `start` (int): Start offset.
- `end` (int): End offset.
