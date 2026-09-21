import re
import sys
import io
import contextlib
import bisect

# Import our NLP refiner
# Wrap in try/except to handle cases where dependencies might be missing if run elsewhere
try:
    from text_refiner import (
        get_spacy_doc,
        refine_creative_text,
        detect_sentence_splices,
        detect_comma_splices,
        get_passive_voice_report, 
        get_tone_profile,
        get_advanced_prose_report,
        get_pro_editing_report,
        detect_dangling_participles,
        get_tense_consistency_report,
        get_nominalization_report,
        get_show_dont_tell_report,
        get_adverbial_dialogue_tags,
        get_paragraph_rhythm,
        get_all_paragraph_rhythm_data,
        get_specificity_ratio,
        get_vocabulary_metrics
    )
    NLP_AVAILABLE = True
except ImportError as e:
    NLP_AVAILABLE = False

try:
    from spellchecker import SpellChecker
    SPELLING_AVAILABLE = True
except ImportError:
    # pyspellchecker is a declared dependency; if it is genuinely missing we
    # disable spell checking rather than emit garbage (the old SpaCy token.prob
    # heuristic flagged every word, since sm/md models ship no word frequencies).
    SPELLING_AVAILABLE = False
    SpellChecker = None

def calculate_reading_time(text, word_count=None):
    """Calculates reading time string for a given text."""
    if word_count is None:
        word_count = len(text.split())
    
    if word_count == 0:
        return "0s"
        
    # Reading Time (Standard 225 WPM)
    reading_time_mins = word_count / 225
    if reading_time_mins >= 1:
        return f"{int(reading_time_mins)}m {int((reading_time_mins % 1) * 60)}s"
    else:
        return f"{max(1, int(reading_time_mins * 60))}s"

_DIALOGUE_PATTERNS = [
    r'"[^"]*"',                # straight double quotes
    r'[“][^”]*[”]',     # smart double quotes
    r'[‘][^’]*[’]',     # smart single quotes
]

def calculate_dialogue_ratio(text):
    """Calculates the percentage of text that is dialogue.

    Each quote type is matched against its correct closing partner so that
    apostrophes don't act as delimiters. Straight single quotes (') are
    deliberately excluded because they are indistinguishable from apostrophes
    (it's, the cat's), which previously caused dialogue to be over-counted.
    """
    if not text: return 0
    dialogue_chars = 0
    for pattern in _DIALOGUE_PATTERNS:
        for match in re.findall(pattern, text, re.DOTALL):
            dialogue_chars += len(match)
    return (dialogue_chars / len(text) * 100)

def get_sentences(text):
    """
    Splits text into a list of cleaned sentences using punctuation boundaries.
    """
    # Simple split on . ! ? followed by space or end of string
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw_sentences if s.strip()]

def count_syllables(word):
    """
    Heuristic to count syllables in an English word.
    """
    word = word.lower()
    if not word: return 0
    # Boundary cases
    if len(word) <= 3: return 1
    
    # Remove final 'e' if not 'le'
    if word.endswith('e') and not word.endswith('le'):
        word = word[:-1]
    
    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    
    return max(1, count)

def calculate_readability(text, sentences):
    """
    Calculates Flesch Reading Ease score and provides suggestions.
    Returns a dict.
    """
    # Clean text to get words
    clean_text = re.sub(r'[^\w\s]', '', text)
    words = [w for w in clean_text.split() if w]
    
    num_words = len(words)
    num_sentences = len(sentences)
    if num_words == 0 or num_sentences == 0:
        return None

    num_syllables = sum(count_syllables(w) for w in words)
    
    # Flesch Reading Ease Formula
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    
    interpretation = ""
    if score >= 90: interpretation = "Very Easy (5th Grade)"
    elif score >= 80: interpretation = "Easy (6th Grade)"
    elif score >= 70: interpretation = "Fairly Easy (7th Grade)"
    elif score >= 60: interpretation = "Standard (8th-9th Grade)"
    elif score >= 50: interpretation = "Fairly Difficult (High School)"
    elif score >= 30: interpretation = "Difficult (College)"
    else: interpretation = "Very Confusing (College Grad)"

    suggestions = []
    avg_sentence_len = num_words / num_sentences
    if avg_sentence_len > 22:
        suggestions.append(f"Your average sentence length is {avg_sentence_len:.1f} words. Try breaking long sentences into smaller ones.")
        
    if score < 60:
        suggestions.append("The language is complex. Consider using simpler words or shorter independent clauses.")
        
    if not suggestions:
        suggestions.append("Your writing is well-balanced for the targeted complexity.")

    # Flesch-Kincaid Grade Level
    grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59

    return {
        "score": score,
        "grade_level": max(0, round(grade, 2)),
        "interpretation": interpretation,
        "suggestions": suggestions,
        "metrics": {
            "word_count": num_words,
            "sentence_count": num_sentences,
            "syllable_count": num_syllables
        }
    }

def calculate_prose_score(report):
    """
    Calculates a 0-100 prose quality score based on detected issues.
    """
    score = 100
    
    # 1. Passive Voice (Penalty if > 10%)
    passive_perc = report.get("executive_summary", {}).get("passive_perc", 0)
    if passive_perc > 10:
        score -= min(20, (passive_perc - 10) * 2)
        
    # 2. Clichés
    adv = report.get("advanced_features") or {}
    cliches = adv.get("cliches", []) or []
    if cliches:
        score -= min(15, len(cliches) * 3)
        
    # 3. Filter Verbs
    pro_ed = report.get("pro_editing") or {}
    filters = pro_ed.get("filter_verbs", []) or []
    if filters:
        score -= min(15, len(filters) * 2)
        
    # 4. Redundancy
    redundant = report.get("redundancy", []) or []
    if redundant:
        score -= min(10, len(redundant) * 1)
        
    # 5. Dialogue Punctuation Errors
    dp_errs = report.get("dialogue_punctuation_errors", []) or []
    if dp_errs:
        score -= min(10, len(dp_errs) * 2)
        
    # 6. Spelling
    spelling = report.get("spelling", []) or []
    if spelling:
        score -= min(20, len(spelling) * 4)

    # 7. Comma Splices
    comma_splices = report.get("comma_splices", []) or []
    if comma_splices:
        score -= min(10, len(comma_splices) * 2)

    return max(0, int(score))

def calculate_drone_factor(sentences):
    """
    Calculates 0-10 score (0=Robotic, 10=Masterful) based on sentence length variety.
    Formula: Uses Coefficient of Variation (StdDev / Mean) of sentence lengths.
    """
    if not sentences:
        return 5.0
        
    word_counts = [len(s.split()) for s in sentences if s.strip()]
    if not word_counts:
        return 5.0
        
    mean = sum(word_counts) / len(word_counts)
    if mean == 0:
        return 0.0
        
    variance = sum((x - mean) ** 2 for x in word_counts) / len(word_counts)
    std_dev = variance ** 0.5
    
    # Coefficient of Variation (CV)
    cv = std_dev / mean
    
    # Heuristic: CV of 0.8+ is generally good/vibrant. CV < 0.3 is "drone-like".
    # Map 0.2 -> 0, 0.9 -> 10
    factor = (cv - 0.2) / 0.7 * 10
    return max(0.0, min(10.0, round(factor, 1)))

def get_sentence_starts(doc):
    """Returns a list of token indices where sentences start."""
    return [sent.start for sent in doc.sents]

def get_sentence_starts_char(doc):
    """Returns a list of character offsets where sentences start."""
    return [sent.start_char for sent in doc.sents]

def find_sentence_id_fast(token_idx, sentence_starts):
    """Finds 1-based sentence ID for a given token index using bisect."""
    return bisect.bisect_right(sentence_starts, token_idx)

def find_sentence_id_fast_char(char_idx, sentence_starts_char):
    """Finds 1-based sentence ID for a given character offset using bisect."""
    return bisect.bisect_right(sentence_starts_char, char_idx)

def find_sentence_id(snippet, sentences):
    """
    Deprecated/Fallback: Finds the index (1-based) of the sentence containing the snippet.
    """
    snippet = snippet.strip()
    if not snippet: return "?"
    
    # Try exact match first
    for i, sent in enumerate(sentences):
        if sent == snippet:
            return i + 1
            
    # Try substring match
    for i, sent in enumerate(sentences):
        if snippet in sent:
            return i + 1
            
    return "?"

def get_passive_voice_analysis(doc, sentence_starts, expected_count=None):
    """
    Identifies passive voice usage.
    """
    if not NLP_AVAILABLE:
        return None
        
    passive_data = get_passive_voice_report(doc, expected_count=expected_count)
    percentage = passive_data['passive_percentage']
    examples = passive_data['examples']
    
    result = {
        "passive_percentage": percentage,
        "examples": []
    }
    
    if examples:
        for ex in examples:
            sid = find_sentence_id_fast(ex.get('token_idx', 0), sentence_starts)
            result["examples"].append({
                "sentence_id": sid, 
                "text": ex.get('text', ''),
                "start": ex.get('start'),
                "end": ex.get('end')
            })
            
    result["total_sentences"] = passive_data.get('total_sentences', 0)
    result["passive_count"] = len(examples)
    return result

def get_tone_analysis(doc):
    """
    Analyzes the consistency and nature of the writing tone.
    """
    if not NLP_AVAILABLE:
        return None
        
    profile = get_tone_profile(doc)
    if not profile:
        return None
        
    return profile

def get_advanced_features_analysis(doc, sentence_starts):
    """
    Reports Adverbs, Echoes, Sentence Distribution, and Clichés.
    """
    if not NLP_AVAILABLE:
        return None
        
    report = get_advanced_prose_report(doc)
    
    adverb_sentences = []
    if report['adverb_heavy_sentences']:
        for item in report['adverb_heavy_sentences']:
            sent = item['text']
            sid = find_sentence_id_fast(item.get('token_idx', 0), sentence_starts)
            adverb_sentences.append({
                "sentence_id": sid, 
                "text": sent, 
                "adverbs": item['adverbs'],
                "start": item.get('start'),
                "end": item.get('end')
            })
            
    weak_dialogue_tags = []
    if report['weak_dialogue_tags']:
        for tag_item in report['weak_dialogue_tags']:
            sid = find_sentence_id_fast(tag_item.get('token_idx', 0), sentence_starts)
            weak_dialogue_tags.append({
                "sentence_id": sid, 
                "text": tag_item['text'],
                "start": tag_item.get('start'),
                "end": tag_item.get('end')
            })
            
    cliches = []
    if report['cliches']:
        for c in report['cliches']:
            sid = find_sentence_id_fast(c.get('token_idx', 0), sentence_starts)
            cliches.append({
                "sentence_id": sid, 
                "text": c['text'],
                "start": c.get('start'),
                "end": c.get('end')
            })

    return {
        "adverb_heavy_sentences": adverb_sentences,
        "word_echoes": report['word_echoes'],
        "sentence_types": report['sentence_types'],
        "weak_dialogue_tags": weak_dialogue_tags,
        "cliches": cliches
    }

def get_pro_editing_analysis(doc, sentence_starts):
    """
    Reports Filter Verbs, Sensory Balance, Pacing (Dialogue Ratio), and Vague Words.
    """
    if not NLP_AVAILABLE:
        return None
        
    report = get_pro_editing_report(doc)
    
    filter_verbs = []
    if report['filter_verbs']:
        for fv_item in report['filter_verbs']:
            sid = find_sentence_id_fast(fv_item.get('token_idx', 0), sentence_starts)
            filter_verbs.append({
                "sentence_id": sid, 
                "text": fv_item['text'],
                "start": fv_item.get('start'),
                "end": fv_item.get('end')
            })
            
    vague_words = []
    if report['vague_words']:
        for vw_item in report['vague_words']:
            sid = find_sentence_id_fast(vw_item.get('token_idx', 0), sentence_starts)
            vague_words.append({
                "sentence_id": sid, 
                "text": vw_item['text'],
                "start": vw_item.get('start'),
                "end": vw_item.get('end')
            })

    ratio = report['dialogue_ratio']
    interpretation = "Narrative Heavy" if ratio < 20 else "Fast-Paced / Balanced"
    if ratio > 60: interpretation = "Dialogue Heavy (Rushed?)"

    return {
        "filter_verbs": filter_verbs,
        "sensory_balance": report['sensory_balance'],
        "dialogue_ratio": ratio,
        "dialogue_interpretation": interpretation,
        "vague_words": vague_words
    }

def get_sentence_splices_analysis(doc):
    """
    Reports potential sentence splices detected by NLP.
    """
    if not NLP_AVAILABLE:
        return None

    splices = detect_sentence_splices(doc)
    return splices

def get_comma_splices_analysis(doc, sentence_starts):
    """
    Reports comma splices: two independent clauses joined by a bare comma.
    """
    if not NLP_AVAILABLE:
        return None

    result = []
    for item in detect_comma_splices(doc):
        result.append({
            "sentence_id": find_sentence_id_fast(item["token_idx"], sentence_starts),
            "text": item["text"],
            "suggestion": "Split into two sentences, or join with a semicolon "
                          "or a comma plus a conjunction.",
            "start": item["start"],
            "end": item["end"],
        })
    return result

def get_dangling_participles_analysis(doc, sentence_starts):
    if not NLP_AVAILABLE: return None
    dangers = detect_dangling_participles(doc)
    result = []
    if dangers:
        for item in dangers:
            sid = find_sentence_id_fast(item.get('token_idx', 0), sentence_starts)
            result.append({
                "sentence_id": sid,
                "clause": item['clause'],
                "subject": item.get('main_subject'),
                "sentence": item['sentence'],
                "start": item.get('start'),
                "end": item.get('end')
            })
    return result

def get_tense_consistency_analysis(doc):
    if not NLP_AVAILABLE: return None
    return get_tense_consistency_report(doc)

def get_nominalizations_analysis(doc, sentence_starts):
    if not NLP_AVAILABLE: return None
    zombies = get_nominalization_report(doc)
    result = []
    if zombies:
        for z_item in zombies[:10]:
            sid = find_sentence_id_fast(z_item.get('token_idx', 0), sentence_starts)
            result.append({
                "sentence_id": sid, 
                "text": z_item.get('text', ''),
                "start": z_item.get('start'),
                "end": z_item.get('end')
            })
    return result


def get_dialogue_beats_analysis(text, sentences, sentence_starts_char=None):
    """
    Checks for: "Stop," He said. (Should be "Stop." He said.)
    Where the tag is NOT a dialogue tag but an action beat.
    """
    speech_verbs = {"said", "asked", "replied", "whispered", "shouted", "muttered", "screamed", "yelled", "told", "answered"}
    
    matches = re.finditer(r'(?:^|\s)(["\'][^"\']*?,\s*["\']?)\s*(He|She|They|It|We|I|You)\s+([a-z]+(?:\'[a-z]+)?)', text)
    errors = []
    for m in matches:
        verb = m.group(3).lower()
        if verb not in speech_verbs:
            snippet = m.group(0).strip()
            if sentence_starts_char:
                sid = find_sentence_id_fast_char(m.start(), sentence_starts_char)
            else:
                sid = find_sentence_id(snippet, sentences)
            errors.append({
                "sentence_id": sid, 
                "text": snippet, 
                "verb": verb,
                "start": m.start(),
                "end": m.end()
            })
            
    return errors

def get_redundancy_analysis(text, sentences, sentence_starts_char=None):
    """
    Checks for common pleonasms.
    """
    redundancies = {
        "nodded his head": "nodded", "nodded her head": "nodded",
        "shrugged his shoulders": "shrugged", "shrugged her shoulders": "shrugged",
        "small in size": "small", "large in size": "large",
        "true facts": "facts", "added bonus": "bonus", "free gift": "gift"
    }
    
    found = []
    for phrase, correction in redundancies.items():
        # Using re.finditer to get offsets for fast lookups
        matches = re.finditer(re.escape(phrase), text, re.IGNORECASE)
        for m in matches:
            if sentence_starts_char:
                sid = find_sentence_id_fast_char(m.start(), sentence_starts_char)
            else:
                sid = find_sentence_id(m.group(0), sentences)
            found.append({
                "sentence_id": sid, 
                "phrase": phrase, 
                "correction": correction,
                "start": m.start(),
                "end": m.end()
            })
    return found

def get_possessive_errors_analysis(text, doc, sentence_starts):
    """
    Detects "it's" used where the possessive "its" is intended
    (e.g. "it's tail was wagging" -> "its tail").

    Uses POS tags: the contraction "it's" tokenizes as [it, 's]. It is almost
    certainly a possessive error when the "'s" is immediately followed by a
    NOUN/PROPN with no intervening determiner ("its color"), and is correct
    when followed by a verb/adjective/adverb/determiner ("it's gone",
    "it's cold", "it's a problem"). Falls back to no detections without NLP,
    since a reliable check requires POS information.
    """
    if not NLP_AVAILABLE or doc is None:
        return []

    errors = []
    for tok in doc:
        if tok.lower_ != "'s":
            continue
        prev = doc[tok.i - 1] if tok.i > 0 else None
        if prev is None or prev.lower_ != "it":
            continue
        nxt = doc[tok.i + 1] if tok.i + 1 < len(doc) else None
        if nxt is None or nxt.pos_ not in ("NOUN", "PROPN"):
            continue

        start = prev.idx
        end = nxt.idx + len(nxt.text)
        sid = find_sentence_id_fast(prev.i, sentence_starts)
        errors.append({
            "sentence_id": sid,
            "text": text[start:end],
            "suggestion": "its " + nxt.text,
            "start": start,
            "end": end
        })
    return errors


def get_banned_phrases_analysis(text, spec=None):
    """
    Checks for phrases that the user explicitly wants to avoid.
    """
    banned = []
    if spec and spec.get("banned_phrases"):
        for phrase in spec["banned_phrases"]:
            # Case insensitive search with word boundaries
            pattern = r'\b' + re.escape(phrase) + r'\b'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                banned.append({
                    "phrase": phrase,
                    "text": m.group(0),
                    "start": m.start(),
                    "end": m.end()
                })
    return banned


def get_spelling_analysis(text: str, known_words: list[str] | None = None, doc=None) -> list[dict]:
    """Identify potential typos and suggestions using pyspellchecker.

    When a parsed ``doc`` is supplied, candidate words are taken from its tokens
    so proper nouns (character names, places) can be skipped — this avoids
    flagging invented names in creative writing. Without a doc, words are
    extracted with a regex. All-caps acronyms are ignored.
    """
    if not SPELLING_AVAILABLE:
        return []

    spell = SpellChecker()
    # pyspellchecker stores its dictionary in lowercase; mirror that for known words.
    if known_words:
        spell.word_frequency.load_words([w.lower() for w in known_words])
    spell.word_frequency.load_words(['ai', 'noir', 'cyberpunk', 'holo', 'vid', 'neuro'])

    # Collect (word, start, end) candidates.
    candidates = []
    if doc is not None:
        for token in doc:
            # spaCy splits contractions ("don't" -> "do", "n't"); the "n't"
            # piece is not alpha and is skipped here.
            if not token.is_alpha or len(token.text) < 2:
                continue
            if token.pos_ == "PROPN":  # skip names / places to avoid false positives
                continue
            candidates.append((token.text, token.idx, token.idx + len(token.text)))
    else:
        for m in re.finditer(r"\b[A-Za-z]+(?:'[a-z]+)?\b", text):
            w = m.group(0)
            if len(w) > 1:
                candidates.append((w, m.start(), m.end()))

    # unknown() lowercases internally, so compare on the lowercased form and
    # report the original spelling. (The old code compared original-case words
    # against a lowercased set, silently missing every capitalized typo.)
    lowered = [w.lower() for (w, _, _) in candidates]
    unknown = spell.unknown(lowered)

    found = []
    for (word, start, end), lower in zip(candidates, lowered):
        if word.isupper() and len(word) > 1:
            continue  # acronym (NASA, AI, ...)
        if lower in unknown:
            found.append({
                "word": word,
                # candidates() returns None (not an empty set) when it has
                # no suggestions — e.g. invented words.
                "suggestions": list(spell.candidates(lower) or []) if len(word) < 15 else [],
                "start": start,
                "end": end
            })
    return found

def get_show_dont_tell_analysis(doc, sentence_starts):
    if not NLP_AVAILABLE: return None
    items = get_show_dont_tell_report(doc)
    result = []
    if items:
        for item in items[:10]:
            sid = find_sentence_id_fast(item.get('token_idx', 0), sentence_starts)
            result.append({
                "sentence_id": sid,
                "phrase": item.get('phrase', ''),
                "subject": item.get('subject', 'Someone'),
                "emotion": item.get('emotion', 'N/A'),
                "type": item.get('type', 'telling'),
                "start": item.get('start'),
                "end": item.get('end')
            })
    return result

def get_paragraph_rhythm_analysis(doc):
    if not NLP_AVAILABLE: return None
    return get_paragraph_rhythm(doc)

def get_specificity_analysis(doc):
    if not NLP_AVAILABLE: return None
    data = get_specificity_ratio(doc)
    return data

def get_adverbial_tags_analysis(doc, sentence_starts):
    if not NLP_AVAILABLE: return None
    tags = get_adverbial_dialogue_tags(doc)
    result = []
    if tags:
        for t in tags[:5]:
            sid = find_sentence_id_fast(t.get('token_idx', 0), sentence_starts)
            result.append({
                "sentence_id": sid, 
                "text": t['text'],
                "start": t.get('start'),
                "end": t.get('end')
            })
    return result

def extract_and_blank_suppressions(text):
    """
    Parses suppression markers and replaces them with spaces to preserve offsets.
    Returns (blanked_text, suppressions_list)
    """
    suppressions = []
    active_blocks = {}
    file_level_suppressions = set()
    
    pattern = re.compile(r'<!--\s*lint-(disable|enable|disable-next-line|disable-file)(?:\s+([a-zA-Z0-9_-]+))?\s*-->', re.IGNORECASE)
    
    blanked_text = list(text)
    
    for m in pattern.finditer(text):
        # Blank it out
        for i in range(m.start(), m.end()):
            if blanked_text[i] not in ('\r', '\n'):
                blanked_text[i] = ' '
            
        action = m.group(1).lower()
        issue = m.group(2).lower() if m.group(2) else "all"
        
        if action == "disable-file":
            file_level_suppressions.add(issue)
        elif action == "disable-next-line":
            next_line_start = m.end()
            while next_line_start < len(text) and text[next_line_start] in ('\r', '\n', ' ', '\t'):
                next_line_start += 1
            next_line_end = text.find('\n', next_line_start)
            if next_line_end == -1:
                next_line_end = len(text)
            suppressions.append({"start": next_line_start, "end": next_line_end, "issue": issue})
        elif action == "disable":
            if issue not in active_blocks:
                active_blocks[issue] = m.end()
        elif action == "enable":
            if issue in active_blocks:
                suppressions.append({"start": active_blocks[issue], "end": m.start(), "issue": issue})
                del active_blocks[issue]
            elif issue == "all":
                for k, start_pos in list(active_blocks.items()):
                    suppressions.append({"start": start_pos, "end": m.start(), "issue": k})
                active_blocks.clear()

    for issue, start_pos in active_blocks.items():
        suppressions.append({"start": start_pos, "end": len(text), "issue": issue})
        
    for issue in file_level_suppressions:
        suppressions.append({"start": 0, "end": len(text), "issue": issue})
        
    return "".join(blanked_text), suppressions

def filter_suppressed_issues(report, suppressions):
    """
    Filters issues from the report based on suppression regions.
    Returns (filtered_report, suppression_count)
    """
    if not suppressions:
        return report, 0
        
    suppression_count = 0
    
    def is_suppressed(start, end, issue_type):
        if start is None or end is None:
            return False
        for sup in suppressions:
            if sup["issue"] == "all" or sup["issue"] == issue_type:
                if start < sup["end"] and end > sup["start"]:
                    return True
        return False
        
    # special handling for grouped items
    if "advanced_features" in report and isinstance(report["advanced_features"], dict):
        adv = report["advanced_features"]
        for k in ["adverb_heavy_sentences", "weak_dialogue_tags", "cliches"]:
            if k in adv and adv[k]:
                new_list = []
                for item in adv[k]:
                    if not is_suppressed(item.get("start"), item.get("end"), k):
                        new_list.append(item)
                    else:
                        suppression_count += 1
                adv[k] = new_list
                
    if "pro_editing" in report and isinstance(report["pro_editing"], dict):
        pe = report["pro_editing"]
        for k in ["filter_verbs", "vague_words"]:
            if k in pe and pe[k]:
                new_list = []
                for item in pe[k]:
                    if not is_suppressed(item.get("start"), item.get("end"), k):
                        new_list.append(item)
                    else:
                        suppression_count += 1
                pe[k] = new_list
                
    # Flat lists in report
    for k in ["dialogue_punctuation_errors", "possessive_errors", "spelling", 
              "sentence_splices", "dangling_participles", "nominalizations", 
              "dialogue_beats", "redundancy", "show_dont_tell", "adverbial_tags"]:
        if k in report and isinstance(report[k], list):
            new_list = []
            for item in report[k]:
                if isinstance(item, dict):
                    start = item.get("start")
                    end = item.get("end")
                    if not is_suppressed(start, end, k):
                        new_list.append(item)
                    else:
                        suppression_count += 1
                else:
                    new_list.append(item)
            report[k] = new_list

    return report, suppression_count

def filter_by_config(report, config):
    if not config: return report
    
    # 1. Ignore Rules
    for rule in config.get("ignore_rules", []):
        if rule in report:
            del report[rule]
            
    # 2. Accepted Patterns
    patterns = config.get("accepted_patterns", [])
    if patterns:
        compiled_patterns = [re.compile(p) for p in patterns]
        
        def matches_pattern(text_val):
            if not text_val: return False
            for p in compiled_patterns:
                if p.search(text_val): return True
            return False
            
        for k, v in report.items():
            if isinstance(v, list):
                new_list = []
                for item in v:
                    if isinstance(item, dict):
                        text_val = item.get("text", item.get("word", item.get("phrase", "")))
                        if not matches_pattern(text_val):
                            new_list.append(item)
                    else:
                        new_list.append(item)
                report[k] = new_list
                
            elif isinstance(v, dict):
                # E.g. pro_editing, advanced_features
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, list):
                        new_list = []
                        for item in sub_v:
                            if isinstance(item, dict):
                                text_val = item.get("text", item.get("word", item.get("phrase", "")))
                                if not matches_pattern(text_val):
                                    new_list.append(item)
                            else:
                                new_list.append(item)
                        v[sub_k] = new_list
    
    return report

def lint_prose(text, sentences=None, doc=None, show_suppressions=False, config=None):
    """
    Runs prose analysis on the given text.
    If 'doc' is provided, it uses it instead of re-parsing.
    If sentences are not provided, they will be generated from the NLP doc.
    Returns a dictionary report.
    """
    report = {}

    # Process suppressions before anything else
    blanked_text, suppressions = extract_and_blank_suppressions(text)
    
    sentence_starts_char = []
    if NLP_AVAILABLE:
        if doc is None:
            doc = get_spacy_doc(blanked_text)
        if sentences is None:
            sentences = [sent.text.strip() for sent in doc.sents]
        sentence_starts = get_sentence_starts(doc)
        sentence_starts_char = get_sentence_starts_char(doc)
    else:
        doc = None
        if sentences is None:
            sentences = get_sentences(blanked_text)
        sentence_starts = []
        sentence_starts_char = []

    # Pre-calculate detailed reports to feed into the executive summary
    report["readability"] = calculate_readability(blanked_text, sentences)
    
    # 1. PASSIVE VOICE
    total_sent_count = len(sentences)
    report["passive_voice"] = get_passive_voice_analysis(doc, sentence_starts, expected_count=total_sent_count)
    
    # 2. TONE & PRO-EDITING
    report["tone"] = get_tone_analysis(doc)
    report["pro_editing"] = get_pro_editing_analysis(doc, sentence_starts)
    report["advanced_features"] = get_advanced_features_analysis(doc, sentence_starts)
    report["vocabulary"] = get_vocabulary_metrics(doc) if NLP_AVAILABLE else None
    
    # 3. CONSOLIDATED SENTENCE LOOP (I-clusters, Run-ons, Clipped, Repetition)
    i_streak = 0
    i_clusters = []
    run_ons = []
    clipped = []
    
    sentence_data = [] # List of (text, start, end)
    if NLP_AVAILABLE:
        for sent in doc.sents:
            sentence_data.append((sent.text.strip(), sent.start_char, sent.end_char))
    else:
        # Fallback to regex split but try to maintain offsets
        pos = 0
        for sent_text in sentences:
            start = text.find(sent_text, pos)
            if start != -1:
                end = start + len(sent_text)
                sentence_data.append((sent_text, start, end))
                pos = end
            else:
                sentence_data.append((sent_text, 0, 0))

    for i, (sent_text, start, end) in enumerate(sentence_data):
        # I-Start Clusters
        if sent_text.startswith("I ") or sent_text == "I":
            i_streak += 1
        else:
            if i_streak >= 3:
                # Store start/end of the cluster
                cluster_start = sentence_data[i - i_streak][1]
                cluster_end = sentence_data[i - 1][2]
                i_clusters.append({"start_index": i - i_streak, "count": i_streak, "start": cluster_start, "end": cluster_end})
            i_streak = 0
            
        # Run-on Sentences (>30 words)
        words = sent_text.split()
        if len(words) > 30:
            run_ons.append({"sentence_id": i + 1, "text": sent_text, "word_count": len(words), "start": start, "end": end})
            
        # Clipped Sentences (<5 words)
        if 0 < len(words) < 5:
            clipped.append({"sentence_id": i + 1, "text": sent_text, "word_count": len(words), "start": start, "end": end})

    # Catch trailing I-streak
    if i_streak >= 3:
        cluster_start = sentence_data[len(sentence_data) - i_streak][1]
        cluster_end = sentence_data[len(sentence_data) - 1][2]
        i_clusters.append({"start_index": len(sentence_data) - i_streak, "count": i_streak, "start": cluster_start, "end": cluster_end})

    # Perform clustering for clipped sentences (3 or more short sentences in a row)
    clipped_clusters = []
    current_cluster = []
    for item in clipped:
        if not current_cluster or item['sentence_id'] == current_cluster[-1]['sentence_id'] + 1:
            current_cluster.append(item)
        else:
            if len(current_cluster) >= 3:
                clipped_clusters.append(current_cluster)
            current_cluster = [item]
    if len(current_cluster) >= 3:
        clipped_clusters.append(current_cluster)
        
    report["repetitive_structure"] = [
        {"start_index": item["start_index"], "count": item["count"], "sample": sentence_data[item["start_index"]][0], "start": item["start"], "end": item["end"]}
        for item in i_clusters
    ]
    report["run_ons"] = run_ons
    report["clipped_sentences"] = clipped_clusters

    # 4. REGEX PASSES (Enhanced with offsets)
    dp_errs = []
    for m in re.finditer(r'["\'“”‘’].*?\.(?=["\'“”‘’]\s+[a-z])', text):
        dp_errs.append({"text": m.group(0), "start": m.start(), "end": m.end()})
    report["dialogue_punctuation_errors"] = dp_errs

    report["possessive_errors"] = get_possessive_errors_analysis(text, doc, sentence_starts)
    
    # Spelling
    report["spelling"] = get_spelling_analysis(text, doc=doc)

    # 5. NLP-DEPENDENT FEATURES
    if NLP_AVAILABLE:
        report["sentence_splices"] = get_sentence_splices_analysis(doc)
        report["comma_splices"] = get_comma_splices_analysis(doc, sentence_starts)
        report["dangling_participles"] = get_dangling_participles_analysis(doc, sentence_starts)
        report["tense_consistency"] = get_tense_consistency_analysis(doc)
        report["nominalizations"] = get_nominalizations_analysis(doc, sentence_starts)
        report["dialogue_beats"] = get_dialogue_beats_analysis(text, sentences, sentence_starts_char)
        report["redundancy"] = get_redundancy_analysis(text, sentences, sentence_starts_char)
        report["show_dont_tell"] = get_show_dont_tell_analysis(doc, sentence_starts)
        report["paragraph_rhythm"] = get_paragraph_rhythm(doc)
        report["rhythm_heatmap"] = get_all_paragraph_rhythm_data(doc)
        report["specificity"] = get_specificity_analysis(doc)
        report["adverbial_tags"] = get_adverbial_tags_analysis(doc, sentence_starts)
        
        try:
            from tts_analyzer import run_tts_analysis
            report["tts_analysis"] = run_tts_analysis(text)
        except ImportError:
            report["tts_analysis"] = None
    else:
        for key in ["sentence_splices", "comma_splices", "dangling_participles",
                   "tense_consistency", "nominalizations", "dialogue_beats",
                   "redundancy", "show_dont_tell", "paragraph_rhythm",
                   "rhythm_heatmap", "specificity", "adverbial_tags", "tts_analysis"]:
            report[key] = None

    # 6. EXECUTIVE SUMMARY (CONSUME WORK)
    # Extract data from pre-calculated reports to avoid redundant NLP passes
    word_count = len(text.split())
    # Safe division for average sentence length
    num_sents = len(sentences) if sentences else 0
    avg_sent_len = word_count / num_sents if num_sents > 0 else 0
    
    # Safe access to sub-dictionaries
    pv = report.get("passive_voice")
    passive_perc = pv.get("passive_percentage", 0.0) if isinstance(pv, dict) else 0.0
    
    if config:
        report = filter_by_config(report, config)
        
    # Execute suppression filtering before summary
    report, suppression_count = filter_suppressed_issues(report, suppressions)
    
    # Count total issues
    total_issues = 0
    issue_keys = [
        "passive_voice", "run_ons", "clipped_sentences", "sentence_splices",
        "comma_splices", "dangling_participles", "tense_consistency",
        "nominalizations", "dialogue_beats", "redundancy", "show_dont_tell",
        "adverbial_tags", "dialogue_punctuation_errors", "possessive_errors"
    ]
    for key in issue_keys:
        issues = report.get(key)
        if isinstance(issues, list):
            total_issues += len(issues)
    
    report["executive_summary"] = {
        "word_count": word_count,
        "avg_sent_len": avg_sent_len,
        "reading_time": calculate_reading_time(blanked_text, word_count=word_count),
        "dialogue_ratio": calculate_dialogue_ratio(blanked_text),
        "passive_perc": passive_perc,
        "specificity": report.get("specificity"),
        "tone": report.get("tone"),
        "vocabulary": report.get("vocabulary"),
        "suppressions_count": suppression_count,
        "total_issues": total_issues
    }
    
    if show_suppressions:
        report["suppressions"] = suppressions
    
    # Calculate Prose Score and Drone Factor
    report["executive_summary"]["prose_score"] = calculate_prose_score(report)
    report["executive_summary"]["drone_factor"] = calculate_drone_factor(sentences)

    return report

def incremental_lint(old_text, new_text, old_report, doc=None, show_suppressions=False, config=None):
    """
    Returns the cached ``old_report`` when the text is unchanged; otherwise
    performs a full re-lint of ``new_text``.

    A full re-lint is required on any change because character offsets and
    context-dependent checks (paragraph rhythm, tense consistency, word echoes)
    can span paragraph boundaries. True paragraph-level incremental linting
    would need dynamic offset shifting for unchanged blocks and is not yet
    implemented.
    """
    if old_text == new_text:
        return old_report

    return lint_prose(new_text, doc=doc, show_suppressions=show_suppressions, config=config)

# --- Mechanical fix tables ------------------------------------------------ #
# Missing-apostrophe contractions. Only forms that are not themselves real
# words are listed, with two documented exceptions: "cant" (jargon) and "wont"
# (habit) are archaic/rare enough that treating them as typos is the right
# call for this tool's audience.
_CONTRACTION_MAP = {
    "dont": "don't", "didnt": "didn't", "doesnt": "doesn't", "isnt": "isn't",
    "wasnt": "wasn't", "werent": "weren't", "arent": "aren't",
    "couldnt": "couldn't", "shouldnt": "shouldn't", "wouldnt": "wouldn't",
    "mustnt": "mustn't", "havent": "haven't", "hasnt": "hasn't",
    "hadnt": "hadn't", "cant": "can't", "wont": "won't",
    "theyre": "they're", "youre": "you're", "thats": "that's",
    "whats": "what's", "whos": "who's", "hes": "he's", "shes": "she's",
    "ive": "I've", "youve": "you've", "weve": "we've", "theyve": "they've",
    "youll": "you'll", "theyll": "they'll", "youd": "you'd", "theyd": "they'd",
    "couldve": "could've", "shouldve": "should've", "wouldve": "would've",
    "mustve": "must've", "mightve": "might've",
    "im": "I'm",
}
_CONTRACTION_RE = re.compile(r"\b(" + "|".join(_CONTRACTION_MAP) + r")\b", re.IGNORECASE)

_MODAL_OF_RE = re.compile(r"\b(could|should|would|must|might|may)(\s+)of\b", re.IGNORECASE)
_THEIR_BE_RE = re.compile(r"\b(their)(\s+)(is|are|was|were)\b", re.IGNORECASE)
# Narrow, high-precision your->you're contexts only ("your running shoes" is
# why a general your+gerund rule is unsafe without parsing).
_YOUR_RE = re.compile(r"\b(your)(\s+)(gonna|going\s+to|kidding|joking|not)\b", re.IGNORECASE)
# Doubled function words. Deliberately excludes words with legitimate doubles:
# "had had", "that that", "in in" ("turned it in in March"), "on on".
_DOUBLED_RE = re.compile(r"\b(the|a|an|and|of|for|with|from)\s+\1\b", re.IGNORECASE)
# Standalone pronoun "i" (also catches i'm / i've via the boundary before the
# apostrophe); the lookahead protects "i.e.".
_LONE_I_RE = re.compile(r"\bi\b(?!\.\w)")

# Vowel-letter words pronounced with a consonant sound take "a".
_LONG_U_PREFIXES = ("uni", "use", "usu", "ute", "uto", "ubi", "eu", "one", "once")
# Silent-h words take "an".
_SILENT_H_PREFIXES = ("hour", "honest", "honor", "honour", "heir")

def _wants_an(word):
    wl = word.lower()
    if wl.startswith(_SILENT_H_PREFIXES):
        return True
    if wl[0] not in "aeiou":
        return False
    if wl.startswith(_LONG_U_PREFIXES):
        return False
    return True

def _match_case(replacement, original):
    return replacement.capitalize() if original[0].isupper() else replacement

def fix_mechanical_errors(text):
    """
    Deterministic regex layer: fixes errors that are unambiguous without
    parsing — missing-apostrophe contractions, modal + "of", "their is",
    doubled function words, a/an agreement, and the lone pronoun "i".
    """
    text = _CONTRACTION_RE.sub(
        lambda m: _match_case(_CONTRACTION_MAP[m.group(1).lower()], m.group(1)), text)
    text = _MODAL_OF_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}have", text)
    text = _THEIR_BE_RE.sub(
        lambda m: f"{_match_case('there', m.group(1))}{m.group(2)}{m.group(3)}", text)
    text = _YOUR_RE.sub(
        lambda m: _match_case("you're", m.group(1)) + m.group(2) + m.group(3), text)
    text = _DOUBLED_RE.sub(lambda m: m.group(1), text)
    text = _LONE_I_RE.sub("I", text)

    def _article_repl(m):
        art, gap, word = m.group(1), m.group(2), m.group(3)
        correct = "an" if _wants_an(word) else "a"
        if art.lower() == correct:
            return m.group(0)
        return f"{_match_case(correct, art)}{gap}{word}"
    # [A-Z]?[a-z]+ excludes acronyms ("a FBI agent" is a letter-sound problem
    # this rule can't judge) and single letters ("a U turn").
    text = re.sub(r"\b(a|an|A|An)(\s+)([A-Za-z][a-z]+)\b", _article_repl, text)
    return text

def fix_comma_splices(text, doc=None):
    """
    Promotes the comma in a comma splice to a period and capitalizes the word
    that now opens the new sentence ("He ran fast, he was late." -> "He ran
    fast. He was late.").

    A period is used rather than a semicolon: it is always correct here, and
    this tool's readers are better served by plain sentences.
    """
    if not NLP_AVAILABLE:
        return text
    if doc is None:
        doc = get_spacy_doc(text)
    splices = detect_comma_splices(doc)
    if not splices:
        return text

    edits = []
    for splice in splices:
        edits.append((splice["fix_start"], splice["fix_start"] + 1, "."))
        cap_at = splice["capitalize_at"]
        if cap_at is not None and text[cap_at].islower():
            edits.append((cap_at, cap_at + 1, text[cap_at].upper()))

    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text

def fix_spelling(text, doc=None):
    """
    Auto-corrects unambiguous typos: only words the spelling analysis flags
    (proper nouns and acronyms are already excluded there), and only when the
    most probable correction is a real word one edit away. Case-preserving.
    """
    if not (SPELLING_AVAILABLE and NLP_AVAILABLE):
        # Without spaCy there is no proper-noun filter, and auto-"correcting"
        # character names would corrupt the manuscript. Detection-only then.
        return text
    if doc is None:
        doc = get_spacy_doc(text)
    issues = get_spelling_analysis(text, doc=doc)
    if not issues:
        return text

    spell = SpellChecker()
    edits = []
    for issue in issues:
        lower = issue["word"].lower()
        one_edit = spell.known(spell.edit_distance_1(lower))
        if not one_edit:
            continue  # nothing plausible within one edit — leave it flagged
        correction = spell.correction(lower)
        if not correction or correction == lower or correction not in one_edit:
            continue
        edits.append((issue["start"], issue["end"],
                      _match_case(correction, issue["word"])))

    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text

def fix_prose(text, doc=None):
    """
    Applies deterministic fixes to the text using Regex and NLP.
    If 'doc' is provided, NLP refinements use it directly to avoid re-parsing.

    Fix layers, in order: dialogue-tag punctuation, mechanical regex fixes,
    NLP refinements (its/it's, then/than, capitalization), spelling.
    """
    # 1. Fix Dialogue Tags (Regex is good for this structure - supports smart quotes)
    # "Quote." he said -> "Quote," he said
    fixed_text = re.sub(r'((?:^|[\s\r\n])(?:[\"“\u201c\u2018]).*?)\.(?=(?:[\"”\u201d\u2019])\s+[a-z])', r'\1,', text)

    # 2. Mechanical fixes (contractions, a/an, modal+of, doubled words, ...)
    fixed_text = fix_mechanical_errors(fixed_text)

    if NLP_AVAILABLE:
        # Each stage below needs a doc that matches the current text, so the
        # doc is dropped as soon as any stage edits it and re-parsed lazily by
        # the next stage that needs one.
        if fixed_text != text:
            doc = None

        # 3. Comma splices -> separate sentences.
        spliced = fix_comma_splices(fixed_text, doc)
        if spliced != fixed_text:
            doc = None
            fixed_text = spliced

        # 4. NLP refinements (Its/It's, then/than, Capitalization)
        refined = refine_creative_text(fixed_text, doc)
        if refined != fixed_text:
            doc = None
            fixed_text = refined

        # 5. Spelling autocorrect.
        fixed_text = fix_spelling(fixed_text, doc=doc)
    else:
        # Fallback simple regex for "it's" -> "its" (heuristic)
        fixed_text = re.sub(r"it's\s+(?!\b(?:a|an|the|is|has|been|was|going|about)\b)(\w+)", r"its \1", fixed_text, flags=re.IGNORECASE)

    return fixed_text

def print_report(report):
    """
    Helper to print the dict report to stdout (mimicking original behavior).
    """
    if not report:
        print("No report data available.")
        return

    print(format_executive_summary(report.get("executive_summary")))
    print("--- PROSE LINT REPORT ---")
    print(format_readability(report.get("readability")))
    print(format_passive_voice(report.get("passive_voice")))
    print(format_tone(report.get("tone")))
    print(format_pro_editing(report.get("pro_editing")))
    print(format_advanced_features(report.get("advanced_features")))
    tts = report.get("tts_analysis")
    if tts:
        print(format_tts_analysis(tts, report.get("original_text")))
    print(format_errors(report))


def format_tts_analysis(tts, text=None):
    if not tts:
        return ""
    lines = ["\n--- TTS & AUDIO ANALYSIS ---"]
    
    ph = tts.get("phoneme_collisions", [])
    if ph:
        lines.append(f"[!] PHONEME COLLISIONS: Found {len(ph)} instances.")
        for item in ph:
            lines.append(f"    - \"{item['text']}\"")
    
    tw = tts.get("tongue_twisters", [])
    if tw:
        lines.append(f"[!] TONGUE TWISTERS: Found {len(tw)} dense clusters.")
        for item in tw:
            lines.append(f"    - \"{item['text']}\"")
            
    br = tts.get("breath_pacing", [])
    if br:
        lines.append(f"[!] BREATH PACING ISSUES: Found {len(br)} segments.")
        for item in br:
            lines.append(f"    - {item['syllables']} syllables: \"{item['text'][:60]}...\"")
            
    qa = tts.get("quote_attribution", [])
    if qa:
        lines.append(f"[!] QUOTE ATTRIBUTION: Found {len(qa)} long un-attributed quotes.")
        for item in qa:
            lines.append(f"    - \"{item['text'][:60]}...\"")
            
    nf = tts.get("narrator_fatigue", [])
    if nf:
        lines.append(f"[!] NARRATOR FATIGUE: Found {len(nf)} fatiguing sentences.")
        for item in nf:
            lines.append(f"    - {item['syllables']} syllables: \"{item['text'][:60]}...\"")

    hg = tts.get("homographs", [])
    if hg:
        lines.append(f"[!] HOMOGRAPHS: Found {len(hg)} words that may need pronunciation guidance.")
        words = sorted(list(set(item['text'].lower() for item in hg)))
        lines.append(f"    - {', '.join(words)}")

    cc = tts.get("cadence_cliffs", [])
    if cc:
        lines.append(f"[!] CADENCE CLIFFS: Found {len(cc)} abrupt pacing shifts.")
        for item in cc:
            lines.append(f"    - Long ({len(item['long_sentence'].split())} words) followed by Short ({len(item['short_sentence'].split())} words)")

    if text:
        from tts_analyzer import generate_read_aloud_preview
        lines.append("\n[READ ALOUD PREVIEW]")
        lines.append(generate_read_aloud_preview(text, tts))
        
    if not any([ph, tw, br, qa, nf, hg, cc]):
        lines.append("[+] Audio flows well with no obvious TTS pronunciation or breath issues.")
        
    return "\n".join(lines)

def format_executive_summary(summ):
    if not summ:
        return ""
    lines = [
        "",
        "==================================================",
        "               EXECUTIVE SUMMARY",
        "==================================================",
        f" Word Count       : {summ['word_count']}",
        f" Reading Time     : {summ.get('reading_time', '0s')}",
        f" Avg Sentence     : {summ['avg_sent_len']:.1f} words",
        f" Dialogue %       : {summ['dialogue_ratio']:.1f}%",
        f" Passive Voice    : {summ['passive_perc']:.1f}%",
        f" Suppressed Issues: {summ.get('suppressions_count', 0)}",
    ]
    vocab = summ.get('vocabulary')
    if vocab:
        lines.append(f" Vocabulary       : {vocab['rating']} ({vocab['diversity_ratio']}% unique)")
        
    lines.append("--------------------------------------------------")
    
    spec = summ.get('specificity')
    if spec:
        lines.append(f" Specificity      : {spec['rating']} ({spec['ratio']:.1f}%)")
    tone = summ.get('tone')
    if tone:
        lines.extend([
            f" Pacing           : {tone['pacing']}",
            f" Complexity       : {tone['complexity']}"
        ])
    else:
        lines.append(" Pacing           : N/A")
    lines.append("==================================================\n")
    return "\n".join(lines)


def format_readability(readability):
    if not readability:
        return ""
    lines = [
        "\n--- READABILITY SCORE ---",
        f"Flesch Reading Ease: {readability['score']:.2f} ({readability['interpretation']})",
        "\n[Suggestions to improve score]"
    ]
    for sugg in readability.get('suggestions', []):
        lines.append(f" - {sugg}")
    return "\n".join(lines)


def format_passive_voice(pv):
    if not pv: return "Passive Voice: N/A"
    
    perc = pv.get('passive_percentage', 0)
    count = pv.get('passive_count', 0)
    total = pv.get('total_sentences', 0)
    
    lines = [
        "\n--- PASSIVE VOICE ANALYSIS ---",
        f"Passive constructions: {perc:.1f}% ({count} of {total} sentences)"
    ]
    if perc > 20: # Threshold for sentence-based ratio
        lines.extend([
            " [!] WARNING: High passive voice usage detected.",
            "     More than 20% of your sentences are passive. Try converting some",
            "     of these to active voice for more direct impact."
        ])
    examples = pv.get('examples', [])
    if examples:
        lines.append("\n[Examples in your text]")
        # Group by text to avoid redundancy if multiple tokens in same sentence
        seen_sentences = set()
        for ex in examples:
            ex_text = ex.get('text', 'Unknown')
            if ex_text not in seen_sentences:
                lines.append(f" - {ex_text}")
                seen_sentences.add(ex_text)
    return "\n".join(lines)


def format_tone(tone):
    if not tone:
        return ""
    lines = [
        "\n--- TONE ANALYSIS ---",
        f"Pacing: {tone['pacing']}",
        f"Complexity: {tone['complexity']}",
        f"Intensity: {tone['intensity']}"
    ]
    if tone.get('is_all_over_the_place'):
        lines.extend([
            "\n[!] TONE WARNING: Your writing style is inconsistent.",
            "    Paragraphs vary significantly in sentence length and structure.",
            "    This can make the read feel jerky or 'all over the place'.",
            "    -> Aim for smoother transitions between descriptive and action-heavy sections."
        ])
    else:
        lines.append("\n[+] Tone Consistency: Your pacing and complexity are relatively stable.")
    return "\n".join(lines)


def format_pro_editing(pe):
    if not pe:
        return ""
    lines = []
    
    fv = pe.get('filter_verbs')
    if fv:
        lines.append("\n--- FILTER VERBS (Distancing) ---")
        lines.append("Try removing these filters to make the prose more immediate:")
        for item in fv:
            lines.append(f" - [Sentence {item['sentence_id']}] \"{item['text']}\"")
    else:
        lines.append("\n--- FILTER VERBS (Distancing) ---")
        lines.append("No filter verbs detected. Excellent immersion!")
    
    lines.append("\n--- SENSORY BALANCE ---")
    for sense, count in pe.get('sensory_balance', {}).items():
        lines.append(f" - {sense}: {count}")
    
    lines.append(f"\n--- PACING (Dialogue Ratio) ---")
    lines.append(f"Dialogue percentage: {pe['dialogue_ratio']:.1f}% ({pe['dialogue_interpretation']})")
    
    vw = pe.get('vague_words')
    if vw:
        lines.append("\n--- VAGUE WORD CHECK ---")
        lines.append("Consider replacing these weak words with stronger alternatives:")
        for item in vw:
            lines.append(f" - [Sentence {item['sentence_id']}] \"{item['text']}\"")
    else:
        lines.append("\n--- VAGUE WORD CHECK ---")
        lines.append("No vague words detected. Strong vocabulary!")
    
    return "\n".join(lines)


def format_advanced_features(adv):
    if not adv:
        return ""
    lines = []
    
    ahs = adv.get('adverb_heavy_sentences')
    if ahs:
        lines.append(f"\n--- ADVERB USAGE ---")
        lines.append(f"Found {len(ahs)} sentence(s) with multiple adverbs.")
        for item in ahs:
            lines.append(f" - [Sentence {item['sentence_id']}] \"{item['text']}\" (contains: {', '.join(item['adverbs'])})")
    else:
        lines.append("\n--- ADVERB USAGE ---")
        lines.append("No adverb-heavy sentences found. Great job on 'showing'!")
    
    we = adv.get('word_echoes')
    if we:
        lines.append("\n--- WORD ECHOES (Repetition) ---")
        lines.append("Distinctive words used too close together:")
        for word, matches in we:
            lines.append(f" - \"{word}\" echoed by: {', '.join(matches[:3])}")
    else:
        lines.append("\n--- WORD ECHOES (Repetition) ---")
        lines.append("No significant word echoes detected.")
    
    lines.append("\n--- SENTENCE STRUCTURE DISTRIBUTION ---")
    for stype, perc in adv.get('sentence_types', {}).items():
        lines.append(f" - {stype}: {perc:.1f}%")
    
    wdt = adv.get('weak_dialogue_tags')
    if wdt:
        lines.append("\n--- DIALOGUE ANALYSIS ---")
        lines.append("Detected weak or 'stiff' dialogue tags:")
        for tag in wdt:
            lines.append(f" - [Sentence {tag['sentence_id']}] {tag['text']}")
    else:
        lines.append("\n--- DIALOGUE ANALYSIS ---")
        lines.append("Dialogue tags look natural or simple. Good focus on the voice.")
    
    cl = adv.get('cliches')
    if cl:
        lines.append("\n--- CLICHÉ ALERTS ---")
        lines.append("Consider replacing these overused phrases with more original descriptions:")
        # Dedup by text
        seen_cliches = set()
        for c in cl:
            c_text = c.get('text', 'Unknown')
            if c_text not in seen_cliches:
                lines.append(f" - Found: \"{c_text}\"")
                seen_cliches.add(c_text)
    else:
        lines.append("\n--- CLICHÉ ALERTS ---")
        lines.append("No common clichés found. Originality points!")
    
    return "\n".join(lines)


def format_errors(report):
    lines = []
    
    if report.get('dialogue_punctuation_errors'):
        errs = report['dialogue_punctuation_errors']
        lines.append(f"\n[!] DIALOGUE TAG ERROR: Found {len(errs)} instance(s) of periods inside quotes before a tag.")
        lines.append("    Example: \"Hello.\" he said. -> \"Hello,\" he said.")
    
    if report.get('possessive_errors'):
        errs = report['possessive_errors']
        lines.append(f"\n[!] POSSESSIVE WARNING: Found {len(errs)} possible 'its/it's' errors.")
        lines.append(f"    Check: {errs}")
    
    if report.get('repetitive_structure'):
        items = report['repetitive_structure']
        lines.append(f"\n[!] REPETITIVE STRUCTURE: Found {len(items)} cluster(s) of sentences starting with 'I'.")
        for item in items:
            lines.append(f"    - Starting at sentence {item['start_index'] + 1}, streak of {item['count']} sentences.")
            lines.append(f"      Sample: \"{item['sample']}\"")
    
    if report.get('run_ons'):
        items = report['run_ons']
        lines.append(f"\n[!] RUN-ON SENTENCE WARNINGS: Found {len(items)} potential run-ons.")
        for item in items:
            snippet = (item['text'][:60] + '...') if len(item['text']) > 60 else item['text']
            lines.append(f"    - Sentence {item['sentence_id']} ({item['word_count']} words): \"{snippet}\"")
    
    if report.get('clipped_sentences'):
        clusters = report['clipped_sentences']
        lines.append(f"\n[!] CLIPPED SENTENCE WARNINGS: Found {len(clusters)} clusters of short sentences.")
        for cluster in clusters:
            lines.append(f"    - Cluster starting at sentence {cluster[0]['sentence_id']}:")
            for item in cluster:
                lines.append(f"      [{item['sentence_id']}] {item['text']}")
            lines.append("      -> Consider combining these with commas or conjunctions.")
    
    if report.get('sentence_splices'):
        items = report['sentence_splices']
        lines.append(f"\n[!] POTENTIAL SENTENCE SPLICES: Found {len(items)} possible splicing errors.")
        lines.append("    These are detected sentence boundaries that might need a period.")
        for idx, item in enumerate(items):
            context = item.get('text', '') if isinstance(item, dict) else item
            lines.append(f"    - [{idx+1}] Context: ...{context}...")
    
    if report.get('comma_splices'):
        items = report['comma_splices']
        lines.append(f"\n[!] COMMA SPLICES: Found {len(items)} comma splice(s).")
        lines.append("    Two complete sentences joined by only a comma. Auto-fix splits them.")
        for item in items:
            lines.append(f"    - [Sentence {item['sentence_id']}] \"{item['text']}\"")

    if report.get('dangling_participles'):
        items = report['dangling_participles']
        lines.append(f"\n[!] DANGLING PARTICIPLE CHECK: Found {len(items)} potential issues.")
        lines.append("    Introductory -ing phrases might not match the subject.")
        for item in items:
            lines.append(f"    - [Sentence {item['sentence_id']}] \"{item['clause']}\" -> Subject: [{item['subject']}]")
            lines.append(f"      In: \"{item['sentence'][:60]}...\"")
    
    if report.get('tense_consistency'):
        items = report['tense_consistency']
        lines.append(f"\n[!] TENSE CONSISTENCY: Found {len(items)} paragraphs with mixed tenses.")
        for issue in items:
            minority = "Present" if issue['dominant'] == 'Past' else "Past"
            lines.append(f"    - Para {issue['paragraph_idx']}: 90%+ {issue['dominant']}, but {issue['minority_count']} {minority} verbs.")
            lines.append(f"      Snippet: {issue['snippet']}")
    
    if report.get('nominalizations'):
        items = report['nominalizations']
        lines.append(f"\n[!] ZOMBIE NOUNS (Nominalizations): Found {len(items)} instances.")
        lines.append("    Verbs turned into nouns + weak verbs. Consider reverting to the verb.")
        for item in items:
            lines.append(f"    - [Sentence {item['sentence_id']}] {item['text']}")
    
    if report.get('dialogue_beats'):
        items = report['dialogue_beats'][:5]
        lines.append(f"\n[!] DIALOGUE ACTION BEATS: Found {len(items)} potential punctuation errors.")
        lines.append("    If the character is doing an action, use a period, not a comma.")
        for item in items:
            lines.append(f"    - {item['text']}")
    
    if report.get('redundancy'):
        items = report['redundancy']
        lines.append(f"\n[!] REDUNDANCY CHECK: Found {len(items)} pleonasms.")
        for item in items:
            lines.append(f"    - [Sentence {item['sentence_id']}] {item['phrase']} -> {item['correction']}")
    
    if report.get('show_dont_tell'):
        items = report['show_dont_tell'][:5]
        lines.append(f"\n[!] SHOW, DON'T TELL: Found {len(items)} instances of 'Telling' emotions.")
        lines.append("    Try describing the physical reaction instead of using [To Be] + [Emotion].")
        for item in items:
            lines.append(f"    - [Sentence {item['sentence_id']}] \"{item.get('subject', 'Someone')} {item.get('phrase', '')}\" ({item.get('emotion', 'N/A')})")
    
    if report.get('paragraph_rhythm'):
        items = report['paragraph_rhythm']
        lines.append(f"\n[!] PARAGRAPH VITALITY: Found {len(items)} monotonous paragraphs.")
        lines.append("    These paragraphs have sentences of nearly identical length. Vary them!")
        for s in items:
            lines.append(f"    - Para {s['para_idx']} (Avg {s['avg_len']} words, StdDev {s['std_dev']}): \"{s['snippet']}\"")
    
    spec = report.get('specificity')
    if spec:
        lines.append("\n--- SPECIFICITY SCORE ---")
        lines.append(f"Ratio of Proper Nouns to Common Nouns: {spec['ratio']:.1f}% ({spec['rating']})")
        if spec['rating'] == "Vague":
            lines.append(" -> Consider adding more concrete details, names, or specific places.")
    
    if report.get('adverbial_tags'):
        items = report['adverbial_tags'][:5]
        lines.append(f"\n[!] ADVERBIAL DIALOGUE TAGS: Found {len(items)} instances.")
        lines.append("    Avoid 'said angrily'. let the dialogue convey the tone.")
        for item in items:
            lines.append(f"    - [Sentence {item['sentence_id']}] {item['text']}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python linter.py <filename.txt>")
        print("Running default test sample...")
        sample_text = """I raised my guns. I fired at the mouth. I felt the heat. \"Stop.\" he said. It's tail was wagging. This is a very long sentence that goes on and on and seems to never end because the writer simply refused to use a period and instead decided that adding more words was the best course of action for this specific moment in time."""
        # Now lint_prose handles its own doc and sentence splitting if none provided
        report = lint_prose(sample_text)
        print_report(report)
        
        # Also demo the fix
        print("\n--- AUTO-FIX PREVIEW ---")
        print(fix_prose(sample_text))
        
    else:
        filename = sys.argv[1]
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
            base, ext = filename.rsplit('.', 1) if '.' in filename else (filename, "txt")
            report_filename = f"{base}_report.{ext}"
            fixed_filename = f"{base}_fixed.{ext}"
            lines_filename = f"{base}_lines.{ext}"

            print(f"Analyzing file: {filename}\n")
            
            # Initial parse to get doc and sentences
            if NLP_AVAILABLE:
                doc = get_spacy_doc(content)
                sentences = [sent.text.strip() for sent in doc.sents]
            else:
                sentences = get_sentences(content)

            # Generate Numbered Lines File with Paragraph Markers
            with open(lines_filename, 'w', encoding='utf-8') as f:
                current_pos = 0
                for idx, sent in enumerate(sentences):
                    # Find where this sentence starts in the original content (from current_pos)
                    # We assume sentences appear in order (which they do from get_sentences)
                    found_idx = content.find(sent, current_pos)
                    
                    if found_idx != -1:
                        # Check gaps for double newlines
                        gap = content[current_pos:found_idx]
                        if "\n\n" in gap:
                            f.write("[P]\n")
                        
                        # Update position to end of this sentence
                        current_pos = found_idx + len(sent)
                    
                    f.write(f"[{idx + 1}] {sent}\n")

            # Capture lint process output
            if NLP_AVAILABLE:
                # Reuse the already created doc
                report = lint_prose(content, sentences) 
                # Note: lint_prose would recreate doc if we didn't pass it, 
                # but we can optimize further by passing the doc if we update lint_prose signature
                # For now, let's keep it simple: text and sentences.
            else:
                report = lint_prose(content, sentences)
            
            # Save report
            report_io = io.StringIO()
            with contextlib.redirect_stdout(report_io):
                print(f"Analysis for: {filename}\n")
                print_report(report)
                
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_io.getvalue())
            
            # Save fixed version (Auto-fix is separate but we can still save it)
            # The user asked for Reconstructor improvements, but let's keep this basic fix for now
            fixed_content = fix_prose(content)
            with open(fixed_filename, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            print(f"[+] Report saved to: {report_filename}")
            print(f"[+] Numbered lines saved to: {lines_filename}")
            if fixed_content != content:
                print(f"[+] Fixed version saved to: {fixed_filename}")
            else:
                print(f"[=] No changes made to fixed version (matches original).")
                
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
