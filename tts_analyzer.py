import re
import nltk
from nltk.corpus import cmudict
import syllapy

# Ensure cmudict is available
try:
    cmu = cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    cmu = cmudict.dict()

def get_phonemes(word):
    """Returns the primary phoneme list for a word, or None if not in dictionary."""
    word = word.lower().strip(".,!?;:'\"-()")
    if word and word in cmu:
        return cmu[word][0]
    return None

def _strip_stress(phoneme):
    """Remove stress markers from a CMU phoneme (e.g., 'IY1' -> 'IY', 'EH2' -> 'EH')."""
    return re.sub(r'\d+$', '', phoneme)

def _is_consonant(phoneme):
    """Check if a phoneme is a consonant (vowels end with stress digits in CMU)."""
    return not phoneme[-1].isdigit()

def check_phoneme_collision(text):
    """Detects consecutive words ending/starting with identical or similar consonant phonemes."""
    words_iter = re.finditer(r'\b\w+\b', text)
    words_list = list(words_iter)
    collisions = []
    
    for i in range(len(words_list) - 1):
        m1, m2 = words_list[i], words_list[i+1]
        w1, w2 = m1.group(0), m2.group(0)
        p1 = get_phonemes(w1)
        p2 = get_phonemes(w2)
        
        if p1 and p2:
            last_p1 = _strip_stress(p1[-1])
            first_p2 = _strip_stress(p2[0])
            # Match identical consonants or similar sibilant pairs
            sibilant_groups = [{'S', 'SH', 'Z', 'ZH'}, {'TH', 'DH'}, {'F', 'V'}, {'B', 'P'}, {'D', 'T'}, {'G', 'K'}]
            is_match = (last_p1 == first_p2 and _is_consonant(p1[-1]))
            if not is_match:
                for group in sibilant_groups:
                    if last_p1 in group and first_p2 in group:
                        is_match = True
                        break
            if is_match:
                collisions.append({
                    "text": f"{w1} {w2}",
                    "start": m1.start(),
                    "end": m2.end()
                })
                
    return collisions

def check_tongue_twisters(text):
    """Detects 3+ consecutive words with the same or similar starting consonant phoneme."""
    words_iter = re.finditer(r'\b\w+\b', text)
    words_list = list(words_iter)
    twisters = []
    
    if not words_list:
        return twisters

    count = 0
    current_phoneme = None
    current_seq = []
    
    # Groups of similar consonants that cause tongue twisters
    sibilant_groups = [{'S', 'SH', 'Z', 'ZH'}, {'TH', 'DH'}, {'F', 'V'}, {'B', 'P'}, {'D', 'T'}, {'G', 'K'}]
    
    def _phoneme_matches(a, b):
        if a == b:
            return True
        for group in sibilant_groups:
            if a in group and b in group:
                return True
        return False
    
    for m in words_list:
        w = m.group(0)
        p = get_phonemes(w)
        if p and _is_consonant(p[0]):
            stripped = _strip_stress(p[0])
            if current_phoneme and _phoneme_matches(current_phoneme, stripped):
                count += 1
                current_seq.append(m)
            else:
                if count >= 3:
                    twisters.append({
                        "text": " ".join(m.group(0) for m in current_seq),
                        "start": current_seq[0].start(),
                        "end": current_seq[-1].end()
                    })
                count = 1
                current_phoneme = stripped
                current_seq = [m]
        else:
            if count >= 3:
                twisters.append({
                    "text": " ".join(m.group(0) for m in current_seq),
                    "start": current_seq[0].start(),
                    "end": current_seq[-1].end()
                })
            count = 0
            current_phoneme = None
            current_seq = []
            
    if count >= 3:
        twisters.append({
            "text": " ".join(m.group(0) for m in current_seq),
            "start": current_seq[0].start(),
            "end": current_seq[-1].end()
        })
        
    return twisters

def check_breath_pacing(text):
    """Identifies clauses (between punctuation) that are too long (>30 syllables)."""
    # Split by punctuation that allows breathing, keeping offsets
    pattern = re.compile(r'[,.!?;:-]+')
    last_end = 0
    long_clauses = []
    
    for m in pattern.finditer(text):
        clause = text[last_end:m.start()]
        words = re.findall(r'\b\w+\b', clause)
        syllables = sum(syllapy.count(w) for w in words)
        if syllables > 30:
            long_clauses.append({
                "text": clause.strip(),
                "syllables": syllables,
                "start": last_end,
                "end": m.start()
            })
        last_end = m.end()
    
    # Check trailing text
    clause = text[last_end:]
    words = re.findall(r'\b\w+\b', clause)
    syllables = sum(syllapy.count(w) for w in words)
    if syllables > 30:
        long_clauses.append({
            "text": clause.strip(),
            "syllables": syllables,
            "start": last_end,
            "end": len(text)
        })
            
    return long_clauses

def check_quote_attribution(text):
    """Detects dialogue exceeding 20 words without an attribution."""
    issues = []
    # Match content inside quotes
    quotes = re.finditer(r'["“](.*?)["”]', text)
    for q in quotes:
        quote_text = q.group(1)
        words = quote_text.split()
        if len(words) > 20:
            issues.append({
                "text": quote_text,
                "start": q.start(),
                "end": q.end()
            })
    return issues

def check_narrator_fatigue(text):
    """Detects sentences with > 40 syllables total."""
    # Simple sentence split while maintaining offsets
    pattern = re.compile(r'[.!?;]+')
    last_end = 0
    fatiguing = []
    
    for m in pattern.finditer(text):
        sent = text[last_end:m.end()]
        words = re.findall(r'\b\w+\b', sent)
        syllables = sum(syllapy.count(w) for w in words)
        if syllables > 40:
            fatiguing.append({
                "text": sent.strip(),
                "syllables": syllables,
                "start": last_end,
                "end": m.end()
            })
        last_end = m.end()
            
    return fatiguing

def check_homographs(text):
    """Flags words with multiple pronunciations based on context."""
    homographs = {"read", "wind", "tear", "lead", "bow", "live", "object", "close", "minute", "invalid", "project", "record"}
    found = []
    for m in re.finditer(r'\b\w+\b', text, re.IGNORECASE):
        word = m.group(0).lower()
        if word in homographs:
            found.append({
                "text": m.group(0),
                "start": m.start(),
                "end": m.end()
            })
    return found

def check_cadence_cliffs(text):
    """Detects abrupt shifts from very long sentences (>25 words) to extremely short ones (<5 words)."""
    pattern = re.compile(r'[.!?;]+')
    last_end = 0
    sentences = []
    for m in pattern.finditer(text):
        sentences.append({
            "text": text[last_end:m.end()].strip(),
            "start": last_end,
            "end": m.end()
        })
        last_end = m.end()
    
    cliffs = []
    for i in range(len(sentences) - 1):
        len1 = len(sentences[i]["text"].split())
        len2 = len(sentences[i+1]["text"].split())
        
        if len1 > 25 and 0 < len2 < 5:
            cliffs.append({
                "long_sentence": sentences[i]["text"],
                "short_sentence": sentences[i+1]["text"],
                "drop": len1 - len2,
                "start": sentences[i]["start"],
                "end": sentences[i+1]["end"]
            })
            
    return cliffs

def generate_read_aloud_preview(text, report):
    """
    Produces a markup version of the text highlighting phoneme collisions 
    and breath pacing issues for easy identification during a read-aloud.
    """
    # Collect all offsets to highlight
    highlights = []
    for item in report.get("phoneme_collisions", []):
        highlights.append((item["start"], item["end"], "collision"))
    for item in report.get("breath_pacing", []):
        highlights.append((item["start"], item["end"], "pacing"))
    
    # Sort highlights by start offset
    highlights.sort()
    
    # Merge or handle overlapping highlights if necessary (simplified here)
    # For now, just apply them in reverse to not mess up offsets
    preview = text
    for start, end, kind in sorted(highlights, reverse=True):
        if kind == "collision":
            preview = preview[:start] + "[[COLLISION:" + preview[start:end] + "]]" + preview[end:]
        elif kind == "pacing":
            preview = preview[:start] + "[[BREATH:" + preview[start:end] + "]]" + preview[end:]
            
    return preview

def run_tts_analysis(text):
    """Runs all TTS checks on the provided text."""
    report = {
        "phoneme_collisions": check_phoneme_collision(text),
        "tongue_twisters": check_tongue_twisters(text),
        "breath_pacing": check_breath_pacing(text),
        "quote_attribution": check_quote_attribution(text),
        "narrator_fatigue": check_narrator_fatigue(text),
        "homographs": check_homographs(text),
        "cadence_cliffs": check_cadence_cliffs(text)
    }
    return report

if __name__ == "__main__":
    sample = "She sells seashells by the seashore. The sixth sheep is sick. " \
             "He read the book while the wind was blowing. " \
             "This is a very long quote that goes on and on for a very long time without any attribution to help the voice actor know who is speaking before they start reading it out loud." \
             "This is a massive sentence that goes on for far too long and will definitely exhaust the narrator because they have to keep speaking without a single break for a long time. It stops."
    
    import json
    report = run_tts_analysis(sample)
    print(json.dumps(report, indent=2))
    print("\n--- READ ALOUD PREVIEW ---")
    print(generate_read_aloud_preview(sample, report))

