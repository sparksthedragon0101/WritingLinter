import spacy
import re
import math
from collections import defaultdict, deque
from functools import lru_cache
from typing import Optional


# Dictionary to cache multiple models (sm, md, lg, trf)
_nlp_models = {}

def get_nlp_model(model_name: str = "en_core_web_md"):
    """
    Returns a cached spaCy NLP model, loading it if necessary.
    Supports sm, md, lg, and trf (transformer) models.
    """
    global _nlp_models
    if model_name not in _nlp_models:
        import spacy
        try:
            # We exclude NER and attribute_ruler for speed, unless it's the transformer model
            # where we typically want its full accuracy (though we can still speed up sm/md/lg)
            if "trf" in model_name:
                # Transformer model usually wants its full pipeline for accuracy
                _nlp_models[model_name] = spacy.load(model_name)
            else:
                # We exclude NER for speed, but keep attribute_ruler for the lemmatizer
                _nlp_models[model_name] = spacy.load(model_name, exclude=["ner"])
        except OSError:
            # Fallback to sm if the requested model isn't installed
            print(f"Warning: Model '{model_name}' not found. Falling back to 'en_core_web_sm'.")
            if "en_core_web_sm" not in _nlp_models:
                _nlp_models["en_core_web_sm"] = spacy.load("en_core_web_sm", exclude=["ner"])
            return _nlp_models["en_core_web_sm"]
            
    return _nlp_models[model_name]

def get_spacy_doc(text: str, model_name: str = "en_core_web_md"):
    """Creates a spacy doc for the given text using the specified model."""
    nlp = get_nlp_model(model_name)
    return nlp(text)

# === Constants ===

WEAK_DIALOGUE_TAGS = {"remarked", "stated", "ejaculated", "retorted", "interjected", "commented"}

# Lemmas of verbs that head dialogue tags ("...?" he asked). Used to keep the
# capitalization pass from wrongly uppercasing a tag after ?" or !".
SPEECH_VERBS = {
    "say", "ask", "whisper", "shout", "reply", "murmur", "mutter", "call",
    "cry", "answer", "snap", "hiss", "yell", "add", "continue", "agree",
    "admit", "warn", "wonder", "exclaim", "demand", "insist", "repeat",
    "respond", "groan", "moan", "sigh", "laugh", "chuckle", "growl", "scream",
    "state", "remark", "observe", "note", "urge", "plead", "beg", "interrupt",
    "interject", "retort", "echo", "drawl", "stammer", "stutter", "mumble",
    "whimper", "croak", "rasp", "snarl", "blurt", "gasp", "wail", "sob",
    "tease", "joke", "quip", "counter", "concede", "confess", "declare",
    "announce", "order", "command", "suggest", "protest", "argue", "promise",
    "vow", "swear", "boast", "sneer", "scoff", "mock", "chide", "scold",
    "caution", "advise", "begin", "finish", "conclude", "press", "venture",
    "manage", "offer", "breathe", "purr", "bark", "prompt", "correct",
}

CLICHES = [
    "dead of night", "cold as ice", "heart of gold", "raced like a bullet",
    "at the end of the day", "quick as a wink", "bolt from the blue",
    "burn the midnight oil", "hit the nail on the head", "piece of cake"
]

DIALOGUE_TAG_PATTERN = re.compile(r'(?:[\"“\u201c\u2018]).*?(?:[\"”\u201d\u2019])\s+([a-zA-Z]+)')
DIALOGUE_QUOTE_PATTERN = re.compile(r'(?:[\"“\u201c\u2018]).*?(?:[\"”\u201d\u2019])')
CLICHE_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(c) for c in CLICHES) + r')\b')

FILTER_VERBS = {"see", "hear", "feel", "notice", "watch", "look", "listen", "realize", "think", "wonder", "know", "experience"}

SENSORY_KEYWORDS = {
    "Sight": {"see", "look", "blue", "bright", "dark", "shadow", "glimmer", "color", "red", "green", "visible", "watch", "glance"},
    "Sound": {"hear", "listen", "sound", "noise", "loud", "quiet", "whisper", "shout", "bang", "thud", "melody", "silence", "echo"},
    "Touch": {"feel", "touch", "cold", "hot", "warm", "sharp", "soft", "hard", "rough", "smooth", "pressure"},
    "Smell": {"smell", "scent", "aroma", "fragrance", "stink", "reek", "perfume", "sniff"},
    "Taste": {"taste", "bitter", "sweet", "sour", "salty", "flavor", "delicious", "savory"}
}

SENSORY_LOOKUP = {kw: s_type for s_type, keywords in SENSORY_KEYWORDS.items() for kw in keywords}

VAGUE_WORDS = {"very", "really", "just", "actually", "somehow", "somewhat", "things", "stuff", "basically", "literally"}

EMOTION_ADJECTIVES = {
    "angry", "sad", "happy", "scared", "afraid", "excited", "bored",
    "tired", "confused", "depressed", "anxious", "nervous", "frustrated",
    "mad", "upset", "surprised", "shocked", "furious", "elated", "joyful",
    "exhausted", "lazy", "worried", "ashamed", "guilty", "jealous", "lonely"
}

SPEECH_VERBS = {"say", "ask", "reply", "whisper", "shout", "mutter", "scream", "yell", "tell", "answer", "remark", "state"}

# Common words to filter out from Word Echoes (author-specific distinctive word focus)
COMMON_REPETITIONS_FILTER = {
    "looked", "back", "went", "came", "turned", "eyes", "hand", "hands", 
    "face", "room", "door", "time", "way", "felt", "knew", "thought",
    "seemed", "began", "started", "looked", "look", "standing", "stood"
}

WEAK_VERBS = {"be", "make", "give", "have", "do", "take", "conduct", "perform", "occur", "transpire"}

NOMINALIZATION_SUFFIXES = ("ion", "ment", "ance", "ence", "ions", "ments", "ances", "ences")

LINKING_VERBS = {"be", "seem", "feel", "appear", "look", "become"}

# Simple Sentiment Word Lists for Speed
POSITIVE_WORDS = {
    "excellent", "good", "great", "wonderful", "amazing", "beautiful", "happy", "joy", "love", "awesome",
    "best", "fantastic", "special", "better", "brilliant", "outstanding", "perfect", "worth", "nice"
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "sad", "angry", "hate", "worst", "poor", "difficult",
    "wrong", "ugly", "useless", "cheap", "broken", "pain", "sorry", "unhappy", "mean"
}

def is_independent_clause(token, doc):
    """
    Determines if the span starting at 'token' is likely an independent clause.
    Requires a Subject and a Finite Verb (or Aux).
    """
    curr = token
    local_root = token
    
    steps = 0
    while steps < 20: 
        if curr.head == curr: 
            local_root = curr
            break
        if curr.head.i < token.i: 
            local_root = curr
            break
        curr = curr.head
        steps += 1
        local_root = curr

    if local_root.pos_ not in ['VERB', 'AUX']:
        return False

    has_subject = any(child.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'expl'] for child in local_root.children)
    
    if not has_subject:
        return False
        
    return True


def _get_previous_token(doc, current_idx: int):
    """Get previous non-space token."""
    j = current_idx - 1
    while j >= 0:
        if doc[j].pos_ != 'SPACE':
            return doc[j]
        j -= 1
    return None


def _analyze_sentence_boundary(token, prev, doc):
    """
    Analyzes if there's a sentence boundary between prev and token.
    Returns dict with 'should_capitalize' and 'is_splice' flags.
    """
    result = {"should_capitalize": False, "is_splice": False}
    
    if prev is None:
        result["should_capitalize"] = True
        return result
    
    is_splice = False
    
    if token.is_sent_start or (token.text[0].isupper() or token.text == "I"):
        is_splice = True
    
    # FILTER: Basic Logic
    if is_splice:
        if prev.text in ['.', '!', '?', '"', "'", '"', "'"] or prev.pos_ in ['PUNCT', 'CCONJ', 'SCONJ']:
            is_splice = False
        if token.head == prev.head or token.head == prev or prev.head == token:
            is_splice = False

    # FILTER 1: Right Side Independence
    if is_splice:
        if not is_independent_clause(token, doc):
            is_splice = False

    # FILTER 2: Left Side Independence
    if is_splice:
        curr_prev = prev
        steps = 0
        while steps < 20 and curr_prev.head != curr_prev and curr_prev.head.i < curr_prev.i:
            curr_prev = curr_prev.head
            steps += 1
        left_root = curr_prev
        
        if left_root.pos_ not in ['VERB', 'AUX']:
            is_splice = False
        else:
            has_sub = any(c.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'expl'] for c in left_root.children)
            if not has_sub:
                is_splice = False
    
    # FILTER 3: Conjunction Protection (And/But/So)
    if is_splice and token.text in ["And", "But", "So"]:
        k = prev.i
        count = 0
        limit = 10
        is_short = True
        while k >= 0 and count <= limit:
            t = doc[k]
            if t.text in ['.', '!', '?']:
                break
            count += 1
            k -= 1
        
        if count > 10:
            is_short = False
            
        if is_short:
            is_splice = False

    # FILTER 4: Introductory Phrases
    if is_splice:
        if prev.text.lower() == "anyways":
            is_splice = False
        
        if prev.i >= 2:
            start = prev.i - 2
            end = prev.i + 1
            if start >= 0:
                phrase = doc[start:end].text.lower()
                if "funny thing is" in phrase:
                    is_splice = False
                    
        if prev.i >= 1:
            start = prev.i - 1
            end = prev.i + 1
            if start >= 0:
                phrase = doc[start:end].text.lower()
                if "each time" in phrase:
                    is_splice = False
    
    result["is_splice"] = is_splice
    
    # Capitalization rules
    if prev.text in ['.', '!', '?']:
        result["should_capitalize"] = True
    elif prev.text in ['"', '"', "''"] and prev.i > 0:
        prev_prev = _get_previous_token(doc, prev.i)
        if prev_prev and prev_prev.text == '.':
            result["should_capitalize"] = True
        elif prev_prev and prev_prev.text in ['!', '?']:
            # After ?" or !" a lowercase dialogue tag is correct English
            # ('"Ready?" he asked.'). Only capitalize when neither the word
            # nor its head verb reads as a speech tag.
            is_tag = (token.lemma_.lower() in SPEECH_VERBS or
                      (token.head is not token and token.head.lemma_.lower() in SPEECH_VERBS))
            if not is_tag:
                result["should_capitalize"] = True

    return result


def detect_sentence_splices(doc: spacy.tokens.Doc) -> list[tuple[int, str]]:
    """
    Analyzes text for potential sentence splices where a period might be missing.
    """
    splices = []
    
    for i, token in enumerate(doc):
        try:
            prev = _get_previous_token(doc, i)
            
            if prev is None:
                continue
            
            analysis = _analyze_sentence_boundary(token, prev, doc)
            
            if analysis["is_splice"]:
                splices.append({
                    "start": prev.idx,
                    "end": token.idx + len(token.text),
                    "type": "splice",
                    "text": doc[prev.i : token.i + 1].text
                })
                
        except (AttributeError, IndexError):
            continue
            
    return splices

# Joining two independent clauses with ", and" / ", but" / ", so" is correct
# punctuation, so a comma before one of these is never a splice.
COORDINATORS = {"and", "but", "or", "nor", "for", "yet", "so"}

# A comma before one of these does not fix a splice ("He was late, however he
# tried" is still spliced), so these are detected rather than exempted. They
# are listed so the pass can tell a splice's own trailing comma
# ("..., however, he...") from a second, separate splice.
CONJUNCTIVE_ADVERBS = {
    "however", "therefore", "moreover", "furthermore", "consequently",
    "nevertheless", "nonetheless", "otherwise", "thus", "hence", "besides",
    "instead", "meanwhile", "accordingly", "similarly", "likewise",
}

_QUOTE_CHARS = {'"', "'", "“", "”", "‘", "’", "``", "''"}

# Verbs used almost exclusively for dialogue attribution. Only these protect an
# unquoted tag from the splice rule; broader speech verbs (answer, shout,
# laugh, growl) are common as ordinary predicates in real splices ("The phone
# rang, nobody answered.") and would cause misses.
CORE_DIALOGUE_TAGS = {"say", "ask", "reply", "whisper", "murmur", "mutter"}


def _is_finite(token):
    return token.morph.get("VerbForm") == ["Fin"]


def _finite_clause_head(token):
    """
    The head of the finite clause a subject token belongs to, or None.

    Handles periphrastic verbs ("It was raining"), where the subject's head is
    the participle and the finite verb hangs off it as an `aux`.
    """
    head = token.head
    if head is None or head is token:
        return None
    if _is_finite(head):
        return head
    if any(c.dep_ in ("aux", "auxpass") and _is_finite(c) for c in head.children):
        return head
    return None


def _has_subordinator(head):
    """
    True if a subordinating conjunction *introduces* this clause, making it
    dependent. Covers both `mark` ("After he left") and wh-adverbs, which
    spaCy attaches as `advmod` ("When she arrived").

    Position matters: only a subordinator standing before the clause head
    subordinates it. A trailing one is a complement, not a subordinator
    ("she had no idea where" is still an independent clause), and the parser
    sometimes attaches such a word directly to the head.
    """
    for child in head.children:
        if child.i >= head.i:
            continue
        if child.dep_ == "mark" or child.pos_ == "SCONJ" or child.tag_ == "WRB":
            return True
    return False


def _independent_clause_in(doc, sent, comma, side):
    """
    Finds an independent clause strictly on one `side` ("left" or "right") of
    `comma`. Independent means: a finite verb whose subject sits on the same
    side, carrying no subordinating `mark` (which would make it a dependent
    clause) and not a relative clause.

    Returns (head, subject) or (None, None). For the right side the *first*
    subject after the comma wins, so the clause examined is the one the comma
    actually introduces.
    """
    candidates = sent if side == "right" else reversed(list(sent))
    for token in candidates:
        if token.dep_ not in ("nsubj", "nsubjpass"):
            continue
        if side == "left" and token.i >= comma.i:
            continue
        if side == "right" and token.i <= comma.i:
            continue

        head = _finite_clause_head(token)
        if head is None:
            continue
        # The verb must sit on the same side of the comma as its subject;
        # otherwise the two are one clause spanning the comma (appositives:
        # "Sarah, my sister, is here").
        if side == "left" and head.i >= comma.i:
            continue
        if side == "right" and head.i <= comma.i:
            continue
        # A subordinating conjunction introduces this clause ("After he left,
        # she cried" / "When she arrived, ...") — dependent, not a splice.
        if _has_subordinator(head):
            continue
        # Relative clauses ("The man, who was tall, left") are dependent.
        if head.dep_ in ("relcl", "acl"):
            continue
        if token.tag_ == "WP" or token.lower_ in ("who", "which", "that"):
            continue
        return head, token
    return None, None


def _clause_ends_before(head, comma):
    """
    True if the clause headed by `head` closes before `comma` — i.e. the comma
    is a boundary between clauses rather than punctuation inside one.

    This is what separates "She was cold, tired, and frightened" (the `was`
    clause spans the comma: a list) from "He ran fast, he was late" (the `ran`
    clause stops at the comma: a splice).
    """
    last = max((t.i for t in head.subtree if not t.is_punct and not t.is_space),
               default=head.i)
    return last < comma.i


def _meaningful_neighbor(doc, index, step):
    """Nearest non-space token from `index` walking by `step`."""
    j = index + step
    while 0 <= j < len(doc):
        if not doc[j].is_space:
            return doc[j]
        j += step
    return None


def detect_comma_splices(doc: spacy.tokens.Doc) -> list[dict]:
    """
    Finds comma splices: two independent clauses joined by a bare comma
    ("He ran fast, he was late.").

    Deliberately conservative — a false positive here rewrites correct prose.
    Exempted: comma + coordinator, subordinate clauses, appositives, lists,
    participial phrases, interjections, and dialogue tags ("'I'm late,' he
    said." is structurally a splice but correct English).
    """
    splices = []

    for sent in doc.sents:
        for token in sent:
            if token.text != ",":
                continue

            nxt = _meaningful_neighbor(doc, token.i, 1)
            prev = _meaningful_neighbor(doc, token.i, -1)
            if nxt is None or prev is None:
                continue

            # ", and ..." / ", so ..." — correctly joined.
            if nxt.lower_ in COORDINATORS or nxt.dep_ == "cc":
                continue
            # ", because ..." — dependent clause follows.
            if nxt.pos_ == "SCONJ" or nxt.dep_ == "mark":
                continue
            # Quote adjacent on either side: dialogue tag boundary, not a
            # splice ("'I'm late,' he said.").
            if nxt.text in _QUOTE_CHARS or prev.text in _QUOTE_CHARS:
                continue
            # The trailing comma of a "..., however, ..." splice already
            # reported at the preceding comma.
            if prev.lower_ in CONJUNCTIVE_ADVERBS:
                continue

            right_head, right_subj = _independent_clause_in(doc, sent, token, "right")
            if right_head is None:
                continue
            # Everything between the comma and that subject's noun phrase must
            # be connective filler, otherwise the clause found is not the one
            # this comma introduces (in "She was cold, tired, and frightened",
            # the comma after "cold" opens a list, not a clause).
            subj_start = min(t.i for t in right_subj.subtree)
            gap = [t for t in doc[token.i + 1: subj_start]
                   if not t.is_punct and not t.is_space]
            if any(t.lower_ not in CONJUNCTIVE_ADVERBS for t in gap):
                continue
            # Backstop for dialogue tags written without quotes ("I'm late, he
            # said."). Kept narrow: only bare attribution verbs with no object,
            # so ordinary splices whose second clause happens to use a speech
            # verb ("The phone rang, nobody answered.") are still caught.
            if right_head.lemma_.lower() in CORE_DIALOGUE_TAGS and \
                    not any(c.dep_ == "dobj" for c in right_head.children):
                continue

            left_head, _ = _independent_clause_in(doc, sent, token, "left")
            if left_head is None:
                continue
            # The left clause must close at the comma (lists / appositives fail
            # this because their clause spans the comma).
            if not _clause_ends_before(left_head, token):
                continue

            # The word that would open the new sentence once the comma is
            # promoted to a period.
            first = nxt
            splices.append({
                "start": token.idx,
                "end": token.idx + 1,
                "type": "comma_splice",
                "text": doc[left_head.i: right_head.i + 1].text,
                "fix_start": token.idx,
                "capitalize_at": first.idx if first.is_alpha else None,
                "token_idx": token.i,
            })

    return splices


def refine_creative_text(text: str, doc: Optional[spacy.tokens.Doc] = None) -> str:
    """
    Refines text using NLP for context-aware corrections.
    If 'doc' is provided, it uses the doc's tokens and offsets to modify the original 'text'.
    """
    if not text:
        return text
        
    if doc is None:
        try:
            doc = get_spacy_doc(text)
        except (OSError, ValueError):
            return text
    
    final_edits = []
    
    for i, token in enumerate(doc):
        try:
            prev = _get_previous_token(doc, i)
            analysis = _analyze_sentence_boundary(token, prev, doc)
            
            if analysis["should_capitalize"] and token.is_alpha and token.is_lower:
                # Use token.idx for character offset in the original text
                final_edits.append((token.idx, token.idx+1, token.text[0].upper()))
                
        except (AttributeError, IndexError):
            continue

        try:
            val = token.text.lower()

            # "then" after a comparative should be "than" ("taller then me").
            # Guarded by the next token to avoid the temporal reading
            # ("Things were better then." / "better then suddenly...").
            if val == "then" and prev is not None:
                comparative = (prev.tag_ in ("JJR", "RBR") or
                               prev.lower_ in {"more", "less", "rather", "other",
                                               "better", "worse", "further", "fewer"})
                nxt = doc[i + 1] if i + 1 < len(doc) else None
                if comparative and nxt is not None and nxt.is_alpha and \
                        nxt.pos_ in ("PRON", "DET", "PROPN", "NOUN", "NUM", "ADJ"):
                    replacement = "Than" if token.text[0].isupper() else "than"
                    final_edits.append((token.idx, token.idx + len(token.text), replacement))

            # "Its" used as a subject (usually should be "It's")
            if val == "its":
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    replacement = "It's" if token.text[0].isupper() else "it's"
                    final_edits.append((token.idx, token.idx+3, replacement))
            
            # "It's" followed by a noun (usually should be "Its")
            if val == "it" and i + 1 < len(doc):
                nxt = doc[i+1]
                if nxt.text.lower() == "'s":
                    if i + 2 < len(doc):
                        nxt2 = doc[i+2]
                        if nxt2.pos_ == "NOUN":
                            # Check if the noun already has a determiner (e.g., "It's the car" is correct)
                            has_det = any(child.dep_ == "det" for child in nxt2.children)
                            if not has_det:
                                replacement = "Its" if token.text[0].isupper() else "its"
                                # Replace "It's" (token + nxt)
                                final_edits.append((token.idx, nxt.idx + len(nxt.text), replacement))
        except (AttributeError, IndexError):
            continue

    # Apply Edits from back to front to preserve offsets
    final_edits.sort(key=lambda x: x[0], reverse=True)
    
    # We use a list of characters for efficient slicing
    result_chars = list(text)
    for start, end, replacement in final_edits:
        if start < len(result_chars) and end <= len(result_chars):
            result_chars[start:end] = list(replacement)
            
    return "".join(result_chars)


def get_vocabulary_metrics(doc: spacy.tokens.Doc) -> dict:
    """
    Calculates lexical diversity and unique word usage.
    """
    words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    unique_words = set(words)
    
    diversity = (len(unique_words) / len(words) * 100) if words else 0
    
    return {
        "unique_count": len(unique_words),
        "total_words": len(words),
        "diversity_ratio": round(diversity, 1),
        "rating": "Rich" if diversity > 50 else ("Average" if diversity > 30 else "Repetitive")
    }


def get_passive_voice_report(doc: spacy.tokens.Doc, expected_count: int = None) -> dict:
    """
    Scans for passive voice constructions (be verb + past participle).
    Returns (percentage of passive sentences, example_list).
    """
    passive_sentences_count = 0
    internal_total_sentences = 0
    examples = []
    
    # We define passive as a "be" verb auxiliary (auxpass) attached to a past participle (VBN)
    BE_VERBS = {"be", "is", "was", "were", "been", "being", "am", "are"}

    for sent in doc.sents:
        internal_total_sentences += 1
        sent_passive = False
        
        # We iterate over all tokens in the sentence to find potential passive heads (VBN)
        for token in sent:
            # 1. Start with the main verb/participle (VBN)
            if token.pos_ == "VERB" and token.tag_ == "VBN":
                # 2. Check its children for a "be" auxiliary with the auxpass relation
                be_aux = next((child for child in token.children 
                             if child.dep_ in ["auxpass", "aux:pass"] 
                             and child.lemma_.lower() in BE_VERBS), None)
                
                if be_aux:
                    sent_passive = True
                    # Calculate a range that covers both the auxiliary and the verb
                    start_pos = min(token.idx, be_aux.idx)
                    end_pos = max(token.idx + len(token.text), be_aux.idx + len(be_aux.text))
                    
                    examples.append({
                        "text": sent.text.strip(),
                        "start": start_pos,
                        "end": end_pos,
                        "token_idx": token.i,
                        "type": "passive",
                        "sentence_start": sent.start_char,
                        "sentence_end": sent.end_char
                    })
        
        if sent_passive:
            passive_sentences_count += 1
    
    # Synchronization: Use the expected count (from regex splitting) if it's higher
    # This prevents 100% reports when spaCy fails to find sentence boundaries
    total_sentences = max(internal_total_sentences, expected_count) if expected_count else internal_total_sentences
    
    # Defensive check: if doc has text but no sentences were found, default to 1 to avoid /0
    if total_sentences == 0 and len(doc.text.strip()) > 0:
        total_sentences = 1

    percentage = (passive_sentences_count / total_sentences * 100) if total_sentences > 0 else 0
    return {
        'passive_percentage': round(percentage, 1),
        'examples': examples,
        'total_sentences': total_sentences,
        'internal_sentences': internal_total_sentences
    }

def get_all_paragraph_rhythm_data(doc: spacy.tokens.Doc) -> list[dict]:
    """
    Extracts sentence lengths for every paragraph in the document.
    Optimized to use existing doc.sents instead of re-parsing.
    """
    if not doc: return []
    
    import math
    text = doc.text
    
    # 1. Map sentences to their character offsets and calculate lengths
    sent_infos = []
    for sent in doc.sents:
        # Word count for rhythm (excluding punctuation tokens and whitespace)
        words = [t for t in sent if not t.is_punct and not t.is_space]
        if words:
            sent_infos.append({
                "length": len(words),
                "text": sent.text.strip(),
                "start_char": sent.start_char,
                "end_char": sent.end_char
            })
    
    if not sent_infos: return []
    
    # 2. Group sentences into paragraphs based on double newline gaps
    rhythm_data = []
    current_para_sents = []
    last_end = 0
    para_idx = 0
    
    for info in sent_infos:
        # Check for paragraph break in the gap since the last sentence
        gap = text[last_end:info["start_char"]]
        if para_idx == 0 or "\n\n" in gap or "\r\n\r\n" in gap:
            if current_para_sents:
                rhythm_data.append(_summarize_para_rhythm(current_para_sents, para_idx))
                para_idx += 1
                current_para_sents = []
        
        current_para_sents.append(info)
        last_end = info["end_char"]
        
    if current_para_sents:
        rhythm_data.append(_summarize_para_rhythm(current_para_sents, para_idx))
        
    return rhythm_data

def _summarize_para_rhythm(sent_infos, para_idx):
    import math
    lengths = [s["length"] for s in sent_infos]
    avg_len = sum(lengths) / len(lengths)
    variance = sum((x - avg_len)**2 for x in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    
    return {
        "para_idx": para_idx,
        "avg_len": round(avg_len, 1),
        "std_dev": round(std_dev, 1),
        "snippet": sent_infos[0]["text"][:50] + ("..." if len(sent_infos[0]["text"]) > 50 else ""),
        "sentences": [{"length": s["length"], "text": s["text"][:50] + ("..." if len(s["text"]) > 50 else "")} for s in sent_infos]
    }

def get_tone_profile(doc: spacy.tokens.Doc) -> dict:
    """
    Analyzes text pacing and complexity based on word choice and sentence length.
    """
    word_count = len([t for t in doc if not t.is_punct])
    sent_count = len(list(doc.sents))
    
    # 1. Pacing (Sentence Length Variance)
    sent_lengths = [len([t for t in sent if not t.is_punct]) for sent in doc.sents]
    if not sent_lengths:
        return None
        
    avg_len = sum(sent_lengths) / len(sent_lengths)
    variance = sum((x - avg_len)**2 for x in sent_lengths) / len(sent_lengths)
    std_dev = math.sqrt(variance)
    
    pacing = "Dynamic" if std_dev > 3.5 else "Monotonous"
    if std_dev > 7: pacing = "Erratic"
    
    # 2. Complexity (Syllable ratio)
    vowels = "aeiouy"
    complex_words = 0
    total_words = 0
    for token in doc:
        if token.is_alpha and not token.is_stop:
            total_words += 1
            word = token.text.lower()
            syllable_count = 0
            prev_v = False
            for char in word:
                is_v = char in vowels
                if is_v and not prev_v: syllable_count += 1
                prev_v = is_v
            if syllable_count >= 3:
                complex_words += 1
                
    complexity_ratio = (complex_words / total_words) if total_words > 0 else 0
    complexity = "Simple" if complexity_ratio < 0.1 else "Academic"
    if complexity_ratio > 0.2: complexity = "Very Technical"
    
    # 3. Intensity & Sentiment (Single pass)
    intensity_markers = 0
    pos_count = 0
    neg_count = 0
    
    for t in doc:
        # Intensity: Exclamation and strong adverbs
        if t.text == "!" or (t.pos_ == "ADV" and t.text.lower().endswith("ly")):
            intensity_markers += 1
        
        # Sentiment
        low_text = t.text.lower()
        if low_text in POSITIVE_WORDS:
            pos_count += 1
        elif low_text in NEGATIVE_WORDS:
            neg_count += 1
            
    intensity_ratio = intensity_markers / total_words if total_words > 0 else 0
    intensity = "Calm" if intensity_ratio < 0.05 else "High-Energy"
    
    sentiment_total = pos_count + neg_count
    if sentiment_total > 0:
        sentiment_score = (pos_count - neg_count) / sentiment_total
    else:
        sentiment_score = 0.0 # Neutral
        
    mood = "Neutral"
    if sentiment_score > 0.2: mood = "Positive"
    if sentiment_score > 0.5: mood = "Inspirational"
    if sentiment_score < -0.2: mood = "Negative"
    if sentiment_score < -0.5: mood = "Gloomy / Critical"

    return {
        "pacing": pacing,
        "complexity": complexity,
        "intensity": intensity,
        "sentiment_score": round(sentiment_score, 2),
        "mood": mood,
        "avg_sent_len": avg_len,
        "std_dev": std_dev
    }


def get_advanced_prose_report(doc: spacy.tokens.Doc) -> dict:
    """
    High-level overview of adverbs, sensory details, and lexical density.
    """
    sents = list(doc.sents)
    
    # 2. Word Echoes (Distinctive words repeated in 50-word window)
    # Optimized to ignore author-specific common repetitions to reduce noise
    words = [t for t in doc if t.is_alpha and not t.is_stop and len(t.text) > 3 and t.lemma_.lower() not in COMMON_REPETITIONS_FILTER]
    window_size = 50
    lemmas = [w.lemma_.lower() for w in words]
    
    echo_matches = defaultdict(list)
    window_positions = defaultdict(deque)
    
    for i, lemma in enumerate(lemmas):
        for prev_idx in list(window_positions[lemma]):
            if i - prev_idx <= window_size:
                echo_matches[lemma].append(words[i].text)
        window_positions[lemma].append(i)
        if len(window_positions[lemma]) > 1:
            window_positions[lemma].popleft()
    
    echoes = [(lemma, list(set(matches))) for lemma, matches in echo_matches.items()]
            
    # 1 & 3. Adverbs and Sentence Type (Combined Pass)
    adverbs = []
    types = {"Simple": 0, "Compound": 0, "Complex": 0, "Compound-Complex": 0}
    
    for sent in sents:
        # 1. Adverb Check
        sent_adverbs = [t.text for t in sent if t.is_alpha and t.pos_ == "ADV" and t.text.lower().endswith("ly")]
        if len(sent_adverbs) >= 2:
            adverbs.append({
                "text": sent.text.strip(),
                "adverbs": sent_adverbs,
                "start": sent.start_char,
                "end": sent.end_char,
                "token_idx": sent.start
            })
            
        # 3. Sentence Type Classification
        conjs = 0
        sub_clauses = 0
        for token in sent:
            if token.dep_ == "conj":
                conjs += 1
            if token.dep_ in ["advcl", "relcl", "ccomp"]:
                sub_clauses += 1
                
        if conjs > 0 and sub_clauses > 0:
            types["Compound-Complex"] += 1
        elif sub_clauses > 0:
            types["Complex"] += 1
        elif conjs > 0:
            types["Compound"] += 1
        else:
            types["Simple"] += 1
            
    total_sents = len(sents)
    type_distribution = {k: (v / total_sents * 100) if total_sents > 0 else 0 for k, v in types.items()}
    
    # 4. Dialogue Tag Variety
    bad_dialogue_tags = []
    # Using finditer on doc.text is okay for regex, but we need token context
    # Better: iterate doc and check dialogue tags
    for i, token in enumerate(doc):
        if token.text.lower() in WEAK_DIALOGUE_TAGS:
            # Check if it follows a quote
            prev = _get_previous_token(doc, i)
            if prev and (prev.text.endswith('"') or prev.text.endswith("'") or prev.text.endswith('”') or prev.text.endswith('’')):
                 bad_dialogue_tags.append({
                     "text": token.text,
                     "start": token.idx,
                     "end": token.idx + len(token.text),
                     "token_idx": token.i
                 })
            
    # 5. Cliché Detector (Curated mini-list)
    found_cliches = []
    cliche_matches = list(CLICHE_PATTERN.finditer(doc.text.lower()))
    if cliche_matches:
        # Pre-calculate token offsets for fast O(log N) lookup
        doc_offsets = [t.idx for t in doc]
        
        for m in cliche_matches:
            # Use bisect to find the token at m.start() in O(log N)
            from bisect import bisect_left
            idx = bisect_left(doc_offsets, m.start())
            token_at = doc[idx] if idx < len(doc) else None
            
            found_cliches.append({
                "start": m.start(),
                "end": m.end(),
                "token_idx": token_at.i if token_at else 0,
                "type": "cliche",
                "text": m.group(0)
            })
            
    return {
        "adverb_heavy_sentences": adverbs[:5],
        "word_echoes": echoes[:10],
        "sentence_types": type_distribution,
        "weak_dialogue_tags": bad_dialogue_tags,
        "cliches": found_cliches
    }


def get_pro_editing_report(doc: spacy.tokens.Doc) -> dict:
    """
    Runs professional-grade prose checks: Filter Verbs, Sensory VAK, Dialogue Ratio, and Vague Words.
    """
    text = doc.text
    
    # 1. Filter Verb Detector
    found_filters = []
    for token in doc:
        if token.lemma_ in FILTER_VERBS and token.pos_ == "VERB":
            subjects = [c for c in token.children if c.dep_ in ["nsubj", "nsubjpass"]]
            if subjects:
                context_start = max(0, token.i - 2)
                context_end = min(len(doc), token.i + 4)
                context = doc[context_start:context_end].text.strip()
                found_filters.append({
                    "text": context,
                    "start": doc[context_start].idx,
                    "end": doc[context_end-1].idx + len(doc[context_end-1].text),
                    "token_idx": token.i
                })
                
    # 2. Sensory Analysis (VAK)
    sensory_counts = {k: 0 for k in SENSORY_KEYWORDS.keys()}
    for token in doc:
        if not token.is_alpha: continue
        lemma = token.lemma_.lower()
        s_type = SENSORY_LOOKUP.get(lemma)
        if s_type:
            sensory_counts[s_type] += 1
                
    # 3. Dialogue-to-Narrative Ratio
    all_quotes = DIALOGUE_QUOTE_PATTERN.findall(text)
    dialogue_chars = sum(len(q) for q in all_quotes)
    total_chars = len(text)
    dialogue_ratio = (dialogue_chars / total_chars * 100) if total_chars > 0 else 0
    
    # 4. Vague Word Highlighter
    found_vague = []
    for token in doc:
        if token.text.lower() in VAGUE_WORDS:
            context_start = max(0, token.i - 1)
            context_end = min(len(doc), token.i + 2)
            found_vague.append({
                "text": doc[context_start:context_end].text.strip(),
                "start": doc[context_start].idx,
                "end": doc[context_end-1].idx + len(doc[context_end-1].text),
                "token_idx": token.i
            })
            
    return {
        "filter_verbs": found_filters[:8],
        "sensory_balance": sensory_counts,
        "dialogue_ratio": dialogue_ratio,
        "vague_words": found_vague[:10]
    }

if __name__ == "__main__":
    # Test cases
    samples = [
        "the dog wagged it's tail.",
        "its going to be a good day.",
        "eating me Can I go",
        "Funny thing is I wanted to fly.",
        "I run fast. And I jump.", # Short sentence before And
        "This is a very long sentence that has many words and checks if the split works well. And it does."
    ]
    for s in samples:
        print(f"Original: {s}")
        print(f"Refined:  {refine_creative_text(s)}")
        print("-" * 20)

def detect_dangling_participles(doc: spacy.tokens.Doc) -> list[dict]:
    """
    Detects potential dangling participles where an introductory -ing phrase
    implied subject conflicts with the main clause subject.
    """
    dangling_candidates = []
    
    for sent in doc.sents:
        # Check if the sentence starts with an advcl or acl (modifier clause)
        # We look at the first few tokens
        if len(sent) < 3: continue
        
        first_token = sent[0]
        
        # Structure: [Walking] down the street, [I] saw...
        # 'Walking' is usually the head of the advcl
        # Sometimes 'Walking' is dep_ = 'advcl' attached to 'saw' (ROOT)
        
        # Case 1: First token is a verb ending in 'ing' (VBG)
        # and it is a dependent clause of the root
        if first_token.tag_ == "VBG" and first_token.dep_ in ["advcl", "acl", "xcomp"]:
            # Check if it has an explicit subject
            has_explicit_subj = any(c.dep_ in ["nsubj", "nsubjpass", "csubj"] for c in first_token.children)
            
            if not has_explicit_subj:
                # The implied subject is the subject of the main clause (first_token.head)
                main_verb = first_token.head
                
                # If main verb is ROOT or close to it
                if main_verb.pos_ in ["VERB", "AUX"]:
                    main_subjects = [c for c in main_verb.children if c.dep_ in ["nsubj", "nsubjpass", "csubj"]]
                    
                    if main_subjects:
                        subj_text = main_subjects[0].text
                        # Heuristic: If the subject is inanimate, flag it?
                        # Or simply flag all implicit intro clauses for review?
                        # The user asked: "detect this by checking if the ROOT ... and advcl share a logical subject"
                        # Since we can't easily check "logic" (AI knowledge), we flag the structure 
                        # and show the user: "[Walking] ... attached to subject [Trees]"
                        
                        clause_span = doc[first_token.left_edge.i : first_token.right_edge.i+1]
                        # Limit span length for display
                        snippet = clause_span.text
                        if len(snippet) > 40: snippet = snippet[:40] + "..."
                        
                        dangling_candidates.append({
                            "clause": snippet,
                            "main_subject": subj_text,
                            "sentence": sent.text.strip(),
                            "start": first_token.idx,
                            "end": first_token.idx + len(first_token.text),
                            "token_idx": first_token.i
                        })
                        
    return dangling_candidates

def get_tense_consistency_report(doc: spacy.tokens.Doc) -> list[dict]:
    """
    Checks for accidental tense shifting between Past (VBD) and Present (VBP/VBZ).
    Returns list of paragraphs with suspicious splits (e.g. 90/10).
    """
    text = doc.text
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    issues = []
    
    for i, para in enumerate(paragraphs):
        p_doc = get_spacy_doc(para)
        past_count = 0
        present_count = 0
        
        for token in p_doc:
            if token.tag_ == "VBD":
                past_count += 1
            elif token.tag_ in ["VBP", "VBZ"]:
                present_count += 1
                
        total = past_count + present_count
        if total < 5: continue # Skip short paragraphs
        
        ratio_past = past_count / total
        ratio_present = present_count / total
        
        # Check for 90/10 split (but not 100/0)
        flag = False
        dominant = ""
        minority = ""
        
        if 0.9 <= ratio_past < 1.0:
            flag = True
            dominant = "Past"
            minority = "Present"
        elif 0.9 <= ratio_present < 1.0:
            flag = True
            dominant = "Present"
            minority = "Past"
            
        if flag:
            issues.append({
                "paragraph_idx": i + 1,
                "dominant": dominant,
                "minority_count": present_count if dominant == "Past" else past_count,
                "snippet": para[:60] + "..."
            })
            
    return issues

def get_nominalization_report(doc: spacy.tokens.Doc) -> list[str]:
    """
    Detects 'Zombie Nouns' (Nominalizations) checks for -ion, -ment, -ance, -ence
    followed by weak verbs (be, make, give, have).
    """
    zombies = []
    
    for i, token in enumerate(doc):
        if token.pos_ == "NOUN" and token.text.lower().endswith(NOMINALIZATION_SUFFIXES):
            is_zombie = False
            
            if token.dep_ in ["nsubj", "nsubjpass"]:
                head = token.head
                if head.lemma_ in WEAK_VERBS:
                    is_zombie = True
                    
            if token.dep_ == "dobj":
                head = token.head
                if head.lemma_ in WEAK_VERBS:
                    is_zombie = True
            
            if is_zombie:
                zombies.append({
                    "text": f"{token.text} + {token.head.text}",
                    "start": token.idx,
                    "end": token.head.idx + len(token.head.text),
                    "token_idx": token.i
                })
                
    return zombies # Removed set() since we want specific instances

def get_show_dont_tell_report(doc: spacy.tokens.Doc) -> list[dict]:
    """
    Flags sentences that use "telling" structures:
    1. Linking Verb + Emotion Adjective.
    2. Filter Verbs (distanced phrasing like "He saw", "He heard")
    """
    telling_examples = []
    
    for token in doc:
        # 1. Emotional Telling (was angry, felt sad)
        if token.lemma_ in LINKING_VERBS:
            for child in token.children:
                if child.dep_ in ["acomp", "attr"] and child.pos_ == "ADJ":
                    if child.lemma_.lower() in EMOTION_ADJECTIVES:
                        context_start = max(0, token.i - 3)
                        context_end = min(len(doc), child.i + 2)
                        
                        subjs = [c.text for c in token.children if c.dep_ in ["nsubj", "nsubjpass"]]
                        subj_str = subjs[0] if subjs else "Someone"
                        
                        telling_examples.append({
                            "phrase": doc[context_start:context_end].text.strip(),
                            "emotion": child.text,
                            "type": "emotion",
                            "subject": subj_str,
                            "start": token.idx,
                            "end": child.idx + len(child.text),
                            "token_idx": token.i
                        })
        
        # 2. Filter Verbs (saw, heard, noticed)
        if token.lemma_.lower() in FILTER_VERBS and token.pos_ == "VERB":
             # Ensure there is a subject and some object/clause being filtered
             has_subj = any(c.dep_.startswith("nsubj") for c in token.children)
             has_content = any(c.dep_ in ["dobj", "ccomp", "xcomp", "advcl"] for c in token.children)
             
             if has_subj and has_content:
                telling_examples.append({
                    "phrase": token.text,
                    "type": "filter",
                    "subject": next((c.text for c in token.children if c.dep_.startswith("nsubj")), "Someone"),
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "token_idx": token.i
                })
                        
    return telling_examples

def get_adverbial_dialogue_tags(doc: spacy.tokens.Doc) -> list[str]:
    """
    Flags dialogue tags that rely on adverbs (e.g., "said angrily").
    """
    adverbial_tags = []
    
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ in SPEECH_VERBS:
            for child in token.children:
                if child.dep_ == "advmod" and child.text.lower().endswith("ly"):
                    adverbial_tags.append({
                        "text": f"{token.text} {child.text}",
                        "start": token.idx,
                        "end": child.idx + len(child.text),
                        "token_idx": token.i
                    })
                    
    return adverbial_tags

def get_paragraph_rhythm(doc: spacy.tokens.Doc) -> list[dict]:
    """
    Consolidated implementation using the existing doc tokens.
    Identifies paragraphs with extremely low variety in sentence length.
    """
    # Reuse the previously captured heatmap data but filter for 'static' paragraphs
    all_data = get_all_paragraph_rhythm_data(doc)
    static_paragraphs = []
    
    for p in all_data:
        # Standard threshold for "static" rhythm
        if len(p["sentences"]) >= 3 and p["std_dev"] < 3.5:
             static_paragraphs.append(p)
             
    return static_paragraphs

def get_specificity_ratio(doc: spacy.tokens.Doc) -> dict:
    """
    Calculates ratio of Proper Nouns to Common Nouns.
    Higher is generally more specific/concrete.
    """
    propn_count = sum(1 for token in doc if token.pos_ == "PROPN")
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
            
    total = propn_count + noun_count
    ratio = (propn_count / total * 100) if total > 0 else 0
    
    return {
        "ratio": ratio,
        "propn_count": propn_count,
        "noun_count": noun_count,
        "rating": "Vague" if ratio < 5 else ("Concrete" if ratio > 15 else "Balanced")
    }
