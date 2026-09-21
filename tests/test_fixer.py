"""
Regression tests for the grammar auto-fix path (fix_prose and its layers).

Each fix rule gets a positive case (the error is corrected) and, where a
false positive is plausible, a guard case (correct text is left alone).
"""
import pytest

import linter
from linter import fix_prose, fix_mechanical_errors, fix_comma_splices
from text_refiner import get_spacy_doc, detect_comma_splices


# --------------------------------------------------------------------------- #
# Mechanical layer (pure regex — cheap, no spaCy needed)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("broken,expected", [
    ("I dont know.", "I don't know."),
    ("She didnt care.", "She didn't care."),
    ("We cant stop.", "We can't stop."),
    ("He wont listen.", "He won't listen."),
    ("Theyre gone.", "They're gone."),
    ("Youre right.", "You're right."),
    ("im tired.", "I'm tired."),
    ("Ive seen it.", "I've seen it."),
    ("Thats life.", "That's life."),
    ("Whos there?", "Who's there?"),
    ("I shouldve known.", "I should've known."),
])
def test_contractions(broken, expected):
    assert fix_mechanical_errors(broken) == expected


@pytest.mark.parametrize("broken,expected", [
    ("I could of won.", "I could have won."),
    ("He should of known.", "He should have known."),
    ("They might of left.", "They might have left."),
    ("She must of forgot.", "She must have forgot."),
])
def test_modal_of(broken, expected):
    assert fix_mechanical_errors(broken) == expected


def test_modal_of_leaves_normal_of_alone():
    assert fix_mechanical_errors("The best of times.") == "The best of times."


@pytest.mark.parametrize("broken,expected", [
    ("Their is a problem.", "There is a problem."),
    ("Their are many reasons.", "There are many reasons."),
])
def test_their_before_be_verb(broken, expected):
    assert fix_mechanical_errors(broken) == expected


def test_their_possessive_untouched():
    text = "Their house is red. They lost their keys."
    assert fix_mechanical_errors(text) == text


@pytest.mark.parametrize("broken,expected", [
    ("Your going to love this.", "You're going to love this."),
    ("Your kidding me.", "You're kidding me."),
    ("Your not serious.", "You're not serious."),
])
def test_your_youre_narrow(broken, expected):
    assert fix_mechanical_errors(broken) == expected


def test_your_possessive_untouched():
    # "your running shoes" is why the rule is deliberately narrow.
    text = "Grab your running shoes and your coat."
    assert fix_mechanical_errors(text) == text


@pytest.mark.parametrize("broken,expected", [
    ("He ate a apple.", "He ate an apple."),
    ("She has an book.", "She has a book."),
    ("It was a honest mistake.", "It was an honest mistake."),
    ("a umbrella.", "an umbrella."),
])
def test_article_agreement(broken, expected):
    assert fix_mechanical_errors(broken) == expected


def test_article_sound_exceptions_untouched():
    text = "He attended a university for an hour. A useful tool. A one-eyed cat. An heir."
    assert fix_mechanical_errors(text) == text


@pytest.mark.parametrize("broken,expected", [
    ("He went to the the store.", "He went to the store."),
    ("She saw a a dog.", "She saw a dog."),
    ("A cup of of tea.", "A cup of tea."),
])
def test_doubled_function_words(broken, expected):
    assert fix_mechanical_errors(broken) == expected


def test_legitimate_doubles_untouched():
    text = "He had had enough. I know that that is true. He turned it in in March."
    assert fix_mechanical_errors(text) == text


def test_lone_i_capitalized():
    assert fix_mechanical_errors("she said i should go, and i agreed.") == \
        "she said I should go, and I agreed."


def test_lone_i_protects_ie():
    assert fix_mechanical_errors("i.e. this stays.") == "i.e. this stays."


# --------------------------------------------------------------------------- #
# NLP layer (spaCy-dependent)
# --------------------------------------------------------------------------- #

def test_then_than_comparative():
    out = fix_prose("She is taller then me. He has more then ten.")
    assert "taller than me" in out
    assert "more than ten" in out


def test_then_temporal_untouched():
    text = "Things were better then. We lived there back then."
    assert fix_prose(text) == text


def test_its_confusion_both_directions():
    out = fix_prose("It's tail was wagging. Its going to rain.")
    assert "Its tail was wagging." in out
    assert "It's going to rain." in out


def test_dialogue_tag_punctuation():
    assert fix_prose('"Hello." he said.') == '"Hello," he said.'


def test_capitalization_after_quote_distinguishes_tags_from_actions():
    # Dialogue tags after ?" / !" stay lowercase; new action sentences get
    # capitalized.
    assert fix_prose('"Ready?" he asked.') == '"Ready?" he asked.'
    assert fix_prose('"Run!" she screamed.') == '"Run!" she screamed.'
    assert fix_prose('"Where?" she turned away.') == '"Where?" She turned away.'


# --------------------------------------------------------------------------- #
# Comma splices
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("broken,expected", [
    ("He ran fast, he was late.", "He ran fast. He was late."),
    ("She opened the door, the room was dark.",
     "She opened the door. The room was dark."),
    ("The dog barked, the cat ran.", "The dog barked. The cat ran."),
    ("It was raining, we stayed inside.", "It was raining. We stayed inside."),
    ("The plan failed, nobody was surprised.",
     "The plan failed. Nobody was surprised."),
    # Periphrastic verbs: the finite verb is an aux on a participle head.
    ("The car had stalled, the engine was dead.",
     "The car had stalled. The engine was dead."),
])
def test_comma_splice_fixed(broken, expected):
    assert fix_prose(broken) == expected


def test_comma_splice_chain():
    assert fix_prose("I came, I saw, I conquered.") == "I came. I saw. I conquered."


def test_comma_splice_with_conjunctive_adverb():
    # The splice is at the FIRST comma; "however" keeps its trailing comma.
    assert fix_prose("He was tired, however, he kept going.") == \
        "He was tired. However, he kept going."


def test_list_before_conjunctive_adverb_splice_is_not_split():
    """
    Regression: the comma after "cold" opens a list, not a clause, and must
    survive. Only the splice before "however" is promoted to a period.
    """
    assert fix_prose("She was cold, tired, and frightened, however, she refused to give up.") == \
        "She was cold, tired, and frightened. However, she refused to give up."


def test_splice_second_clause_may_use_a_speech_verb():
    """
    Regression: a broad speech-verb exemption swallowed ordinary splices whose
    second clause happens to use a verb like "answered" or "screamed".
    """
    assert fix_prose("She knocked twice, nobody answered.") == \
        "She knocked twice. Nobody answered."
    assert fix_prose("The door slammed, he screamed.") == \
        "The door slammed. He screamed."


def test_trailing_wh_word_does_not_block_detection():
    """
    Regression: "where" in "she had no idea where" is a trailing complement,
    not a subordinator — the parser sometimes attaches it straight to the
    clause head, which must not make the clause look dependent.
    """
    assert fix_prose("It must have fallen somewhere, she had no idea where.") == \
        "It must have fallen somewhere. She had no idea where."


def test_multi_sentence_paragraph_fixes_every_splice():
    """
    Regression: parse context shifts between sentences; every splice in a
    paragraph must still be found in one pass.
    """
    para = ("She checked her pocket, the key was gone. It must have fallen "
            "somewhere, she had no idea where.")
    assert fix_prose(para) == ("She checked her pocket. The key was gone. It must "
                               "have fallen somewhere. She had no idea where.")


def test_comma_heavy_correct_prose_is_untouched():
    """The safety property that matters most: correct prose is never rewritten."""
    passage = (
        'Margaret set down her cup, careful not to spill it. The tea had gone '
        'cold hours ago, but she drank it anyway, out of habit more than thirst.\n\n'
        '"You should sleep," Tomas said, leaning against the doorframe.\n\n'
        'Outside, the rain had finally stopped, and the gutters were still '
        'dripping. When the storm broke, it had taken the power with it, so they '
        'had been working by candlelight, which suited her fine.\n\n'
        'Because she had read them all twice, she knew the ending already.\n\n'
        'Tomas, who had never understood any of this, waited. He was patient, '
        'kind, and utterly baffled.'
    )
    assert detect_comma_splices(get_spacy_doc(passage)) == []
    assert fix_prose(passage) == passage


@pytest.mark.parametrize("text", [
    # Comma + coordinator is correct punctuation.
    "The rain fell, and the streets flooded.",
    "It was late, so we went home.",
    "He was tired, but he kept going.",
    # Subordinate clauses are dependent, not spliced.
    "After he left, she cried.",
    "Because it rained, we stayed in.",
    "When she arrived, the room was empty.",
    "While he slept, the storm passed.",
    "If you leave, I will follow.",
    "Although she tried, it failed.",
    # Participial phrase, list, appositive, relative clause.
    "He ran fast, breathing hard.",
    "She was tired, cold, and hungry.",
    "He grabbed his coat, his keys, his phone.",
    "Sarah, my sister, is here.",
    "The man, who was tall, left.",
    # Interjections and direct address.
    "Yes, I agree.",
    "Well, that settles it.",
    "No, he didn't.",
    "Run, John, run!",
    # Introductory phrases.
    "In the morning, she left.",
    "Tired and cold, he pressed on.",
])
def test_correct_commas_untouched(text):
    assert detect_comma_splices(get_spacy_doc(text)) == []
    assert fix_prose(text) == text


@pytest.mark.parametrize("text", [
    '"I\'m late," he said.',
    '"Stop," she whispered.',
    '"Get out," he growled.',
])
def test_dialogue_tags_are_not_splices(text):
    """
    '"I\'m late," he said.' is two independent clauses joined by a comma, but
    it is correct English — the highest-risk false positive for this rule.
    """
    assert detect_comma_splices(get_spacy_doc(text)) == []
    assert fix_prose(text) == text


def test_splice_inside_dialogue_still_fixed():
    # The splice is within the quoted speech, away from the tag boundary.
    out = fix_prose('"He ran fast, he was late," she said.')
    assert out == '"He ran fast. He was late," she said.'


def test_fix_comma_splices_is_idempotent():
    once = fix_comma_splices("He ran fast, he was late.")
    assert fix_comma_splices(once) == once


# --------------------------------------------------------------------------- #
# Spelling autocorrect
# --------------------------------------------------------------------------- #

def test_spelling_distance_one_corrected():
    out = fix_prose("Teh dog barked. He recieved a letter. She beleived him.")
    assert "The dog barked." in out
    assert "received a letter" in out
    assert "believed him" in out


def test_spelling_never_touches_names_or_acronyms():
    text = "Kaelith walked to Zhargath. NASA launched it."
    assert fix_prose(text) == text


def test_spelling_leaves_far_off_words_flagged_not_fixed():
    # "xqzvpt" has no dictionary word within one edit: must survive untouched.
    text = "The xqzvpt hummed."
    assert "xqzvpt" in fix_prose(text)


# --------------------------------------------------------------------------- #
# Full pipeline sanity
# --------------------------------------------------------------------------- #

def test_clean_text_passes_through_unchanged():
    text = ('The rain fell softly on the roof. She opened her umbrella and '
            'stepped outside. "Ready?" he asked.')
    assert fix_prose(text) == text


def test_compound_errors_all_fixed_in_one_pass():
    broken = "i dont think she could of seen a apple. Their is more then enough."
    out = fix_prose(broken)
    assert out == "I don't think she could have seen an apple. There is more than enough."
