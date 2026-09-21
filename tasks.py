from gevent import monkey
monkey.patch_all()

import os
import io
import contextlib
from app_factory import socketio
import linter

def run_analysis_task(text, room, model_name="en_core_web_md"):
    """
    Background worker function running in a separate thread.
    Communicates progress via SocketIO directly without a message queue.
    """
    def emit_progress(metric_name, value, status="Calculating..."):
        socketio.emit('analysis_progress', {
            'metric': metric_name,
            'value': value,
            'status': status
        }, to=room)

    # Step 0: NLP Parsing
    emit_progress("NLP Model", None, "Loading SpaCy Engine...")
    try:
        doc = linter.get_spacy_doc(text, model_name=model_name)
        sentences = [sent.text.strip() for sent in doc.sents]
        sentence_starts = linter.get_sentence_starts(doc)
        emit_progress("NLP Model", "Ready", "Parsing Complete")

        # Step 1: Readability
        emit_progress("Readability", None, "Calculating Flesch Score...")
        readability = linter.calculate_readability(text, sentences)
        emit_progress("Readability", round(readability['score'], 1), "Done")

        # Step 2: Passive Voice
        emit_progress("Directness", None, "Scanning for Passive Voice...")
        pv = linter.get_passive_voice_analysis(doc, sentence_starts)
        pv_perc = pv.get('passive_percentage', 0) if isinstance(pv, dict) else 0
        emit_progress("Directness", f"{100 - pv_perc:.1f}% Active", "Done")

        # Step 3: Tone & Intensity
        emit_progress("Sentiment", None, "Analyzing Emotional Tone...")
        tone = linter.get_tone_analysis(doc)
        sentiment_val = tone.get('sentiment_score', 0.0) if isinstance(tone, dict) else 0.0
        emit_progress("Sentiment", sentiment_val, "Done")

        # Step 4: Vocabulary
        emit_progress("Vocabulary", None, "Measuring Lexical Diversity...")
        vocab = linter.get_vocabulary_metrics(doc)
        div_ratio = vocab.get('diversity_ratio', 0) if isinstance(vocab, dict) else 0
        emit_progress("Vocabulary", f"{div_ratio}%", "Done")

        # Full Report Content
        report_data = linter.lint_prose(text, doc=doc, sentences=sentences)
        
        report_io = io.StringIO()
        with contextlib.redirect_stdout(report_io):
            linter.print_report(report_data)
        report_content = report_io.getvalue()
        
        fixed_text = linter.fix_prose(text, doc=doc)

        # Final result
        socketio.emit('analysis_complete', {
            'report': report_data,
            'report_content': report_content,
            'fixed': fixed_text,
            'original_text': text
        }, to=room)

        return "Analysis Complete"
    except Exception as e:
        print(f"Error in analysis task: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('analysis_error', {
            'message': str(e)
        }, to=room)
        return f"Error: {e}"
