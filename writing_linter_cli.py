#!/usr/bin/env python3
import argparse
import sys
import os
import json
import io
import hashlib
from datetime import datetime

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import linter
    import exporter
    import config as linter_config
    from text_refiner import get_spacy_doc
except ImportError as e:
    print(f"Error: Could not import core modules. {e}")
    sys.exit(1)

def compact_report(report, summary_only=False):
    """
    Translates the report into a token-optimized JSON structure.
    """
    mapping = {
        "executive_summary": "es",
        "readability": "r",
        "passive_voice": "pv",
        "tone": "t",
        "pro_editing": "pe",
        "advanced_features": "af",
        "sentence_splices": "ss",
        "comma_splices": "cs",
        "dangling_participles": "dp",
        "tense_consistency": "tc",
        "nominalizations": "nom",
        "dialogue_beats": "db",
        "redundancy": "red",
        "show_dont_tell": "sdt",
        "paragraph_rhythm": "pr",
        "rhythm_heatmap": "rh",
        "specificity": "spec",
        "adverbial_tags": "at",
        "tts_analysis": "tts"
    }
    
    compact = {}
    for key, value in report.items():
        if key in mapping:
            new_key = mapping[key]
            if summary_only and isinstance(value, dict):
                # Remove large lists or examples if summary_only is requested
                filtered_val = {k: v for k, v in value.items() if not isinstance(v, (list, tuple))}
                compact[new_key] = filtered_val
            else:
                compact[new_key] = value

    return compact

def get_report_content_string(report):
    """
    Generates a technical report string similar to how the web app does it.
    """
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        linter.print_report(report)
    return output.getvalue()

import contextlib
import tempfile

def export_or_print(export_fn, text, report, report_str, output_file, suffix):
    """
    Writes the export to output_file, or — when no output file was given —
    routes it through a private tempfile (random name, mode 0600, always
    removed) and prints it to stdout.
    """
    if output_file:
        export_fn(text, report, report_str, output_file)
        print(f"Report saved to {output_file}")
        return
    fd, temp_path = tempfile.mkstemp(prefix="writing_linter_", suffix=suffix)
    os.close(fd)
    try:
        export_fn(text, report, report_str, temp_path)
        with open(temp_path, "r", encoding="utf-8") as f:
            print(f.read())
    finally:
        os.remove(temp_path)

def main():
    parser = argparse.ArgumentParser(description="Writing Linter CLI - Professional Prose Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- LINT COMMAND ---
    lint_parser = subparsers.add_parser("lint", help="Analyze prose for style, grammar, and rhythm")
    lint_parser.add_argument("input", nargs="?", default="-", help="Input file path (use '-' for stdin)")
    
    # Analysis Modules for Lint
    module_group = lint_parser.add_argument_group("Analysis Modules")
    module_group.add_argument("-r", "--readability", action="store_true", help="Include readability analysis")
    module_group.add_argument("-p", "--passive", action="store_true", help="Include passive voice analysis")
    module_group.add_argument("-t", "--tone", action="store_true", help="Include tone/mood analysis")
    module_group.add_argument("-e", "--editing", action="store_true", help="Include professional editing checks")
    module_group.add_argument("-a", "--advanced", action="store_true", help="Include advanced prose metrics")
    module_group.add_argument("-s", "--structure", action="store_true", help="Include sentence structure warnings")
    module_group.add_argument("-c", "--creative", action="store_true", help="Include creative writing checks")
    module_group.add_argument("--rhythm", action="store_true", help="Include paragraph rhythm and heatmap")
    module_group.add_argument("--tts", action="store_true", help="Include TTS (Text-to-Speech) analysis")
    module_group.add_argument("--all", action="store_true", help="Include all analysis modules")
    
    # Output Options for Lint
    output_group = lint_parser.add_argument_group("Output Options")
    output_group.add_argument("--json", action="store_true", help="Output in JSON format (default)")
    output_group.add_argument("--html", action="store_true", help="Output in HTML format")
    output_group.add_argument("--MD", action="store_true", help="Output in Markdown format")
    output_group.add_argument("--obsidian", action="store_true", help="Output in Obsidian Markdown format")
    output_group.add_argument("--docx", action="store_true", help="Output in DOCX format")
    output_group.add_argument("--txt", action="store_true", help="Output in Plain Text format")
    output_group.add_argument("--output", "-o", help="Save output to a specific file")
    output_group.add_argument("--compact", action="store_true", default=True, help="Use token-optimized JSON")
    output_group.add_argument("--no-compact", dest="compact", action="store_false", help="Disable token-optimized JSON")
    output_group.add_argument("--summary", action="store_true", help="Output summary data only")
    output_group.add_argument("--show-suppressions", action="store_true", help="Include suppression details")
    output_group.add_argument("--vscode", action="store_true", help="Output diagnostics in VSCode-compatible JSON")

    # Watch Mode for Lint
    lint_parser.add_argument("--watch", action="store_true", help="Watch the input file for changes")
    
    # Configuration for Lint
    lint_parser.add_argument("--model", default="en_core_web_md", help="SpaCy model")
    lint_parser.add_argument("--save-pattern", help="Save a regex pattern as an accepted exception")

    # --- TTS-CHECK COMMAND ---
    tts_parser = subparsers.add_parser("tts-check", help="Analyze text for Text-to-Speech and voice performance")
    tts_parser.add_argument("input", nargs="?", default="-", help="Input file path")
    tts_parser.add_argument("--checks", default="all", help="Specific checks (phoneme,breath,attribution,fatigue,homograph,cadence) or 'all'")
    tts_parser.add_argument("--read-aloud-preview", action="store_true", help="Generate a textual preview with TTS problem markers")
    tts_parser.add_argument("--output", "-o", help="Save output to a specific file")

    # --- FIX COMMAND ---
    fix_parser = subparsers.add_parser("fix", help="Automatically refine text using NLP rules")
    fix_parser.add_argument("input", nargs="?", default="-", help="Input file path")
    fix_parser.add_argument("--output", "-o", help="Save refined text to a specific file")
    fix_parser.add_argument("--model", default="en_core_web_md", help="SpaCy model")

    # Backward compatibility and default behavior
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in ["lint", "tts-check", "fix"]:
        sys.argv.insert(1, "lint")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # Determine workspace dir for config early if we are just saving a pattern
    workspace_dir = os.path.dirname(os.path.abspath(args.input)) if args.input and args.input != "-" else None

    if getattr(args, "save_pattern", None):
        saved = linter_config.save_accepted_pattern(args.save_pattern, workspace_dir)
        if saved:
            sys.exit(0)
        else:
            sys.exit(1)

    def process_file():
        # Determine input source
        input_file = getattr(args, "input", "-")
        if input_file == "-":
            text = sys.stdin.read()
        else:
            try:
                with open(input_file, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading input file: {e}")
                sys.exit(1)
    
        if not text.strip():
            print("Error: No input text provided.")
            sys.exit(1)
    
        # --- FIX COMMAND ---
        if args.command == "fix" or getattr(args, "fix", False):
            doc = get_spacy_doc(text, model_name=args.model)
            refined_text = linter.fix_prose(text, doc=doc)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(refined_text)
            else:
                print(refined_text)
            return

        # --- TTS-CHECK COMMAND ---
        if args.command == "tts-check":
            try:
                import tts_analyzer
            except ImportError:
                print("Error: tts_analyzer module not found.")
                sys.exit(1)
            
            report = tts_analyzer.run_tts_analysis(text)
            
            if getattr(args, "read_aloud_preview", False):
                preview = tts_analyzer.generate_read_aloud_preview(text, report)
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(preview)
                else:
                    print(preview)
            else:
                # Filter checks if requested
                if args.checks != "all":
                    check_list = args.checks.split(",")
                    report = {k: v for k, v in report.items() if k in check_list or k.replace("_analysis", "") in check_list}
                
                json_out = json.dumps(report, indent=2)
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(json_out)
                else:
                    print(json_out)
            return

        # --- LINT COMMAND ---
        # Determine which modules to include
        no_modules_selected = not any([getattr(args, "readability", False), getattr(args, "passive", False), 
                                     getattr(args, "tone", False), getattr(args, "editing", False), 
                                     getattr(args, "advanced", False), getattr(args, "structure", False), 
                                     getattr(args, "creative", False), getattr(args, "rhythm", False), 
                                     getattr(args, "tts", False)])
        
        all_mode = getattr(args, "all", False) or no_modules_selected
    
        # Load Config
        workspace_dir = os.path.dirname(os.path.abspath(input_file)) if input_file != "-" else None
        cfg = linter_config.load_config(workspace_dir)
        model = getattr(args, "model", cfg.get("model", "en_core_web_md"))
    
        # Caching Integration Logic
        cache_dir = ".writing_linter_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Include every input that affects the report so the cache invalidates
        # when the model, suppression flag, or config (ignore_rules /
        # accepted_patterns) changes.
        cache_key_parts = {
            "text": text,
            "model": model,
            "show_suppressions": getattr(args, "show_suppressions", False),
            "ignore_rules": sorted(cfg.get("ignore_rules", [])),
            "accepted_patterns": sorted(cfg.get("accepted_patterns", [])),
        }
        hash_content = json.dumps(cache_key_parts, sort_keys=True)
        text_hash = hashlib.md5(hash_content.encode('utf-8')).hexdigest()
        cache_file = os.path.join(cache_dir, f"{text_hash}.json")
    
        doc = get_spacy_doc(text, model_name=model)
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                full_report = json.load(f)
        else:
            full_report = linter.lint_prose(text, doc=doc,
                                          show_suppressions=getattr(args, "show_suppressions", False),
                                          config=cfg)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(full_report, f)
        
        report_to_export = full_report
        if not all_mode:
            filtered_report = {"executive_summary": full_report["executive_summary"]}
            if getattr(args, "readability", False): filtered_report["readability"] = full_report.get("readability")
            if getattr(args, "passive", False): filtered_report["passive_voice"] = full_report.get("passive_voice")
            if getattr(args, "tone", False): filtered_report["tone"] = full_report.get("tone")
            if getattr(args, "editing", False): filtered_report["pro_editing"] = full_report.get("pro_editing")
            if getattr(args, "advanced", False): filtered_report["advanced_features"] = full_report.get("advanced_features")
            if getattr(args, "structure", False):
                for k in ["run_ons", "clipped_sentences", "sentence_splices", "comma_splices", "dangling_participles", "tense_consistency"]:
                    filtered_report[k] = full_report.get(k)
            if getattr(args, "creative", False):
                for k in ["nominalizations", "dialogue_beats", "redundancy", "show_dont_tell", "adverbial_tags"]:
                    filtered_report[k] = full_report.get(k)
            if getattr(args, "rhythm", False):
                filtered_report["paragraph_rhythm"] = full_report.get("paragraph_rhythm")
                filtered_report["rhythm_heatmap"] = full_report.get("rhythm_heatmap")
            if getattr(args, "tts", False):
                filtered_report["tts_analysis"] = full_report.get("tts_analysis")
            if getattr(args, "show_suppressions", False) and "suppressions" in full_report:
                filtered_report["suppressions"] = full_report["suppressions"]
            report_to_export = filtered_report
        
        report_to_export["original_text"] = text
        report_to_export["fixed_text"] = linter.fix_prose(text, doc=doc)
    
        # Handle Output Formats
        output_file = getattr(args, "output", None)
        if getattr(args, "docx", False):
            if not output_file:
                print("Error: --docx requires an output file path via --output or -o")
                sys.exit(1)
            report_str = get_report_content_string(report_to_export)
            exporter.export_to_docx(text, report_to_export, report_str, output_file)
            print(f"Report saved to {output_file}")
            
        elif getattr(args, "obsidian", False):
            report_str = get_report_content_string(report_to_export)
            export_or_print(exporter.export_to_obsidian, text, report_to_export, report_str, output_file, ".md")

        elif getattr(args, "MD", False):
            report_str = get_report_content_string(report_to_export)
            export_or_print(exporter.export_to_markdown, text, report_to_export, report_str, output_file, ".md")

        elif getattr(args, "txt", False):
            report_str = get_report_content_string(report_to_export)
            export_or_print(exporter.export_to_text, text, report_to_export, report_str, output_file, ".txt")

        elif getattr(args, "html", False):
            report_str = get_report_content_string(report_to_export)
            export_or_print(exporter.export_to_html, text, report_to_export, report_str, output_file, ".html")

        elif getattr(args, "vscode", False):
            diagnostics = []
            for k in ["passive_voice", "pro_editing", "advanced_features", "run_ons", "clipped_sentences", "sentence_splices", "comma_splices", "dangling_participles", "tense_consistency", "nominalizations", "dialogue_beats", "redundancy", "show_dont_tell", "adverbial_tags"]:
                if k in report_to_export:
                    for issue in report_to_export[k]:
                        if isinstance(issue, dict):
                            text_val = issue.get("text", "")
                            start = text.find(text_val) if text_val else 0
                            end = start + len(text_val) if text_val else 0
                            diagnostics.append({
                                "message": f"[{k}] {issue.get('message', issue.get('type', 'Issue found'))}",
                                "severity": "Warning",
                                "start": max(0, start),
                                "end": max(0, end)
                            })
            
            json_out = json.dumps({"diagnostics": diagnostics}, indent=2)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(json_out)
            else:
                print(json_out)
    
        else: # Default is JSON
            if args.compact:
                final_json = compact_report(report_to_export, summary_only=args.summary)
                json_out = json.dumps(final_json, separators=(',', ':'))
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(json_out)
                else:
                    print(json_out)
            else:
                json_out = exporter.export_to_json(text, report_to_export, args.output)
                if not args.output:
                    print(json_out)
    

    if getattr(args, "watch", False) and args.input and args.input != "-":
        import time
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            print("Error: watchdog package is required for --watch mode. Install with pip install watchdog")
            sys.exit(1)

        class FileChangeHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if os.path.abspath(event.src_path) == os.path.abspath(args.input):
                    print(f"\n--- File changed: {args.input} at {datetime.now().strftime('%H:%M:%S')} ---")
                    try:
                        process_file()
                    except Exception as e:
                        print(f"Error during linting: {e}")

        process_file()
        
        event_handler = FileChangeHandler()
        observer = Observer()
        watch_dir = os.path.dirname(os.path.abspath(args.input)) or "."
        observer.schedule(event_handler, path=watch_dir, recursive=False)
        observer.start()
        print(f"Watching for changes to {args.input}. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    else:
        process_file()

if __name__ == "__main__":
    main()
