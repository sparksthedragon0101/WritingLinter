from gevent import monkey
monkey.patch_all()

import os
import io
import contextlib
import tempfile
import time
from flask import Flask, request, render_template, jsonify, send_file
from flask_socketio import join_room, emit
import linter
import exporter
from app_factory import app, socketio, executor
import tasks

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_file(os.path.join('templates', 'favicon.ico'), mimetype='image/vnd.microsoft.icon')

@app.route('/results', methods=['GET'])
def results():
    return render_template('result.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    HTTP route to queue analysis job.
    """
    text = request.form.get('text', '')
    room = request.form.get('room', '')
    model = request.form.get('model', 'en_core_web_md')

    if not text.strip():
        return jsonify({"error": "Please enter some text."}), 400

    # Queue task for background execution within Flask-Executor threads
    executor.submit(tasks.run_analysis_task, text, room, model_name=model)
    
    return jsonify({"status": "queued"}), 202

EXPORT_FORMATS = {
    'docx': (exporter.export_to_docx, '.docx',
             'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
    'markdown': (exporter.export_to_markdown, '.md', 'text/markdown'),
}

@app.route('/export/<format>', methods=['POST'])
def export_file(format):
    if format not in EXPORT_FORMATS:
        return "Invalid format", 400

    data = request.json
    text = data.get('text', '')
    report = data.get('report', {})
    report_content = data.get('report_content', '')
    include_analysis = data.get('include_analysis', True)

    if not text:
        return "No text provided", 400

    export_fn, ext, mimetype = EXPORT_FORMATS[format]
    filename = f"Writing_Linter_Report_{int(time.time())}{ext}"

    try:
        # Build the file in a private per-request temp dir (mode 0700, random
        # name), then buffer it into memory so the file is gone from disk
        # before the response is sent — nothing lingers in a shared /tmp.
        with tempfile.TemporaryDirectory(prefix="writing_linter_export_") as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            export_fn(text, report, report_content, filepath, include_analysis=include_analysis)
            with open(filepath, 'rb') as f:
                buffer = io.BytesIO(f.read())
        return send_file(buffer, as_attachment=True, download_name=filename, mimetype=mimetype)
    except Exception as e:
        print(f"[Export Error] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Export failed"}), 500

# WebSocket Events
@socketio.on('join_analysis')
def on_join(data):
    room = data['room']
    join_room(room)
    print(f"User joined room: {room}")

if __name__ == '__main__':
    print(f"--- Starting Writing Linter on http://0.0.0.0:5010 ---")
    # Using socketio.run with gevent for proper async handling
    # Disable use_reloader to avoid issues with gevent monkey patching and forking
    socketio.run(app, host='0.0.0.0', debug=True, port=5010, use_reloader=False)
