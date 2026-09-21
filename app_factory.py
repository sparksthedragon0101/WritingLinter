from gevent import monkey
monkey.patch_all()

import os
from flask import Flask
from flask_socketio import SocketIO
from flask_executor import Executor

# Builds the Flask app and its extensions. Analysis runs on Flask-Executor
# worker threads (not Celery), and progress is pushed to the browser over
# SocketIO using the gevent async worker.
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'writing-linter-secret'),
    EXECUTOR_TYPE='thread',
    EXECUTOR_MAX_WORKERS=4
)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')
executor = Executor(app)
