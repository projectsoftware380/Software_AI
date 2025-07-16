from flask import Flask, Response, render_template_string
from src.log_analyzer import GCPLogStreamer, LogAnalyzerConfig
import json
import threading

app = Flask(__name__)

# Configuración mínima (puedes ajustar el path del YAML si lo usas)
config = LogAnalyzerConfig.from_yaml("log_analyzer_config.yaml")
streamer = GCPLogStreamer(config)

def event_stream():
    for log in streamer.stream_logs():
        yield f"data: {json.dumps(log, ensure_ascii=False)}\n\n"

@app.route('/')
def index():
    # HTML básico con JavaScript para mostrar logs en tiempo real
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Logs en Tiempo Real (GCP)</title>
        <style>
            body { font-family: monospace; background: #181818; color: #e0e0e0; }
            #logs { white-space: pre-wrap; background: #222; padding: 1em; border-radius: 8px; max-height: 80vh; overflow-y: auto; }
        </style>
    </head>
    <body>
        <h2>Logs en Tiempo Real (GCP)</h2>
        <div id="logs"></div>
        <script>
            const logDiv = document.getElementById('logs');
            const evtSource = new EventSource('/stream');
            evtSource.onmessage = function(event) {
                const log = JSON.parse(event.data);
                const line = `[${log.timestamp}] [${log.severity}] ${log.message}`;
                logDiv.textContent += line + '\n';
                logDiv.scrollTop = logDiv.scrollHeight;
            };
        </script>
    </body>
    </html>
    ''')

@app.route('/stream')
def stream():
    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True) 