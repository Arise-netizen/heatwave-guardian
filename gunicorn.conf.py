import os

# Gunicorn configuration for Render, Railway, and Linux cloud deployments
# Ensures automatic binding to 0.0.0.0:$PORT provided by Render/cloud environments
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
threads = int(os.environ.get("PYTHON_MAX_THREADS", "2"))
timeout = int(os.environ.get("WEB_TIMEOUT", "120"))
accesslog = "-"
errorlog = "-"
loglevel = "info"
