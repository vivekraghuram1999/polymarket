import sys
import os

# Add parent directory to path so dashboard.py can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import app

# Vercel expects a WSGI app named `app` or `application`
# Dash's underlying Flask server is the WSGI app
server = app.server
