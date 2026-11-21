"""Fake DRES v2 server; the implementation lives in backend/app/mock_dres.py.

Run standalone:
    uvicorn tools.mock_dres:app --port 8090
"""

from backend.app.mock_dres import *  # noqa: F401,F403
from backend.app.mock_dres import app  # noqa: F401
