"""Punto de entrada raíz para Render; delega en backend/web_api.py."""
from backend.web_api import app, serve_app

if __name__ == "__main__":
    serve_app()
