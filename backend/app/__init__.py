"""Flask application factory and HTTP interface wiring."""

from flask import Flask, jsonify

from app.api.http import recognize_bp


def create_app() -> Flask:
    """Create and configure the Flask app instance."""
    app = Flask(__name__)
    app.register_blueprint(recognize_bp)

    @app.get("/health")
    def health_check():
        return jsonify({"status": "ok"}), 200

    return app
