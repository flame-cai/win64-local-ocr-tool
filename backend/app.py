from flask import Flask
from flask_cors import CORS
from config import Config
from database.connection import engine, Base
from route.userroutes import user_bp
from route.manuscriptroutes import manuscript_bp
from model.models import db #remvelater old db
from route.routes import bp as routes_bp


def create_app():
    app = Flask(__name__)

    app.config.from_object(Config())

    CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})

    with app.app_context():
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created successfully!")
    
    db.init_app(app)
    with app.app_context():
        db.create_all()


    app.register_blueprint(user_bp)
    app.register_blueprint(manuscript_bp)
    app.register_blueprint(routes_bp)


    @app.route("/")
    def home():
        return {"msg": "Flask + MySQL + OAuth setup working"}

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

















