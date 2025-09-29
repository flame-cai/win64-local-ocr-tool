

from flask import Blueprint, request, jsonify
from database.connection import get_db
from model.usermodels import User
from config import Config
import requests

user_bp = Blueprint("user", __name__)


@user_bp.route("/auth/google", methods=["POST"])
def google_auth():
    data = request.get_json()
    code = data.get("code")

    if not code:
        return jsonify({"error": "Authorization code is required"}), 400

    token_url = "https://oauth2.googleapis.com/token"
    payload = {
        "code": code,
        "client_id": Config.GOOGLE_CLIENT_ID,
        "client_secret": Config.GOOGLE_CLIENT_SECRET,
        "redirect_uri": "http://localhost:5173",  # match frontend
        "grant_type": "authorization_code",
    }

    # Exchange code for tokens
    r = requests.post(token_url, data=payload)
    if r.status_code != 200:
        return jsonify({"error": "Failed to fetch token"}), 400

    token_data = r.json()
    access_token = token_data.get("access_token")

    # Fetch user info from Google
    userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}
    userinfo = requests.get(userinfo_url, headers=headers).json()

    google_id = userinfo.get("sub")
    email = userinfo.get("email")
    username = userinfo.get("name")
    picture = userinfo.get("picture")

    db = next(get_db())
    user = db.query(User).filter_by(email=email).first()

    if not user:
        user = User(
            google_id=google_id,
            email=email,
            username=username,
            picture=picture,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    db.close()

    return jsonify({
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "picture": user.picture,
        }
    })
