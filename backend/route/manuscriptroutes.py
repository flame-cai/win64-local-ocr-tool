

from flask import Blueprint, jsonify
from database.connection import get_db
from model.manuscriptmodel import Manuscript

manuscript_bp = Blueprint("manuscript", __name__, url_prefix="/manuscripts")

@manuscript_bp.route("/<int:userid>", methods=["GET"])
def get_manuscripts(userid):
    db = next(get_db())
    manuscripts = db.query(Manuscript).filter(Manuscript.id == userid).all()
    db.close()

    return jsonify([
        {
            "id": m.id,
            "userid": m.userid,
            "manuscript_name": m.manuscript_name,
            "model_selected": m.model_selected,
            "created_at": str(m.created_at)
        } for m in manuscripts
    ])
