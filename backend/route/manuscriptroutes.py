from flask import Blueprint, jsonify, request, current_app
from datetime import datetime
from database.connection import get_db
from model.manuscriptmodel import Manuscript, AnnotationLog
import json

import os
import shutil


manuscript_bp = Blueprint("manuscript", __name__, url_prefix="/manuscripts")

@manuscript_bp.route("/add-manuscript", methods=["POST"])
def create_manuscript():
    db = None
    try:
        data = request.json
        if not data:
            return jsonify({"message": "Invalid JSON payload"}), 400
        required_fields = ["userid", "username", "manuscript_name", "model_selected", "created_at"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"message": f"Missing required data fields: {', '.join(missing_fields)}"}), 400
        created_at_iso = data["created_at"]
        if created_at_iso.endswith("Z"):
            created_at_iso = created_at_iso[:-1]  
        try:
            created_at_dt = datetime.fromisoformat(created_at_iso)
        except ValueError:
            return jsonify({"message": "Invalid datetime format for 'created_at'. Use ISO 8601 format."}), 400
        db = next(get_db())


        manuscript = Manuscript(
        userid=data["userid"],
        username=data["username"],
        manuscript_name=data["manuscript_name"],
        model_selected=data["model_selected"],
        fileimagename = json.dumps(data["fileimagename"][::-1]) , # reverse the array of filenames
        created_at=created_at_dt
        )

        db.add(manuscript)
        db.commit()

        return jsonify({"message": "Manuscript created successfully"}), 201

    except Exception as e:
        if db:
            db.rollback()
        print(f"Error creating manuscript: {e}")
        return jsonify({"message": "Server error: Failed to process request due to internal database error."}), 500

    finally:
        if db:
            db.close()


@manuscript_bp.route("/get-manuscripts/<int:userid>", methods=["GET"])
def get_manuscripts(userid):
    db = None
    try:
        db = next(get_db())
        manuscripts = db.query(Manuscript).filter_by(userid=userid).all()

        manuscripts_list = []
        for m in manuscripts:
            manuscripts_list.append({
                "id": m.id,
                "userid": m.userid,
                "username": m.username,
                "manuscript_name": m.manuscript_name,
                "model_selected": m.model_selected,
                "fileimagename": m.fileimagename,
                "created_at": m.created_at.isoformat()  
            })

        return jsonify({"manuscripts": manuscripts_list}), 200

    except Exception as e:
        print(f"Error fetching manuscripts: {e}")
        return jsonify({"message": "Server error: Failed to fetch manuscripts."}), 500

    finally:
        if db:
            db.close()


@manuscript_bp.route("/check", methods=["GET"])
def check():
    return jsonify({"message": "Manuscript route is working"}), 200



@manuscript_bp.route("/delete-manuscript", methods=["DELETE"])
def delete_manuscript():
    db = None
    try:
        data = request.json
        if not data:
            return jsonify({"message": "Invalid JSON payload"}), 400

        required_fields = ["userid", "manuscript_name", "model_selected"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"message": f"Missing required data fields: {', '.join(missing_fields)}"}), 400

        db = next(get_db())

        manuscript = (
            db.query(Manuscript)
            .filter(
                Manuscript.userid == data["userid"],
                Manuscript.manuscript_name == data["manuscript_name"],
                Manuscript.model_selected == data["model_selected"]
            )
            .first()
        )

        if not manuscript:
            return jsonify({"message": "No matching manuscript found for deletion."}), 404

        db.delete(manuscript)
        db.commit()
        deleted_count = (
            db.query(AnnotationLog)
            .filter(
                AnnotationLog.manuscript_name == data["manuscript_name"],
                AnnotationLog.model_selected == data["model_selected"]
            )
            .delete(synchronize_session=False)
        )

        if deleted_count == 0:
            return jsonify({"message": "No matching annotation logs found for deletion."}), 404

        db.commit()

        root_directory = current_app.config.get("MANUSCRIPTS_ROOT", "instance/manuscripts")
        manuscript_folder = os.path.join(root_directory, data["manuscript_name"])

        if os.path.exists(manuscript_folder) and os.path.isdir(manuscript_folder):
            shutil.rmtree(manuscript_folder)
        else:
            print(f"Folder not found or already deleted: {manuscript_folder}")

        return jsonify({"message": "Manuscript and its folder deleted successfully."}), 200

    except Exception as e:
        if db:
            db.rollback()
        print(f"Error deleting manuscript: {e}")
        return jsonify({"message": "Server error: Failed to delete manuscript."}), 500

    finally:
        if db:
            db.close()
