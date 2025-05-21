from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

class RecognitionLog(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    predicted_label: Mapped[str]
    confidence_score: Mapped[float]
    manuscript_name: Mapped[str]
    page: Mapped[str]
    line: Mapped[str]
    image_path: Mapped[str]
    timestamp: Mapped[datetime]

class UserAnnotationLog(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    manuscript_name: Mapped[str]
    page: Mapped[str]
    line: Mapped[str]
    ground_truth: Mapped[str]
    levenshtein_distance: Mapped[int]
    image_path: Mapped[str]
    timestamp: Mapped[datetime]