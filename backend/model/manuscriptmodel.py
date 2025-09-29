from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from database.connection import Base

class Manuscript(Base):
    __tablename__ = "manuscript"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    userid: Mapped[int] = mapped_column(Integer)
    username: Mapped[str] = mapped_column(String(255))
    manuscript_name: Mapped[str] = mapped_column(String(255))
    model_selected: Mapped[str] = mapped_column(String(255))
    fileimagename: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime]


class AnnotationLog(Base):
    __tablename__ = "annotation_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    manuscriptid: Mapped[int] = mapped_column(Integer)
    predicted_label: Mapped[str] = mapped_column(Text)
    confidence_score: Mapped[float]
    manuscript_name: Mapped[str] = mapped_column(String(255))
    ground_truth: Mapped[str] = mapped_column(Text,nullable=True)
    levenshtein_distance: Mapped[int] = mapped_column(Integer,nullable=True)
    page: Mapped[str] = mapped_column(String(255))
    line: Mapped[str] = mapped_column(String(255))
    image_path: Mapped[str] = mapped_column(String(255))
    timestamp: Mapped[datetime]