from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, Text
from database.connection import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    google_id: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    picture: Mapped[str | None] = mapped_column(Text, nullable=True)
