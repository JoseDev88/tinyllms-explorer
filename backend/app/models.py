from sqlalchemy import Integer, String, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from .db import Base

class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    repo_url: Mapped[str] = mapped_column(String)
    license: Mapped[str] = mapped_column(String)
    task_type: Mapped[str | None] = mapped_column(String, nullable=True)
    params_b: Mapped[float | None] = mapped_column(Float, nullable=True)
    quantizations: Mapped[str | None] = mapped_column(String, nullable=True)
    size_mb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    memory_min_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed_cpu_tok_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed_gpu_tok_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    bench_mmlu: Mapped[float | None] = mapped_column(Float, nullable=True)
    bench_gsm8k: Mapped[float | None] = mapped_column(Float, nullable=True)
    hardware_target: Mapped[str | None] = mapped_column(String, nullable=True)
    tags: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
