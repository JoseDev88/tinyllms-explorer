from pydantic import BaseModel
from datetime import datetime

class ModelOut(BaseModel):
    id: int
    name: str
    repo_url: str
    license: str
    task_type: str | None = None
    params_b: float | None = None
    quantizations: str | None = None
    size_mb: int | None = None
    memory_min_gb: float | None = None
    speed_cpu_tok_s: float | None = None
    speed_gpu_tok_s: float | None = None
    bench_mmlu: float | None = None
    bench_gsm8k: float | None = None
    hardware_target: str | None = None
    tags: str | None = None
    updated_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True

class PageEnvelope(BaseModel):
    items: list[ModelOut]
    total: int
    page: int
    page_size: int
    next_url: str | None = None
    prev_url: str | None = None
