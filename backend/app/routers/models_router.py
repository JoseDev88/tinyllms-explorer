from fastapi import APIRouter, Depends, Request, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from typing import List

from ..db import get_db
from ..models import Model
from ..schemas import ModelOut, PageEnvelope

# Router must be defined before using decorators
router = APIRouter(prefix="/models", tags=["models"])

# ---------- List with filters/sorting/pagination ----------
@router.get("", response_model=PageEnvelope)
def list_models(
    request: Request,
    page: int = 1,
    page_size: int = 10,
    # Filters
    q: str | None = None,
    license: str | None = None,
    task_type: str | None = None,
    hardware_target: str | None = None,
    max_params_b: float | None = Query(None, description="params_b <= this"),
    max_memory_gb: float | None = Query(None, description="memory_min_gb <= this"),
    min_bench_mmlu: float | None = None,
    # Sorting
    sort_by: str = Query("updated_at", pattern="^(params_b|bench_mmlu|name|updated_at)$"),
    sort_dir: str = Query("desc", pattern="^(asc|desc)$"),
    db: Session = Depends(get_db),
):
    page = max(page, 1)
    page_size = max(min(page_size, 100), 1)

    stmt = select(Model)

    if q:
        stmt = stmt.where(Model.name.ilike(f"%{q}%"))
    if license:
        stmt = stmt.where(Model.license == license)
    if task_type:
        stmt = stmt.where(Model.task_type == task_type)
    if hardware_target:
        stmt = stmt.where(Model.hardware_target.ilike(f"%{hardware_target}%"))
    if max_params_b is not None:
        stmt = stmt.where(Model.params_b <= max_params_b)
    if max_memory_gb is not None:
        stmt = stmt.where(Model.memory_min_gb <= max_memory_gb)
    if min_bench_mmlu is not None:
        stmt = stmt.where(Model.bench_mmlu >= min_bench_mmlu)

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0

    sort_col = {
        "params_b": Model.params_b,
        "bench_mmlu": Model.bench_mmlu,
        "name": Model.name,
        "updated_at": Model.updated_at,
    }[sort_by]
    order_expr = sort_col.asc() if sort_dir == "asc" else sort_col.desc()

    rows = db.scalars(
        stmt.order_by(order_expr, Model.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
    ).all()

    base = str(request.base_url).rstrip("/")
    path = str(request.url.path)

    # rebuild query string without "page"
    qp = [(k, v) for (k, v) in request.query_params.multi_items() if k != "page"]
    if qp:
        from urllib.parse import urlencode
        qbase = f"{base}{path}?{urlencode(qp)}&page_size={page_size}"
    else:
        qbase = f"{base}{path}?page_size={page_size}"

    next_url = f"{qbase}&page={page+1}" if (page * page_size) < total else None
    prev_url = f"{qbase}&page={page-1}" if page > 1 else None

    return PageEnvelope(
        items=[ModelOut.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        next_url=next_url,
        prev_url=prev_url,
    )

# ---------- Batch by ids (define BEFORE /{model_id}) ----------
@router.get("/batch", response_model=list[ModelOut], summary="Get multiple models by id")
def get_models_batch(ids: List[str] = Query(default=[]), db: Session = Depends(get_db)):
    """
    GET /models/batch?ids=1&ids=4
    GET /models/batch?ids=1,4
    """
    parsed: List[int] = []
    for token in ids:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.append(int(part))
            except ValueError:
                raise HTTPException(status_code=422, detail=f"Invalid id: {part}")

    if not parsed:
        raise HTTPException(status_code=422, detail="Provide at least one id via ?ids=")

    rows = db.query(Model).filter(Model.id.in_(parsed)).all()
    if not rows:
        raise HTTPException(status_code=404, detail="No models found for given ids")
    return [ModelOut.model_validate(m) for m in rows]

# ---------- Single model ----------
@router.get("/{model_id}", response_model=ModelOut)
def get_model(model_id: int, db: Session = Depends(get_db)):
    row = db.get(Model, model_id)
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelOut.model_validate(row)
