from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from ..db import get_db
from ..models import Model
from ..schemas import ModelOut, PageEnvelope

router = APIRouter(prefix="/models", tags=["models"])

@router.get("", response_model=PageEnvelope)
def list_models(
    request: Request,
    page: int = 1,
    page_size: int = 10,
    db: Session = Depends(get_db),
):
    page = max(page, 1)
    page_size = max(min(page_size, 100), 1)

    total = db.scalar(select(func.count()).select_from(Model)) or 0

    stmt = (
        select(Model)
        .order_by(Model.updated_at.desc(), Model.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = db.scalars(stmt).all()

    base = str(request.base_url).rstrip("/")
    path = str(request.url.path)
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
