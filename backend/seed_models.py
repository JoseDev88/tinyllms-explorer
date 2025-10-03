from app.db import SessionLocal
from app.models import Model
from datetime import datetime

seed = [
    dict(name="Phi-2-2.7B", repo_url="https://huggingface.co/microsoft/phi-2", license="MIT",
         task_type="general", params_b=2.7, quantizations="int4,int8,fp16",
         size_mb=1600, memory_min_gb=6, bench_mmlu=46.0, hardware_target="CPU,M1/M2", tags="tiny,fast"),
    dict(name="Qwen2.5-3B", repo_url="https://huggingface.co/Qwen/Qwen2.5-3B", license="Apache-2.0",
         task_type="chat", params_b=3.0, quantizations="int4,int8,fp16",
         size_mb=1900, memory_min_gb=8, bench_mmlu=58.0, hardware_target="RTX 3060", tags="chat"),
    dict(name="Llama-3.2-1B", repo_url="https://huggingface.co/meta-llama/Llama-3.2-1B", license="custom",
         task_type="general", params_b=1.0, quantizations="int4,int8,fp16",
         size_mb=800, memory_min_gb=4, bench_mmlu=40.0, hardware_target="CPU,Raspberry Pi", tags="edge"),
    dict(name="Gemma-2-2B", repo_url="https://huggingface.co/google/gemma-2-2b", license="custom",
         task_type="instruct", params_b=2.0, quantizations="int4,int8,fp16",
         size_mb=1200, memory_min_gb=6, bench_mmlu=52.0, hardware_target="M1/M2", tags="instruct"),
    dict(name="TinyLlama-1.1B", repo_url="https://huggingface.co/jzhang/TinyLlama-1.1B", license="Apache-2.0",
         task_type="general", params_b=1.1, quantizations="int4,int8,fp16",
         size_mb=700, memory_min_gb=4, bench_mmlu=38.0, hardware_target="CPU", tags="tiny"),
]

def main():
    db = SessionLocal()
    try:
        for d in seed:
            m = Model(**d)
            m.created_at = m.updated_at = datetime.utcnow()
            db.add(m)
        db.commit()
        print(f"Seeded {len(seed)} models.")
    finally:
        db.close()

if __name__ == "__main__":
    main()
