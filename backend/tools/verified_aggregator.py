# -*- coding: utf-8 -*-
"""
Verified-only Tiny LLM Aggregator (rate-limit friendly)
- Curated HF orgs; text-generation only
- Pre-filters by params inferred from model id (skip big ids without API calls)
- Hydrates only likely-≤4B ids; collects verified benchmarks from HF card/README; optional leaderboard
- 429-safe with exponential backoff + jitter; request budget per pass
- Resumeable disk cache + periodic checkpoints
"""
from __future__ import annotations
import os, re, json, time, math, random
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from huggingface_hub import HfApi, ModelInfo
    from huggingface_hub.errors import HfHubHTTPError
except Exception:
    HfApi = None
    ModelInfo = object  # type: ignore
    HfHubHTTPError = Exception  # type: ignore


# ------------------ Config ------------------

DEFAULT_ORGS = [
    "meta-llama","google","microsoft","Qwen","ibm-granite","apple",
    "allenai","HuggingFaceTB","mistralai","stabilityai","TinyLlama",
    "openaccess-ai-collective","tiiuae","EleutherAI"
]

TEXTGEN_TAGS = {"text-generation", "textgeneration", "causal-lm", "causallm"}

# quant hints
BPP_MAP = {"q2_k":0.25,"q3_k":0.375,"q4_k":0.50,"q5_k":0.625,"q6_k":0.75,"q8_k":1.0,
           "int4":0.50,"int8":1.0,"gguf":0.5,"fp16":2.0,"bf16":2.0}
QUANT_RE = re.compile(r"(q[234568]_k(?:_[a-z])?|int[48]|fp16|bf16|gguf)", re.I)

# metrics we accept from official sources (HF card/README; leaderboard optional)
METRIC_KEYS = {"mmlu","gsm8k","hellaswag","arc","truthfulqa","winogrande","mt-bench","mtbench","bbh"}

# optional: open-llm-leaderboard API (can be flaky; safe to skip)
LEADERBOARD_API = "https://huggingface.co/api/spaces/HuggingFaceH4/open_llm_leaderboard/rows?full=true"

# ------------------ Utils ------------------

def env(name, default=None):
    v = os.getenv(name)
    return v if v not in ("", None) else default

def load_json(path: Path, default=None):
    if not path.exists(): return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def is_textgen(info: ModelInfo) -> bool:
    t = set([(getattr(info,"pipeline_tag",None) or "").lower()])
    tags = set([(x or "").lower() for x in getattr(info,"tags",[])])
    return bool(TEXTGEN_TAGS & (t | tags))

def parse_params_from_id(model_id: str) -> Optional[float]:
    """
    Infer params from id tail: '...-3B', '...-1.1B', '...-560m', '...-270M'.
    Very conservative; returns None if not obvious.
    """
    tail = model_id.split("/")[-1].lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", tail)
    if m: return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", tail)
    if m: return round(float(m.group(1)) / 1000.0, 3)
    return None

def quant_hints_from_names(names: List[str]) -> List[str]:
    out=set()
    for n in names:
        for m in QUANT_RE.finditer(n or ""):
            out.add(m.group(1).lower())
    return sorted(out)

def estimate_params_from_files(total_bytes: int, quant_hints: List[str], prefer_fp: bool=False) -> Optional[float]:
    if total_bytes <= 0: return None
    if prefer_fp:
        bpp = 2.0
    else:
        bpps = [BPP_MAP[q] for q in quant_hints if q in BPP_MAP]
        bpp = max(bpps) if bpps else 1.0
    return round(total_bytes / bpp / 1_000_000_000, 3)

def derive_memory(params_b: Optional[float], quant_str: str) -> Optional[float]:
    if params_b is None: return None
    q = (quant_str or "").lower()
    bpp = 2.0
    if "int8" in q or "q8_k" in q: bpp = 1.0
    if any(x in q for x in ("int4","q4_k","gguf")): bpp = 0.5
    size_mb = params_b * 1_000_000_000 * bpp / (1024*1024)
    return max(2.0, round((size_mb/1024.0)*1.25, 1))

def parse_metrics_from_text(text: str, repo_url: str) -> Dict[str, Dict[str,Any]]:
    found={}
    for line in (text or "").splitlines():
        l=line.lower()
        if not any(k in l for k in METRIC_KEYS): continue
        # try to capture a number near the metric
        m_pct = re.search(r"(\d{1,2}(?:\.\d{1,2})?)\s*%", line)
        m_num = re.search(r"(?<!\d)(\d{1,3}(?:\.\d{1,3})?)(?!\d)", line)
        metric=None
        for k in METRIC_KEYS:
            if k in l:
                metric = "mt-bench" if k in ("mtbench","mt-bench") else k
                break
        if metric:
            if m_pct:
                found[metric] = {"value": float(m_pct.group(1)), "unit":"%", "source_url": repo_url, "label":"hf_readme"}
            elif m_num:
                found[metric] = {"value": float(m_num.group(1)), "unit":"", "source_url": repo_url, "label":"hf_readme"}
    return found

def backoff_sleep(attempt: int, base: float = 1.5, cap: float = 60.0):
    # exponential backoff with jitter; attempt starts at 1
    t = min(cap, (base ** attempt)) * (1.0 + random.random()*0.25)
    time.sleep(t)

def hf_api() -> HfApi:
    if HfApi is None:
        raise SystemExit("Missing dependency: pip install huggingface_hub")
    tok = env("HF_TOKEN") or env("HUGGINGFACE_HUB_TOKEN")
    return HfApi(token=tok)

def fetch_leaderboard(verbose=False) -> Dict[str, Dict[str,Any]]:
    out={}
    try:
        r=requests.get(LEADERBOARD_API, timeout=25)
        if r.ok:
            data = r.json()
            rows = data.get("rows") or data
            for row in rows or []:
                rec = row if isinstance(row, dict) else (row[0]["data"] if isinstance(row, list) and row and isinstance(row[0], dict) and "data" in row[0] else None)
                if not rec: continue
                model = rec.get("model") or rec.get("Model") or rec.get("model_name")
                if not model: continue
                metrics={}
                for k,v in rec.items():
                    kl=str(k).lower()
                    if "mmlu" in kl or kl in ("gsm8k","hellaswag","arc","truthfulqa","winogrande","mt-bench","mtbench"):
                        try: fv=float(v)
                        except: continue
                        mk = "mmlu" if "mmlu" in kl else kl.replace("mtbench","mt-bench")
                        metrics[mk]={"value":fv,"unit":"%","source_url":LEADERBOARD_API,"label":"open_llm_leaderboard"}
                if metrics:
                    out[str(model)] = metrics
        elif verbose:
            print(f"[leaderboard] http {r.status_code}")
    except Exception as e:
        if verbose: print(f"[leaderboard] skip: {e}")
    return out

# ------------------ Core ------------------

def main():
    import argparse
    if load_dotenv:
        load_dotenv(Path(__file__).with_name(".env"))

    ap = argparse.ArgumentParser("Verified-only tiny LLM aggregator (429-safe)")
    ap.add_argument("--out", type=str, default=str(Path(__file__).parents[1]/"data"/"seeds.verified.json"))
    ap.add_argument("--cache", type=str, default=str(Path(__file__).parents[1]/"data"/".verified_cache.json"))
    ap.add_argument("--max-b", type=float, default=4.0)
    ap.add_argument("--orgs", type=str, nargs="*", default=DEFAULT_ORGS)
    ap.add_argument("--limit-per-org", type=int, default=300)
    ap.add_argument("--require-bench", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--budget", type=int, default=850, help="Max model_info calls this run (stay <1000/5min)")
    ap.add_argument("--resume", action="store_true", help="Use cache to skip processed ids")
    args = ap.parse_args()

    api = hf_api()
    print(f"[env] HF_TOKEN loaded: {'yes' if env('HF_TOKEN') or env('HUGGINGFACE_HUB_TOKEN') else 'NO'}")

    # optional leaderboard
    leaderboard = fetch_leaderboard(verbose=args.verbose)
    if args.verbose:
        print(f"[leaderboard] entries: {len(leaderboard)}")

    # load cache
    cache_path = Path(args.cache)
    cache = load_json(cache_path, default={"kept":{}, "skipped":[], "seen":[], "ts":None})
    kept_cache: Dict[str,Any] = cache.get("kept",{})
    skipped_cache: Set[str] = set(cache.get("skipped",[]))
    seen_cache: Set[str] = set(cache.get("seen",[]))

    # discover (cheap; 1 call per org)
    infos: List[ModelInfo] = []
    for org in args.orgs:
        try:
            it = api.list_models(author=org, filter="text-generation", sort="downloads", direction=-1, limit=args.limit_per_org)
            batch = list(it)
            if args.verbose: print(f"[discover] {org}: {len(batch)}")
            infos.extend(batch)
        except Exception as e:
            if args.verbose: print(f"[discover] {org} -> {e}")

    # dedupe
    seen=set()
    uniq=[]
    for m in infos:
        mid=getattr(m,"id",None)
        if not mid or mid in seen: continue
        seen.add(mid); uniq.append(m)
    if args.verbose: print(f"[discover] total unique: {len(uniq)}")

    outputs=[]
    kept_count=0
    budget_left = args.budget
    attempts=0

    for i, info in enumerate(uniq, 1):
        mid = getattr(info,"id","")
        if not mid: continue

        # resume / cache
        if args.resume and (mid in kept_cache or mid in skipped_cache):
            if mid in kept_cache:
                outputs.append(kept_cache[mid]); kept_count += 1
            continue
        if mid in seen_cache:
            continue
        seen_cache.add(mid)

        # fast gates WITHOUT API calls:
        # (1) must be text-gen at discovery level; (2) infer size from id and drop > max-b
        size_guess = parse_params_from_id(mid)
        if size_guess is not None and size_guess > args.max_b:
            skipped_cache.add(mid)
            continue

        # budget guard
        if budget_left <= 0:
            if args.verbose: print("[budget] exhausted; stopping this run")
            break

        # 429-safe model_info pull with backoff
        attempt=0
        while True:
            try:
                full = api.model_info(mid)
                break
            except HfHubHTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    attempt += 1
                    if args.verbose: print(f"[rate-limit] 429 on {mid}; backoff attempt {attempt}")
                    backoff_sleep(attempt, base=1.7, cap=65)
                    continue
                else:
                    if args.verbose: print(f"[warn] model_info({mid}) -> {e}")
                    skipped_cache.add(mid)
                    full=None
                    break
            except Exception as e:
                if args.verbose: print(f"[warn] model_info({mid}) -> {e}")
                skipped_cache.add(mid)
                full=None
                break

        if full is None:
            continue

        budget_left -= 1

        # Final text-gen check (sometimes discovery lies)
        if not is_textgen(full):
            skipped_cache.add(mid)
            continue

        # license
        license_id = getattr(full, "license", None)
        if not license_id:
            for t in getattr(full, "tags", []) or []:
                tl = (t or "").lower()
                if tl.startswith("license:"):
                    license_id = t.split(":",1)[1]
                    break

        # siblings & README
        siblings = getattr(full, "siblings", []) or []
        card_url = f"https://huggingface.co/{mid}"
        card_data = getattr(full, "card_data", {}) or {}

        readme_text = None
        try:
            raw = api.get_repo_file_content(mid, "README.md", repo_type="model")
            readme_text = raw.decode("utf-8", errors="ignore")
        except Exception:
            pass

        # params_b
        params_b = None
        for key in ("params","parameters","params_b"):
            if key in card_data and isinstance(card_data[key], (int,float,str)):
                try:
                    params_b = float(str(card_data[key]).lower().replace("b",""))
                    break
                except: pass
        if params_b is None:
            total_st = sum(int(getattr(s,"size",0) or 0) for s in siblings if str(getattr(s,"rfilename","")).lower().endswith(".safetensors"))
            total_gg = sum(int(getattr(s,"size",0) or 0) for s in siblings if str(getattr(s,"rfilename","")).lower().endswith(".gguf"))
            qh = quant_hints_from_names([getattr(s,"rfilename","") for s in siblings])
            if total_st>0: params_b = estimate_params_from_files(total_st, [], prefer_fp=True)
            elif total_gg>0: params_b = estimate_params_from_files(total_gg, qh, prefer_fp=False)
            if params_b is None:
                params_b = size_guess  # last resort

        if params_b is None or params_b > args.max_b:
            skipped_cache.add(mid)
            continue

        # quantizations/artifacts
        quants = set(quant_hints_from_names([getattr(s,"rfilename","") for s in siblings]))
        if any(str(getattr(s,"rfilename","")).lower().endswith(".gguf") for s in siblings):
            quants.add("gguf")
        artifacts=[]
        for s in siblings:
            name=getattr(s,"rfilename",None) or ""
            size=getattr(s,"size",None)
            if size is None: continue
            nl=name.lower()
            if nl.endswith(".safetensors") or nl.endswith(".gguf"):
                artifacts.append({"filename": name, "size_bytes": int(size), "kind":"weights"})

        quant_str = ",".join(sorted(quants)) if quants else None
        mem_gb = derive_memory(params_b, quant_str or "")

        # benchmarks: HF card_data, README parse, (optional) leaderboard
        benches = {}
        # card_data exact numbers
        for k,v in (card_data.items() if isinstance(card_data, dict) else []):
            lk=str(k).lower()
            if lk in METRIC_KEYS and isinstance(v,(int,float)):
                benches[lk]={"value": float(v), "unit":"%", "source_url": card_url, "label":"hf_card"}
        # README
        if readme_text and not benches:
            parsed = parse_metrics_from_text(readme_text, card_url)
            benches.update({k:v for k,v in parsed.items() if k in METRIC_KEYS})
        # leaderboard (if available)
        # NOTE: we do exact-id match only to avoid false positives
        # (leaderboard payloads often use plain names; keep strict by default)
        # if mid in leaderboard:
        #     for mk, mv in leaderboard[mid].items():
        #         benches.setdefault(mk, mv)

        if args.require_bench and not benches:
            skipped_cache.add(mid)
            continue

        if not license_id:
            skipped_cache.add(mid)
            continue

        out = {
            "name": mid.split("/")[-1],
            "display_name": mid.split("/")[-1].replace("-"," ").replace("_"," "),
            "source_provider": "huggingface",
            "repo_url": card_url,
            "license": license_id,
            "params_b": round(params_b,3) if params_b is not None else None,
            "quantizations": quant_str,
            "memory_min_gb": mem_gb,
            "context_max": card_data.get("context_length") or card_data.get("max_position_embeddings"),
            "artifacts": artifacts,
            "benchmarks": benches,
            "provenance": {
                "from": "hf_card+readme",
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "org": mid.split("/")[0],
            }
        }
        outputs.append(out)
        kept_count += 1
        kept_cache[mid]=out

        # checkpoint
        if kept_count % args.checkpoint_every == 0:
            save_json(Path(args.out), outputs)
            save_json(cache_path, {"kept": kept_cache, "skipped": list(skipped_cache), "seen": list(seen_cache), "ts": time.time()})
            if args.verbose: print(f"[checkpoint] kept={kept_count} written")

        # soft throttle when nearing budget
        if budget_left < 20 and args.verbose:
            print(f"[budget] nearly exhausted ({budget_left} left)")

    # final write
    save_json(Path(args.out), outputs)
    save_json(cache_path, {"kept": kept_cache, "skipped": list(skipped_cache), "seen": list(seen_cache), "ts": time.time()})
    print(f"[out] wrote {len(outputs)} models -> {args.out}")
    print("[done]")

if __name__ == "__main__":
    main()
