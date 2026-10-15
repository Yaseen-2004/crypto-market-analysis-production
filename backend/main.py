"""Production-oriented read API for the crypto analysis dashboard."""
from __future__ import annotations
import os, subprocess, sys, threading
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "reports" / "coin_statistics.csv"
HISTORY = ROOT / "data" / "processed" / "crypto_analysis_dataset.csv"
OUTLIERS = ROOT / "reports" / "outlier_report.csv"
PIPELINE_LOCK = threading.Lock()
app = FastAPI(title="Crypto Market Analysis API", version="1.0.0",
              description="Historical crypto market statistics. Not financial advice.")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

def load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required: raise HTTPException(status_code=503, detail=f"Dataset not available: {path.name}. Run the analysis pipeline first.")
        return pd.DataFrame()
    try: return pd.read_csv(path)
    except Exception as exc: raise HTTPException(status_code=500, detail=f"Could not read {path.name}") from exc

def records(df: pd.DataFrame) -> list[dict]:
    frame = df.astype(object).where(pd.notna(df), None)
    return frame.to_dict(orient="records")

def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df: df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@app.get("/")
def root():
    return {"name":"Crypto Market Analysis API","version":"1.0.0","docs":"/docs","health":"/health"}

@app.get("/health")
def health():
    return {"status":"ok", "datasets":{"summary":SUMMARY.exists(),"history":HISTORY.exists(),"outliers":OUTLIERS.exists()}}

@app.get("/api/overview")
def overview():
    df = numeric(load_csv(SUMMARY), ["market_cap_rank","price","market_cap","volume_24h","five_day_return_pct","std_return_pct","outlier_count","extreme_outlier_count","price_change_24h_pct_api"])
    if df.empty: return {"coin_count":0,"outlier_count":0,"coins":[]}
    valid = df.dropna(subset=["coin_id"])
    def best(col, largest=True):
        x=valid.dropna(subset=[col])
        if x.empty: return None
        r=x.loc[x[col].idxmax() if largest else x[col].idxmin()]
        return {"coin_id":r.get("coin_id"),"name":r.get("name"),"symbol":r.get("symbol"),"value":float(r[col])}
    return {"coin_count":int(len(valid)),
      "outlier_count":int(valid.get("outlier_count",pd.Series(dtype=float)).fillna(0).sum()),
      "extreme_outlier_count":int(valid.get("extreme_outlier_count",pd.Series(dtype=float)).fillna(0).sum()),
      "highest_market_cap":best("market_cap"),"most_volatile":best("std_return_pct"),
      "best_return":best("five_day_return_pct"),"worst_return":best("five_day_return_pct",False),
      "coins":records(valid.sort_values("market_cap_rank",na_position="last"))}

@app.get("/api/coins")
def coins(search: Optional[str]=None, sort_by: str=Query("rank", pattern="^(rank|price|return|volatility|market_cap|outliers)$"),
          limit: int=Query(100,ge=1,le=500), offset:int=Query(0,ge=0)):
    df=numeric(load_csv(SUMMARY),["market_cap_rank","price","market_cap","five_day_return_pct","std_return_pct","outlier_count"])
    if search:
        term=search.strip().lower()
        df=df[df.astype(str).apply(lambda c:c.str.lower().str.contains(term,regex=False)).any(axis=1)]
    sort={"rank":"market_cap_rank","price":"price","return":"five_day_return_pct","volatility":"std_return_pct","market_cap":"market_cap","outliers":"outlier_count"}[sort_by]
    df=df.sort_values(sort,ascending=(sort=="market_cap_rank"),na_position="last")
    total=len(df)
    return {"total":total,"limit":limit,"offset":offset,"items":records(df.iloc[offset:offset+limit])}

@app.get("/api/coins/{coin_id}")
def coin_detail(coin_id: str):
    df=load_csv(SUMMARY)
    row=df[df["coin_id"].astype(str).str.lower()==coin_id.lower()]
    if row.empty: raise HTTPException(status_code=404,detail="Cryptocurrency not found")
    return records(row)[0]

@app.get("/api/coins/{coin_id}/history")
def coin_history(coin_id: str, limit:int=Query(500,ge=1,le=5000)):
    df=load_csv(HISTORY)
    df=df[df["coin_id"].astype(str).str.lower()==coin_id.lower()].copy()
    if df.empty: raise HTTPException(status_code=404,detail="No history found for this cryptocurrency")
    df=numeric(df,["price","pct_return","ma_3","ma_5","z_score","timestamp_ms"])
    if "timestamp" in df: df=df.sort_values("timestamp")
    cols=[c for c in ["timestamp","datetime_utc","price","pct_return","ma_3","ma_5","z_score","is_outlier","is_extreme_outlier"] if c in df]
    return {"coin_id":coin_id,"count":min(len(df),limit),"items":records(df[cols].tail(limit))}

@app.get("/api/outliers")
def outliers(limit:int=Query(100,ge=1,le=1000), severity:Optional[str]=None):
    df=load_csv(OUTLIERS, required=False)
    if df.empty:return {"total":0,"items":[]}
    if severity and "Severity" in df: df=df[df["Severity"].astype(str).str.lower()==severity.lower()]
    if "Date/Time (UTC)" in df: df=df.sort_values("Date/Time (UTC)",ascending=False)
    return {"total":len(df),"items":records(df.head(limit))}

def run_pipeline(top_n:int, days:int):
    if not PIPELINE_LOCK.acquire(blocking=False): return
    try:
        subprocess.run([sys.executable, str(ROOT/"main.py"), "--top-n",str(top_n),"--days",str(days),"--no-plots"],
                       cwd=ROOT, check=True, timeout=60*60, capture_output=True, text=True)
    finally: PIPELINE_LOCK.release()

@app.post("/api/refresh", status_code=202)
def refresh(background_tasks:BackgroundTasks, top_n:int=Query(20,ge=5,le=100), days:int=Query(5,ge=1,le=5),
            token:Optional[str]=Header(default=None, alias="X-Refresh-Token")):
    expected=os.getenv("REFRESH_TOKEN")
    if not expected or token != expected: raise HTTPException(status_code=403,detail="Refresh is disabled or token is invalid")
    if PIPELINE_LOCK.locked(): raise HTTPException(status_code=409,detail="An analysis run is already in progress")
    background_tasks.add_task(run_pipeline,top_n,days)
    return {"status":"accepted","message":"Analysis refresh started. Check /health and datasets shortly."}
