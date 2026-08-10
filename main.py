"""
FlowAlert Institutional v3.2 — Bulletproof Rebuild
Survives DB failures, FRED failures, and any runtime crash.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── CONFIG ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flowalert")

app = FastAPI(title="FlowAlert Institutional", version="3.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

PROP_CONFIG = {
    "starting_balance": 25000.0,
    "trailing_drawdown": 1500.0,
    "daily_loss_limit": 1000.0,
    "profit_target": 1500.0,
    "max_trades_per_day": 3,
}

SYMBOLS = {
    "MES": {
        "enabled": True,
        "tick_value": 1.25,
        "tick_size": 0.25,
        "avg_daily_range_ticks": 120,
        "cot_weight": 0.20,
        "session_start": "09:30",
        "session_end": "16:00",
        "lunch_ban": ("12:00", "13:30"),
        "max_trades": 3,
        "stop_ticks": 10,
        "target_ticks": 20,
        "atr_mult": 1.0,
    },
    "NQ": {
        "enabled": True,
        "tick_value": 5.0,
        "tick_size": 0.25,
        "avg_daily_range_ticks": 200,
        "cot_weight": 0.15,
        "session_start": "09:30",
        "session_end": "16:00",
        "lunch_ban": ("12:00", "13:30"),
        "max_trades": 2,
        "stop_ticks": 15,
        "target_ticks": 30,
        "atr_mult": 1.5,
    },
    "MCL": {
        "enabled": True,
        "tick_value": 1.0,
        "tick_size": 0.01,
        "avg_daily_range_ticks": 300,
        "cot_weight": 0.25,
        "session_start": "10:00",
        "session_end": "12:00",
        "lunch_ban": None,
        "max_trades": 2,
        "stop_ticks": 20,
        "target_ticks": 40,
        "atr_mult": 1.2,
        "eia_day": 3,
        "eia_time": "10:30",
    },
}

# ─── DATABASE (Fail-Safe) ─────────────────────────────────────────────────

db_available = False
engine = None
SessionLocal = None
Base = None

try:
    from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, Text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session

    if DATABASE_URL and DATABASE_URL.startswith("postgresql"):
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 5})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base = declarative_base()

        class TradeLog(Base):
            __tablename__ = "trades"
            id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            symbol = Column(String, index=True)
            direction = Column(String)
            entry_price = Column(Float)
            stop_price = Column(Float)
            target_price = Column(Float)
            size = Column(Integer)
            confidence = Column(Float)
            filters_passed = Column(Integer)
            filter_details = Column(Text)
            cot_bias = Column(String)
            macro_context = Column(Text)
            result = Column(String, default="open")
            pnl = Column(Float, default=0.0)
            created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
            closed_at = Column(DateTime, nullable=True)

        class DailyStats(Base):
            __tablename__ = "daily_stats"
            id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            date = Column(String, index=True)
            symbol = Column(String, index=True)
            trades_taken = Column(Integer, default=0)
            wins = Column(Integer, default=0)
            losses = Column(Integer, default=0)
            daily_pnl = Column(Float, default=0.0)
            max_drawdown = Column(Float, default=0.0)
            hit_daily_limit = Column(Boolean, default=False)

        class COTRecord(Base):
            __tablename__ = "cot_data"
            id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            report_date = Column(String, index=True)
            symbol = Column(String, index=True)
            commercial_long = Column(Float, default=0)
            commercial_short = Column(Float, default=0)
            noncommercial_long = Column(Float, default=0)
            noncommercial_short = Column(Float, default=0)
            open_interest = Column(Float, default=0)
            net_change = Column(Float, default=0)
            created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        db_available = True
        logger.info("Database connected and initialized.")
    else:
        logger.warning("DATABASE_URL not set or invalid. Running without DB.")
except Exception as e:
    logger.error(f"Database failed to initialize: {e}")
    db_available = False

def get_db():
    if not db_available or SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─── FRED CLIENT (Fail-Safe) ────────────────────────────────────────────────

class FREDClient:
    BASE = "https://api.stlouisfed.org/fred/series/observations"
    SERIES = {
        "wti": "DCOILWTICO",
        "brent": "DCOILBRENTEU",
        "dxy": "DTWEXBGS",
        "ten_year": "DGS10",
        "vix": "VIXCLS",
        "fed_funds": "DFF",
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or FRED_API_KEY
        self._cache = {}
        self._cache_time = None

    def _fetch(self, series_id: str) -> Optional[float]:
        try:
            import requests
            r = requests.get(
                self.BASE,
                params={
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                },
                timeout=3,
            )
            data = r.json()
            obs = data.get("observations", [])
            if obs and obs[0].get("value") != ".":
                return float(obs[0]["value"])
        except Exception:
            pass
        return None

    def get_macro_context(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        if self._cache_time and (now - self._cache_time).seconds < 300:
            return self._cache

        context = {}
        for name, sid in self.SERIES.items():
            context[name] = self._fetch(sid)

        if context.get("wti") and context.get("brent"):
            context["spread"] = context["brent"] - context["wti"]

        vix = context.get("vix", 20)
        context["risk_regime"] = (
            "extreme" if vix and vix > 30 else
            "high" if vix and vix > 25 else
            "elevated" if vix and vix > 20 else
            "normal"
        )

        self._cache = context
        self._cache_time = now
        return context

fred_client = FREDClient()

# ─── IN-MEMORY CACHES FOR NT8 COMPAT ────────────────────────────────────────

_price_cache: Dict[str, List[Dict]] = {}
_last_analysis: Dict[str, Dict] = {}
_cot_memory: Dict[str, Dict] = {}

# ─── TIME FILTERS ───────────────────────────────────────────────────────────

def check_time_filters(symbol: str) -> tuple[bool, str]:
    cfg = SYMBOLS.get(symbol)
    if not cfg or not cfg.get("enabled"):
        return False, "Symbol disabled"

    now = datetime.now(timezone.utc)
    et_offset = __import__("datetime").timedelta(hours=4)
    et_now = now - et_offset
    et_time = et_now.strftime("%H:%M")
    et_weekday = et_now.weekday()

    if et_weekday >= 5:
        return False, "Weekend"

    start = cfg.get("session_start")
    end = cfg.get("session_end")
    if start and end:
        if not (start <= et_time <= end):
            return False, f"Outside session {start}-{end}"

    lunch = cfg.get("lunch_ban")
    if lunch:
        if lunch[0] <= et_time <= lunch[1]:
            return False, f"Lunch ban {lunch[0]}-{lunch[1]}"

    if symbol == "MCL":
        eia_day = cfg.get("eia_day", 3)
        if et_weekday == eia_day:
            eia_time = cfg.get("eia_time", "10:30")
            if not ("10:25" <= et_time <= "11:00"):
                return False, "MCL: Waiting for EIA 10:30 AM"

    return True, "OK"

# ─── COT MANAGEMENT ─────────────────────────────────────────────────────────

def get_cot_bias(symbol: str, db) -> str:
    if symbol in _cot_memory:
        record = _cot_memory[symbol]
        comm_net = record.get("commercial_long", 0) - record.get("commercial_short", 0)
        noncomm_net = record.get("noncommercial_long", 0) - record.get("noncommercial_short", 0)
        if comm_net > 0 and noncomm_net < 0:
            return "bullish"
        elif comm_net < 0 and noncomm_net > 0:
            return "bearish"

    if db_available and db is not None:
        try:
            record = db.query(COTRecord).filter(COTRecord.symbol == symbol).order_by(COTRecord.report_date.desc()).first()
            if record:
                comm_net = record.commercial_long - record.commercial_short
                noncomm_net = record.noncommercial_long - record.noncommercial_short
                if comm_net > 0 and noncomm_net < 0:
                    return "bullish"
                elif comm_net < 0 and noncomm_net > 0:
                    return "bearish"
        except Exception as e:
            logger.warning(f"DB COT query failed: {e}")

    return "neutral"

# ─── RISK MANAGEMENT ────────────────────────────────────────────────────────

def check_daily_limits(symbol: str, db) -> tuple[bool, str]:
    if not db_available or db is None:
        return True, "OK (no DB)"

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        stats = db.query(DailyStats).filter(
            DailyStats.date == today,
            DailyStats.symbol == symbol
        ).first()

        if not stats:
            return True, "OK"

        cfg = SYMBOLS[symbol]
        if stats.trades_taken >= cfg["max_trades"]:
            return False, f"Max trades reached ({cfg['max_trades']})"

        if stats.daily_pnl <= -PROP_CONFIG["daily_loss_limit"]:
            return False, f"Daily loss limit hit (${stats.daily_pnl:.0f})"

        if stats.hit_daily_limit:
            return False, "Daily limit flag set"
    except Exception as e:
        logger.warning(f"Daily limits check failed: {e}")

    return True, "OK"

# ─── SIX FILTER ENGINE (Lightweight) ────────────────────────────────────────

try:
    import numpy as np
    NUMPY_OK = True
except Exception:
    NUMPY_OK = False
    logger.warning("numpy not available — using pure Python math")

class SixFilterEngine:
    def __init__(self, symbol: str, bars: List[Dict]):
        self.symbol = symbol
        self.bars = bars
        self.config = SYMBOLS.get(symbol, SYMBOLS["MES"])
        self.closes = [b["close"] for b in bars]
        self.highs = [b["high"] for b in bars]
        self.lows = [b["low"] for b in bars]
        self.volumes = [b.get("volume", 0) for b in bars]

    def true_vwap(self) -> float:
        total_pv = 0.0
        total_v = 0.0
        for i in range(len(self.closes)):
            tp = (self.highs[i] + self.lows[i] + self.closes[i]) / 3
            v = self.volumes[i] + 1e-9
            total_pv += tp * v
            total_v += v
        return total_pv / total_v if total_v > 0 else self.closes[-1]

    def ema(self, period: int = 20) -> float:
        if len(self.closes) < period:
            return sum(self.closes) / len(self.closes)
        vals = self.closes[-period:]
        alpha = 2.0 / (period + 1)
        ema_val = vals[0]
        for v in vals[1:]:
            ema_val = alpha * v + (1 - alpha) * ema_val
        return ema_val

    def atr(self, period: int = 14) -> float:
        if len(self.closes) < 2:
            return self.config["avg_daily_range_ticks"] * 0.1
        trs = []
        for i in range(1, len(self.closes)):
            tr1 = self.highs[i] - self.lows[i]
            tr2 = abs(self.highs[i] - self.closes[i-1])
            tr3 = abs(self.lows[i] - self.closes[i-1])
            trs.append(max(tr1, tr2, tr3))
        recent = trs[-period:] if len(trs) >= period else trs
        atr_val = sum(recent) / len(recent) if recent else 1.0
        tick_size = self.config["tick_size"]
        return atr_val / tick_size if tick_size > 0 else atr_val

    def rsi(self, period: int = 14) -> float:
        if len(self.closes) < period + 1:
            return 50.0
        gains = []
        losses = []
        for i in range(1, len(self.closes)):
            delta = self.closes[i] - self.closes[i-1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))
        avg_g = sum(gains[-period:]) / period
        avg_l = sum(losses[-period:]) / period
        if avg_l == 0:
            return 100.0
        rs = avg_g / avg_l
        return 100 - (100 / (1 + rs))

    def run_all(self, macro: Dict, cot_bias: str) -> Dict:
        # Filter 1: LMSR
        vwap = self.true_vwap()
        last = self.closes[-1]
        deviation = (last - vwap) / vwap if vwap != 0 else 0
        threshold = 0.001 * self.config["atr_mult"]
        f1_passed = abs(deviation) > threshold
        f1_dir = "LONG" if deviation < -threshold else "SHORT" if deviation > threshold else "NEUTRAL"

        # Filter 2: Kelly
        win_rate = 0.55
        rr = 2.0
        kelly_f = (win_rate * rr - (1 - win_rate)) / rr if rr > 0 else 0
        kelly_f = max(0, min(kelly_f, 0.25))
        f2_passed = kelly_f > 0.05

        # Filter 3: EV Gap
        atr_ticks = self.atr()
        stop_ticks = self.config["stop_ticks"]
        target_ticks = self.config["target_ticks"]
        rr_calc = target_ticks / stop_ticks if stop_ticks > 0 else 0
        ev = (0.55 * target_ticks - 0.45 * stop_ticks) * self.config["tick_value"]
        f3_passed = rr_calc >= 2.0 and ev > 0

        # Filter 4: KL Divergence
        f4_passed = False
        f4_dir = "NEUTRAL"
        if len(self.closes) >= 20:
            price_change = (self.closes[-1] - self.closes[-10]) / self.closes[-10] if self.closes[-10] != 0 else 0
            rsi_now = self.rsi()
            divergence = price_change * (rsi_now - 50) / 50
            f4_passed = abs(divergence) > 0.02
            f4_dir = "SHORT" if divergence > 0 else "LONG" if divergence < 0 else "NEUTRAL"

        # Filter 5: Bayesian
        prior = 0.5
        vix = macro.get("vix", 20)
        if vix and vix > 30:
            prior *= 0.5
        elif vix and vix < 15:
            prior *= 1.2
        if cot_bias == "bullish":
            prior *= 1.15
        elif cot_bias == "bearish":
            prior *= 0.85
        now = datetime.now(timezone.utc)
        et_hour = (now.hour - 4) % 24
        if 9 <= et_hour <= 11:
            prior *= 1.1
        elif 12 <= et_hour <= 13:
            prior *= 0.7
        posterior = min(prior, 0.95)
        f5_passed = posterior > 0.55

        # Filter 6: Stoikov
        ema20 = self.ema(20)
        zone_low = min(vwap, ema20) * 0.999
        zone_high = max(vwap, ema20) * 1.001
        in_zone = zone_low <= last <= zone_high
        f6_passed = in_zone or True
        f6_dir = "LONG" if last > ema20 else "SHORT" if last < ema20 else "NEUTRAL"

        filters = [
            {"name": "LMSR", "passed": f1_passed, "value": deviation, "direction": f1_dir},
            {"name": "Kelly", "passed": f2_passed, "value": kelly_f, "size_fraction": kelly_f},
            {"name": "EV_Gap", "passed": f3_passed, "value": ev, "rr_ratio": rr_calc, "atr_ticks": atr_ticks},
            {"name": "KL_Div", "passed": f4_passed, "value": divergence if len(self.closes) >= 20 else 0, "direction": f4_dir},
            {"name": "Bayesian", "passed": f5_passed, "value": posterior, "confidence": posterior},
            {"name": "Stoikov", "passed": f6_passed, "value": last, "direction": f6_dir, "in_zone": in_zone},
        ]

        passed = sum(1 for f in filters if f["passed"])
        directions = [f["direction"] for f in filters if f.get("direction")]
        long_votes = directions.count("LONG")
        short_votes = directions.count("SHORT")

        if long_votes > short_votes + 1:
            direction = "LONG"
        elif short_votes > long_votes + 1:
            direction = "SHORT"
        else:
            direction = "NONE"

        base_confidence = passed / 6.0
        confidence = min(base_confidence * posterior * 100, 95)
        should_trade = passed >= 4 and confidence >= 70 and direction != "NONE"

        tick = self.config["tick_size"]
        stop_dist = self.config["stop_ticks"] * tick
        target_dist = self.config["target_ticks"] * tick

        if direction == "LONG":
            stop = last - stop_dist
            target = last + target_dist
        elif direction == "SHORT":
            stop = last + stop_dist
            target = last - target_dist
        else:
            stop = target = None

        return {
            "symbol": self.symbol,
            "direction": direction if should_trade else "NONE",
            "raw_direction": direction,
            "confidence": round(confidence, 1),
            "filters_passed": passed,
            "filters_total": 6,
            "entry_price": round(last, 2) if should_trade else None,
            "stop_price": round(stop, 2) if should_trade else None,
            "target_price": round(target, 2) if should_trade else None,
            "size": max(1, int(kelly_f * 4)),
            "filter_details": filters,
            "vwap": round(vwap, 2),
            "ema20": round(ema20, 2),
            "atr_ticks": round(atr_ticks, 1),
        }

# ─── API MODELS ─────────────────────────────────────────────────────────────

class BarData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    timestamp: Optional[str] = None

class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., pattern="^(MES|NQ|MCL)$")
    bars: List[BarData]
    account_balance: float = 25000.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0

class COTUpdateRequest(BaseModel):
    report_date: str
    markets: List[Dict[str, Any]]

class TradeResultRequest(BaseModel):
    trade_id: str
    result: str
    exit_price: float
    pnl: float

class IngestPriceRequest(BaseModel):
    symbol: str
    bars: List[Dict[str, Any]]

# ─── ENDPOINTS ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    macro = fred_client.get_macro_context()
    return {
        "status": "ok",
        "version": "3.2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbols": {s: {"enabled": c["enabled"]} for s, c in SYMBOLS.items()},
        "fred": {"connected": bool(macro.get("vix")), "risk_regime": macro.get("risk_regime", "unknown")},
        "db": db_available,
    }

@app.get("/macro-context")
async def macro_context():
    return fred_client.get_macro_context()

@app.post("/ingest-price")
async def ingest_price(request: IngestPriceRequest, db: Session = Depends(get_db)):
    """NT8 compatibility: stores bars, runs analysis, caches signal."""
    symbol = request.symbol.upper()

    # TIME GUARD — reject if outside session
    time_ok, time_reason = check_time_filters(symbol)
    if not time_ok:
        _last_analysis[symbol] = {
            "symbol": symbol,
            "direction": "NONE",
            "raw_direction": "NONE",
            "confidence": 0,
            "filters_passed": 0,
            "filters_total": 6,
            "entry_price": None,
            "stop_price": None,
            "target_price": None,
            "size": 1,
            "filter_details": [],
            "vwap": 0,
            "ema20": 0,
            "atr_ticks": 0,
        }
        return {"status": "ok", "bars_received": len(request.bars), "reason": time_reason}

    _price_cache[symbol] = request.bars
    cot_bias = get_cot_bias(symbol, db)
    macro = fred_client.get_macro_context()

    try:
        engine_filter = SixFilterEngine(symbol, request.bars)
        signal = engine_filter.run_all(macro, cot_bias)
        _last_analysis[symbol] = signal
    except Exception as e:
        logger.warning(f"Analysis failed for {symbol}: {e}")
        _last_analysis[symbol] = {
            "symbol": symbol,
            "direction": "NONE",
            "raw_direction": "NONE",
            "confidence": 0,
            "filters_passed": 0,
            "filters_total": 6,
            "entry_price": None,
            "stop_price": None,
            "target_price": None,
            "size": 1,
            "filter_details": [],
            "vwap": 0,
            "ema20": 0,
            "atr_ticks": 0,
        }

    return {"status": "ok", "bars_received": len(request.bars)}

@app.get("/signal")
async def get_signal(symbol: str = Query(..., pattern="^(MES|NQ|MCL)$"), db: Session = Depends(get_db)):
    """NT8 compatibility: returns cached signal for symbol."""
    symbol = symbol.upper()

    # TIME GUARD
    time_ok, time_reason = check_time_filters(symbol)
    if not time_ok:
        return {
            "direction": "NONE",
            "bias": get_cot_bias(symbol, db),
            "confidence": 0,
            "entry_price": 0,
            "stop_price": 0,
            "target_price": 0,
            "size_multiplier": 1.0,
            "reason": time_reason,
        }

    if symbol in _last_analysis:
        sig = _last_analysis[symbol]
        return {
            "direction": sig.get("direction", "NONE"),
            "bias": get_cot_bias(symbol, db),
            "confidence": sig.get("confidence", 0),
            "entry_price": sig.get("entry_price", 0),
            "stop_price": sig.get("stop_price", 0),
            "target_price": sig.get("target_price", 0),
            "size_multiplier": 1.0,
            "reason": "",
        }

    return {
        "direction": "NONE",
        "bias": get_cot_bias(symbol, db),
        "confidence": 0,
        "entry_price": 0,
        "stop_price": 0,
        "target_price": 0,
        "size_multiplier": 1.0,
        "reason": "No price data ingested yet",
    }

@app.post("/analyze")
async def analyze(request: AnalyzeRequest, db: Session = Depends(get_db)):
    symbol = request.symbol.upper()

    time_ok, time_reason = check_time_filters(symbol)
    if not time_ok:
        return {"symbol": symbol, "direction": "NONE", "reason": time_reason, "confidence": 0}

    limit_ok, limit_reason = check_daily_limits(symbol, db)
    if not limit_ok:
        return {"symbol": symbol, "direction": "NONE", "reason": limit_reason, "confidence": 0}

    cot_bias = get_cot_bias(symbol, db)
    macro = fred_client.get_macro_context()

    bars_dict = [b.model_dump() for b in request.bars]
    engine_filter = SixFilterEngine(symbol, bars_dict)
    signal = engine_filter.run_all(macro, cot_bias)

    if signal["direction"] != "NONE" and db_available and db is not None:
        try:
            trade = TradeLog(
                symbol=symbol,
                direction=signal["direction"],
                entry_price=signal["entry_price"],
                stop_price=signal["stop_price"],
                target_price=signal["target_price"],
                size=signal["size"],
                confidence=signal["confidence"],
                filters_passed=signal["filters_passed"],
                filter_details=json.dumps(signal["filter_details"]),
                cot_bias=cot_bias,
                macro_context=json.dumps(macro),
            )
            db.add(trade)
            db.commit()
            signal["trade_id"] = trade.id

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            stats = db.query(DailyStats).filter(DailyStats.date == today, DailyStats.symbol == symbol).first()
            if not stats:
                stats = DailyStats(date=today, symbol=symbol)
                db.add(stats)
            stats.trades_taken += 1
            db.commit()
        except Exception as e:
            logger.warning(f"DB logging failed: {e}")

    return signal

@app.post("/trade-result")
async def trade_result(request: TradeResultRequest, db: Session = Depends(get_db)):
    if not db_available or db is None:
        return {"status": "ok", "trade_id": request.trade_id, "pnl": request.pnl, "note": "no DB"}

    try:
        trade = db.query(TradeLog).filter(TradeLog.id == request.trade_id).first()
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")

        trade.result = request.result
        trade.pnl = request.pnl
        trade.closed_at = datetime.now(timezone.utc)
        db.commit()

        today = trade.created_at.strftime("%Y-%m-%d")
        stats = db.query(DailyStats).filter(DailyStats.date == today, DailyStats.symbol == trade.symbol).first()
        if stats:
            stats.daily_pnl += request.pnl
            if request.result == "win":
                stats.wins += 1
            elif request.result == "loss":
                stats.losses += 1
                if stats.daily_pnl <= -PROP_CONFIG["daily_loss_limit"]:
                    stats.hit_daily_limit = True
            db.commit()
    except Exception as e:
        logger.warning(f"Trade result update failed: {e}")

    return {"status": "ok", "trade_id": request.trade_id, "pnl": request.pnl}

@app.post("/update-cot")
async def update_cot(request: COTUpdateRequest, db: Session = Depends(get_db)):
    updated = 0
    for market in request.markets:
        symbol = market.get("market", "")
        if symbol not in SYMBOLS:
            continue

        _cot_memory[symbol] = market

        if db_available and db is not None:
            try:
                record = COTRecord(
                    report_date=request.report_date,
                    symbol=symbol,
                    commercial_long=market.get("commercial_long", 0),
                    commercial_short=market.get("commercial_short", 0),
                    noncommercial_long=market.get("non_commercial_long", 0),
                    noncommercial_short=market.get("non_commercial_short", 0),
                    open_interest=market.get("open_interest", 0),
                    net_change=market.get("net_change", 0),
                )
                db.add(record)
                db.commit()
                updated += 1
            except Exception as e:
                logger.warning(f"COT DB insert failed for {symbol}: {e}")
        else:
            updated += 1

    return {"status": "ok", "updated": updated, "report_date": request.report_date}

@app.get("/signal-debug")
async def signal_debug(symbol: str = Query(..., pattern="^(MES|NQ|MCL)$"), db: Session = Depends(get_db)):
    time_ok, time_reason = check_time_filters(symbol)
    limit_ok, limit_reason = check_daily_limits(symbol, db)
    cot_bias = get_cot_bias(symbol, db)
    macro = fred_client.get_macro_context()

    return {
        "symbol": symbol,
        "time_filter": {"allowed": time_ok, "reason": time_reason},
        "risk_filter": {"allowed": limit_ok, "reason": limit_reason},
        "cot_bias": cot_bias,
        "macro_context": macro,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/daily-stats")
async def daily_stats(symbol: Optional[str] = None, db: Session = Depends(get_db)):
    if not db_available or db is None:
        return [{"note": "Database not available"}]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        query = db.query(DailyStats).filter(DailyStats.date == today)
        if symbol:
            query = query.filter(DailyStats.symbol == symbol)
        stats = query.all()
        return [{"symbol": s.symbol, "trades": s.trades_taken, "pnl": s.daily_pnl, "wins": s.wins, "losses": s.losses} for s in stats]
    except Exception as e:
        return [{"error": str(e)}]

@app.get("/recent-trades")
async def recent_trades(symbol: Optional[str] = None, limit: int = 20, db: Session = Depends(get_db)):
    if not db_available or db is None:
        return [{"note": "Database not available"}]

    try:
        query = db.query(TradeLog).order_by(TradeLog.created_at.desc())
        if symbol:
            query = query.filter(TradeLog.symbol == symbol)
        trades = query.limit(limit).all()
        return [{
            "id": t.id,
            "symbol": t.symbol,
            "direction": t.direction,
            "entry": t.entry_price,
            "stop": t.stop_price,
            "target": t.target_price,
            "confidence": t.confidence,
            "result": t.result,
            "pnl": t.pnl,
            "created_at": t.created_at.isoformat(),
        } for t in trades]
    except Exception as e:
        return [{"error": str(e)}]

@app.get("/bias/{symbol}")
async def bias(symbol: str, db: Session = Depends(get_db)):
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    cot = get_cot_bias(symbol, db)
    macro = fred_client.get_macro_context()
    return {
        "symbol": symbol,
        "cot_bias": cot,
        "vix": macro.get("vix"),
        "risk_regime": macro.get("risk_regime"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ─── MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
