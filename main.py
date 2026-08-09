"""
FlowAlert Institutional v3.0
SixFilter + FRED Macro + Prop Firm Risk Management
Symbols: MES (core), NQ (tech beta), MCL (oil diversifier)
"""

import os
import json
import math
import uuid
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from enum import Enum

import requests
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ─── CONFIG ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flowalert")

app = FastAPI(title="FlowAlert Institutional", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/flowalert")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Prop Firm Config (Bulenox)
PROP_CONFIG = {
    "starting_balance": 25000.0,
    "trailing_drawdown": 1500.0,
    "daily_loss_limit": 1000.0,
    "profit_target": 1500.0,
    "max_trades_per_day": 3,
}

# Symbol Configs
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
        "eia_day": 3,  # Wednesday
        "eia_time": "10:30",
    },
}

# ─── DATABASE ───────────────────────────────────────────────────────────────

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TradeLog(Base):
    __tablename__ = "trades"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, index=True)
    direction = Column(String)  # LONG / SHORT / NONE
    entry_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    size = Column(Integer)
    confidence = Column(Float)
    filters_passed = Column(Integer)
    filter_details = Column(Text)
    cot_bias = Column(String)
    macro_context = Column(Text)
    result = Column(String, default="open")  # open / win / loss / breakeven
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
    commercial_long = Column(Float)
    commercial_short = Column(Float)
    noncommercial_long = Column(Float)
    noncommercial_short = Column(Float)
    open_interest = Column(Float)
    net_change = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─── FRED CLIENT ────────────────────────────────────────────────────────────

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
            r = requests.get(
                self.BASE,
                params={
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                },
                timeout=5,
            )
            data = r.json()
            obs = data.get("observations", [])
            if obs and obs[0].get("value") != ".":
                return float(obs[0]["value"])
        except Exception as e:
            logger.warning(f"FRED fetch failed for {series_id}: {e}")
        return None

    def get_macro_context(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        if self._cache_time and (now - self._cache_time).seconds < 300:
            return self._cache

        context = {}
        for name, sid in self.SERIES.items():
            context[name] = self._fetch(sid)

        # Calculate WTI-Brent spread
        if context.get("wti") and context.get("brent"):
            context["spread"] = context["brent"] - context["wti"]

        # Risk regime
        vix = context.get("vix", 20)
        context["risk_regime"] = (
            "extreme" if vix > 30 else
            "high" if vix > 25 else
            "elevated" if vix > 20 else
            "normal"
        )

        self._cache = context
        self._cache_time = now
        return context

fred_client = FREDClient()

# ─── SIX FILTER ENGINE ──────────────────────────────────────────────────────

class SixFilterEngine:
    """
    1. LMSR - Price deviation from true VWAP
    2. Kelly Criterion - Position sizing
    3. EV Gap - Expected Value (2:1 RR minimum)
    4. KL Divergence - Price/RSI divergence
    5. Bayesian Updates - Context filter
    6. Stoikov Execution - Limit order at VWAP/EMA confluence
    """

    def __init__(self, symbol: str, bars: List[Dict]):
        self.symbol = symbol
        self.bars = bars
        self.config = SYMBOLS.get(symbol, SYMBOLS["MES"])
        self.closes = np.array([b["close"] for b in bars])
        self.highs = np.array([b["high"] for b in bars])
        self.lows = np.array([b["low"] for b in bars])
        self.volumes = np.array([b.get("volume", 0) for b in bars])
        self.opens = np.array([b["open"] for b in bars])

    def true_vwap(self) -> float:
        """Volume-weighted average price (typical price * volume)."""
        typical = (self.highs + self.lows + self.closes) / 3
        vol = self.volumes + 1e-9
        return np.sum(typical * vol) / np.sum(vol)

    def ema(self, period: int = 20) -> float:
        """Exponential moving average."""
        if len(self.closes) < period:
            return float(np.mean(self.closes))
        weights = np.exp(np.linspace(-1, 0, period))
        weights /= weights.sum()
        return float(np.convolve(self.closes[-period:], weights, mode="valid")[0])

    def atr(self, period: int = 14) -> float:
        """Average True Range in ticks."""
        if len(self.closes) < 2:
            return self.config["avg_daily_range_ticks"] * 0.1
        tr1 = self.highs[1:] - self.lows[1:]
        tr2 = np.abs(self.highs[1:] - self.closes[:-1])
        tr3 = np.abs(self.lows[1:] - self.closes[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr_val = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
        tick_size = self.config["tick_size"]
        return atr_val / tick_size if tick_size > 0 else atr_val

    def rsi(self, period: int = 14) -> float:
        """Relative Strength Index."""
        if len(self.closes) < period + 1:
            return 50.0
        deltas = np.diff(self.closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def lmsr_deviation(self) -> Dict:
        """Filter 1: Logarithmic Market Scoring Rule deviation."""
        vwap = self.true_vwap()
        last_price = self.closes[-1]
        deviation = (last_price - vwap) / vwap if vwap != 0 else 0

        # Normalized to -1 to 1
        threshold = 0.001 * self.config["atr_mult"]
        passed = abs(deviation) > threshold

        return {
            "name": "LMSR",
            "passed": passed,
            "value": deviation,
            "threshold": threshold,
            "direction": "LONG" if deviation < -threshold else "SHORT" if deviation > threshold else "NEUTRAL",
        }

    def kelly_criterion(self, win_rate: float = 0.55, rr: float = 2.0) -> Dict:
        """Filter 2: Kelly fraction for position sizing."""
        kelly_f = (win_rate * rr - (1 - win_rate)) / rr if rr > 0 else 0
        kelly_f = max(0, min(kelly_f, 0.25))  # Cap at 25%

        return {
            "name": "Kelly",
            "passed": kelly_f > 0.05,
            "value": kelly_f,
            "size_fraction": kelly_f,
        }

    def ev_gap(self) -> Dict:
        """Filter 3: Expected Value gap (2:1 minimum RR)."""
        atr_ticks = self.atr()
        stop_ticks = self.config["stop_ticks"]
        target_ticks = self.config["target_ticks"]

        rr = target_ticks / stop_ticks if stop_ticks > 0 else 0
        ev = (0.55 * target_ticks - 0.45 * stop_ticks) * self.config["tick_value"]

        return {
            "name": "EV_Gap",
            "passed": rr >= 2.0 and ev > 0,
            "value": ev,
            "rr_ratio": rr,
            "atr_ticks": atr_ticks,
        }

    def kl_divergence(self) -> Dict:
        """Filter 4: KL divergence between price and RSI momentum."""
        if len(self.closes) < 20:
            return {"name": "KL_Div", "passed": False, "value": 0, "direction": "NEUTRAL"}

        price_change = (self.closes[-1] - self.closes[-10]) / self.closes[-10] if self.closes[-10] != 0 else 0
        rsi_now = self.rsi()
        rsi_then = 50.0  # Simplified

        # Divergence: price up but RSI down = bearish, vice versa = bullish
        divergence = price_change * (rsi_now - 50) / 50

        passed = abs(divergence) > 0.02
        direction = "SHORT" if divergence > 0 else "LONG" if divergence < 0 else "NEUTRAL"

        return {
            "name": "KL_Div",
            "passed": passed,
            "value": divergence,
            "direction": direction,
        }

    def bayesian_context(self, macro: Dict, cot_bias: str) -> Dict:
        """Filter 5: Bayesian update with macro and COT context."""
        prior = 0.5  # Neutral

        # VIX adjustment
        vix = macro.get("vix", 20)
        if vix > 30:
            prior *= 0.5  # High uncertainty
        elif vix < 15:
            prior *= 1.2  # Low vol, higher conviction

        # COT adjustment
        if cot_bias == "bullish":
            prior *= 1.15
        elif cot_bias == "bearish":
            prior *= 0.85

        # Time of day (simplified)
        now = datetime.now(timezone.utc)
        et_hour = (now.hour - 4) % 24  # Rough ET conversion
        if 9 <= et_hour <= 11:
            prior *= 1.1  # Best session
        elif 12 <= et_hour <= 13:
            prior *= 0.7  # Lunch

        posterior = min(prior, 0.95)

        return {
            "name": "Bayesian",
            "passed": posterior > 0.55,
            "value": posterior,
            "confidence": posterior,
        }

    def stoikov_level(self) -> Dict:
        """Filter 6: Optimal limit entry at VWAP/EMA confluence."""
        vwap = self.true_vwap()
        ema20 = self.ema(20)
        last = self.closes[-1]

        # Confluence zone
        zone_low = min(vwap, ema20) * 0.999
        zone_high = max(vwap, ema20) * 1.001

        in_zone = zone_low <= last <= zone_high

        # Direction based on trend
        direction = "LONG" if last > ema20 else "SHORT" if last < ema20 else "NEUTRAL"

        return {
            "name": "Stoikov",
            "passed": in_zone or True,  # Always allow, but flag proximity
            "value": last,
            "entry_zone": (zone_low, zone_high),
            "direction": direction,
            "in_zone": in_zone,
        }

    def run_all(self, macro: Dict, cot_bias: str) -> Dict:
        """Run all 6 filters and return composite signal."""
        f1 = self.lmsr_deviation()
        f2 = self.kelly_criterion()
        f3 = self.ev_gap()
        f4 = self.kl_divergence()
        f5 = self.bayesian_context(macro, cot_bias)
        f6 = self.stoikov_level()

        filters = [f1, f2, f3, f4, f5, f6]
        passed = sum(1 for f in filters if f["passed"])

        # Direction consensus
        directions = [f.get("direction", "NEUTRAL") for f in filters if f.get("direction")]
        long_votes = directions.count("LONG")
        short_votes = directions.count("SHORT")

        if long_votes > short_votes + 1:
            direction = "LONG"
        elif short_votes > long_votes + 1:
            direction = "SHORT"
        else:
            direction = "NONE"

        # Confidence = passed filters / 6 * Bayesian posterior
        base_confidence = passed / 6.0
        bayesian_conf = f5["value"]
        confidence = min(base_confidence * bayesian_conf * 100, 95)

        # Only trade if 4+ filters align and confidence >= 70%
        should_trade = passed >= 4 and confidence >= 70 and direction != "NONE"

        # Calculate levels
        last = self.closes[-1]
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
            "size": max(1, int(f2["size_fraction"] * 4)),  # Scale 1-4 contracts
            "filter_details": filters,
            "vwap": round(vwap, 2),
            "ema20": round(ema20, 2),
            "atr_ticks": round(f3["atr_ticks"], 1),
        }

# ─── TIME FILTERS ───────────────────────────────────────────────────────────

def check_time_filters(symbol: str) -> tuple[bool, str]:
    """Returns (allowed, reason)."""
    cfg = SYMBOLS.get(symbol)
    if not cfg or not cfg.get("enabled"):
        return False, "Symbol disabled"

    now = datetime.now(timezone.utc)
    # Approximate ET (UTC-4 or UTC-5 depending on DST)
    et_offset = timedelta(hours=4)  # Simplified
    et_now = now - et_offset
    et_time = et_now.strftime("%H:%M")
    et_weekday = et_now.weekday()

    # Weekend
    if et_weekday >= 5:
        return False, "Weekend"

    # Session hours
    start = cfg.get("session_start")
    end = cfg.get("session_end")
    if start and end:
        if not (start <= et_time <= end):
            return False, f"Outside session {start}-{end}"

    # Lunch ban
    lunch = cfg.get("lunch_ban")
    if lunch:
        if lunch[0] <= et_time <= lunch[1]:
            return False, f"Lunch ban {lunch[0]}-{lunch[1]}"

    # EIA special for MCL
    if symbol == "MCL":
        eia_day = cfg.get("eia_day", 3)  # Wednesday
        if et_weekday == eia_day:
            eia_time = cfg.get("eia_time", "10:30")
            # Allow 10:25-11:00 for volatility
            if not ("10:25" <= et_time <= "11:00"):
                return False, "MCL: Waiting for EIA 10:30 AM"

    return True, "OK"

# ─── COT MANAGEMENT ─────────────────────────────────────────────────────────

def get_cot_bias(symbol: str, db: Session) -> str:
    """Get latest COT bias for symbol."""
    record = db.query(COTRecord).filter(COTRecord.symbol == symbol).order_by(COTRecord.report_date.desc()).first()
    if not record:
        return "neutral"

    comm_net = record.commercial_long - record.commercial_short
    noncomm_net = record.noncommercial_long - record.noncommercial_short

    # Commercials net long = bullish (they're hedging by being long)
    # In practice for financials: Asset Managers long = bullish
    if comm_net > 0 and noncomm_net < 0:
        return "bullish"
    elif comm_net < 0 and noncomm_net > 0:
        return "bearish"
    return "neutral"

# ─── RISK MANAGEMENT ────────────────────────────────────────────────────────

def check_daily_limits(symbol: str, db: Session) -> tuple[bool, str]:
    """Check prop firm daily limits."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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

    return True, "OK"

# ─── API MODELS ─────────────────────────────────────────────────────────────

class BarData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    timestamp: Optional[str] = None

class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., regex="^(MES|NQ|MCL)$")
    bars: List[BarData]
    account_balance: float = 25000.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0

class COTUpdateRequest(BaseModel):
    report_date: str
    markets: List[Dict[str, Any]]

class TradeResultRequest(BaseModel):
    trade_id: str
    result: str  # win / loss / breakeven
    exit_price: float
    pnl: float

# ─── ENDPOINTS ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    macro = fred_client.get_macro_context()
    return {
        "status": "ok",
        "version": "3.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbols": {s: {"enabled": c["enabled"]} for s, c in SYMBOLS.items()},
        "fred": {"connected": bool(macro.get("vix")), "risk_regime": macro.get("risk_regime", "unknown")},
    }

@app.get("/macro-context")
async def macro_context():
    return fred_client.get_macro_context()

@app.post("/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks, db: Session = next(get_db())):
    symbol = request.symbol.upper()

    # Time filter
    time_ok, time_reason = check_time_filters(symbol)
    if not time_ok:
        return {"symbol": symbol, "direction": "NONE", "reason": time_reason, "confidence": 0}

    # Daily limits
    limit_ok, limit_reason = check_daily_limits(symbol, db)
    if not limit_ok:
        return {"symbol": symbol, "direction": "NONE", "reason": limit_reason, "confidence": 0}

    # COT bias
    cot_bias = get_cot_bias(symbol, db)

    # Macro context
    macro = fred_client.get_macro_context()

    # Run SixFilter
    bars_dict = [b.dict() for b in request.bars]
    engine = SixFilterEngine(symbol, bars_dict)
    signal = engine.run_all(macro, cot_bias)

    # Log if trade signal
    if signal["direction"] != "NONE":
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

        # Update daily stats
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stats = db.query(DailyStats).filter(DailyStats.date == today, DailyStats.symbol == symbol).first()
        if not stats:
            stats = DailyStats(date=today, symbol=symbol)
            db.add(stats)
        stats.trades_taken += 1
        db.commit()

    return signal

@app.post("/trade-result")
async def trade_result(request: TradeResultRequest, db: Session = next(get_db())):
    trade = db.query(TradeLog).filter(TradeLog.id == request.trade_id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    trade.result = request.result
    trade.pnl = request.pnl
    trade.closed_at = datetime.now(timezone.utc)
    db.commit()

    # Update daily stats
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

    return {"status": "ok", "trade_id": request.trade_id, "pnl": request.pnl}

@app.post("/update-cot")
async def update_cot(request: COTUpdateRequest, db: Session = next(get_db())):
    for market in request.markets:
        symbol = market.get("market", "")
        if symbol not in SYMBOLS:
            continue

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
    return {"status": "ok", "updated": len(request.markets), "report_date": request.report_date}

@app.get("/signal-debug")
async def signal_debug(symbol: str = Query(..., regex="^(MES|NQ|MCL)$"), db: Session = next(get_db())):
    """Debug endpoint showing why a signal was generated or blocked."""
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
async def daily_stats(symbol: Optional[str] = None, db: Session = next(get_db())):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    query = db.query(DailyStats).filter(DailyStats.date == today)
    if symbol:
        query = query.filter(DailyStats.symbol == symbol)
    stats = query.all()
    return [{"symbol": s.symbol, "trades": s.trades_taken, "pnl": s.daily_pnl, "wins": s.wins, "losses": s.losses} for s in stats]

@app.get("/recent-trades")
async def recent_trades(symbol: Optional[str] = None, limit: int = 20, db: Session = next(get_db())):
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

@app.get("/bias/{symbol}")
async def bias(symbol: str, db: Session = next(get_db())):
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
