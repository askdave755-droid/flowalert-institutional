"""
FlowAlert Institutional Monitor - Single File
Detects bank positioning via COT + Price Action for MES, NQ, GC
Deploy: railway init -> railway up
NT8 Integration: Poll GET /signal?symbol=MES every 5 min
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="FlowAlert Institutional", version="2.0")

DB_PATH = os.getenv("DB_PATH", "./flowalert.db")

# ============ DATABASE ============
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # COT weekly data
    c.execute('''
        CREATE TABLE IF NOT EXISTS cot_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT,
            symbol TEXT,
            commercial_long INTEGER,
            commercial_short INTEGER,
            noncomm_long INTEGER,
            noncomm_short INTEGER,
            nonrep_long INTEGER,
            nonrep_short INTEGER,
            open_interest INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(report_date, symbol)
        )
    ''')
    
    # Trade signals generated
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            bias TEXT,
            confidence INTEGER,
            direction TEXT,
            entry_price REAL,
            stop_price REAL,
            target_price REAL,
            size_multiplier REAL,
            reason TEXT,
            executed BOOLEAN DEFAULT 0,
            pnl REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Price snapshots from NT8
    c.execute('''
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ============ COT MAPPING ============
# CFTC report names -> our symbols
COT_SYMBOL_MAP = {
    "MES": "E-MINI S&P 500",
    "NQ": "NASDAQ-100 STOCK INDEX (MINI)",
    "GC": "GOLD",
    "ES": "E-MINI S&P 500",
    "MNQ": "NASDAQ-100 STOCK INDEX (MINI)",
    "YM": "DJIA x $5",
    "CL": "CRUDE OIL",
}

# ============ PYDANTIC MODELS ============
class COTUpdate(BaseModel):
    report_date: str  # YYYY-MM-DD
    symbol: str       # MES, NQ, GC
    commercial_long: int
    commercial_short: int
    noncomm_long: int
    noncomm_short: int
    nonrep_long: int = 0
    nonrep_short: int = 0
    open_interest: int

class OHLCVBar(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class PriceData(BaseModel):
    symbol: str
    bars: List[OHLCVBar]

class SignalResponse(BaseModel):
    timestamp: str
    symbol: str
    bias: str
    confidence: int
    direction: str  # LONG, SHORT, NONE
    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    size_multiplier: float
    reason: str

# ============ COT ANALYSIS ENGINE ============
class COTAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.cot_name = COT_SYMBOL_MAP.get(symbol, symbol)
    
    def get_recent_cot(self, weeks: int = 4):
        """Get last N weeks of COT data"""
        conn = get_db()
        rows = conn.execute(
            """SELECT * FROM cot_data 
               WHERE symbol = ? 
               ORDER BY report_date DESC LIMIT ?""",
            (self.symbol, weeks)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def calculate_bias(self) -> Dict[str, Any]:
        """
        Smart Money Logic:
        - Commercial Net Position (hedgers): Positive = accumulating = BULLISH
        - Non-Commercial Net Position (banks/specs): Extreme positive = distribution = BEARISH
        - We track the 3-week trend of commercial positioning
        """
        data = self.get_recent_cot(weeks=4)
        if len(data) < 2:
            return {"bias": "NEUTRAL", "confidence": 0, "reason": "Insufficient COT data"}
        
        # Latest and previous week
        latest = data[0]
        prev = data[1]
        three_wk = data[-1] if len(data) >= 3 else prev
        
        # Commercial Net Position (Smart Money)
        comm_net_now = latest['commercial_long'] - latest['commercial_short']
        comm_net_prev = prev['commercial_long'] - prev['commercial_short']
        comm_net_3wk = three_wk['commercial_long'] - three_wk['commercial_short']
        
        # Non-Commercial Net Position (Banks/Large Specs)
        noncomm_net_now = latest['noncomm_long'] - latest['noncomm_short']
        noncomm_net_prev = prev['noncomm_long'] - prev['noncomm_short']
        
        # Open Interest context
        oi_now = latest['open_interest']
        
        # Calculate scores
        comm_trend = comm_net_now - comm_net_3wk  # Positive = commercials getting longer
        noncomm_extreme = abs(noncomm_net_now) / oi if oi > 0 else 0
        
        # Bias determination
        bias = "NEUTRAL"
        confidence = 50
        reasons = []
        
        # Rule 1: Commercial accumulation (most important)
        if comm_trend > 0:
            bias = "BULLISH"
            confidence += 20
            reasons.append(f"Commercials accumulating (+{comm_trend:,} net)")
        elif comm_trend < 0:
            bias = "BEARISH"
            confidence += 20
            reasons.append(f"Commercials distributing ({comm_trend:,} net)")
        
        # Rule 2: Non-commercial extreme (contrarian)
        if noncomm_extreme > 0.25 and noncomm_net_now > 0:
            # Specs extremely long = they're the ones who get squeezed
            if bias == "BULLISH":
                confidence -= 15  # Reduce confidence
                reasons.append("Specs overcrowded long (caution)")
            else:
                bias = "BEARISH"
                confidence = 75
                reasons.append("Specs extreme long + commercials selling = distribution")
        elif noncomm_extreme > 0.25 and noncomm_net_now < 0:
            if bias == "BEARISH":
                confidence -= 15
                reasons.append("Specs overcrowded short (caution)")
            else:
                bias = "BULLISH"
                confidence = 75
                reasons.append("Specs extreme short + commercials buying = accumulation")
        
        # Rule 3: Commercial net position absolute
        if comm_net_now > 0 and comm_trend >= 0:
            confidence += 10
            reasons.append("Commercials net long")
        elif comm_net_now < 0 and comm_trend <= 0:
            confidence += 10
            reasons.append("Commercials net short")
        
        confidence = max(0, min(100, confidence))
        
        return {
            "bias": bias,
            "confidence": confidence,
            "commercial_net": comm_net_now,
            "commercial_trend": comm_trend,
            "noncomm_net": noncomm_net_now,
            "noncomm_extreme_pct": round(noncomm_extreme * 100, 1),
            "open_interest": oi_now,
            "report_date": latest['report_date'],
            "reason": " | ".join(reasons)
        }

# ============ PRICE ACTION ENGINE ============
class PriceAnalyzer:
    def __init__(self, bars: List[Dict]):
        self.bars = bars
        self.df = None
        if len(bars) >= 20:
            self._build_df()
    
    def _build_df(self):
        import pandas as pd
        self.df = pd.DataFrame(self.bars)
        self.df['ema20'] = self.df['close'].ewm(span=20, adjust=False).mean()
        self.df['ema50'] = self.df['close'].ewm(span=50, adjust=False).mean()
        self.df['atr'] = self._calc_atr(14)
    
    def _calc_atr(self, period: int) -> pd.Series:
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def get_trend(self) -> str:
        if self.df is None or len(self.df) < 20:
            return "NEUTRAL"
        last = self.df.iloc[-1]
        if last['close'] > last['ema20'] > last['ema50']:
            return "UPTREND"
        elif last['close'] < last['ema20'] < last['ema50']:
            return "DOWNTREND"
        return "CHOPPY"
    
    def get_structure(self) -> Dict:
        """Find swing high/low for entry/stop placement"""
        if self.df is None or len(self.df) < 10:
            return {"swing_high": None, "swing_low": None, "atr": 0}
        
        recent = self.df.tail(20)
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        atr = self.df['atr'].iloc[-1] if not pd.isna(self.df['atr'].iloc[-1]) else (swing_high - swing_low) * 0.1
        
        return {
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "atr": round(atr, 2),
            "last_close": round(self.df['close'].iloc[-1], 2)
        }

# ============ SIGNAL GENERATOR ============
class SignalGenerator:
    def __init__(self, symbol: str, cot_analyzer: COTAnalyzer, price_analyzer: PriceAnalyzer):
        self.symbol = symbol
        self.cot = cot_analyzer
        self.price = price_analyzer
    
    def generate(self) -> SignalResponse:
        now = datetime.utcnow().isoformat()
        
        # Get institutional bias
        cot_result = self.cot.calculate_bias()
        bias = cot_result['bias']
        confidence = cot_result['confidence']
        
        # Get price trend
        price_trend = self.price.get_trend()
        structure = self.price.get_structure()
        
        # Decision logic: COT bias MUST align with price trend
        direction = "NONE"
        entry = None
        stop = None
        target = None
        size = 0.0
        reason = cot_result['reason']
        
        if bias == "BULLISH" and price_trend in ["UPTREND", "CHOPPY"]:
            if structure['swing_low'] and structure['atr']:
                direction = "LONG"
                entry = structure['last_close']
                stop = structure['swing_low'] - (structure['atr'] * 0.5)
                stop = round(stop, 2)
                risk = entry - stop
                target = round(entry + (risk * 2), 2)  # 2:1 RR
                size = self._calculate_size(confidence)
                reason += f" | Price: {price_trend}, Entry {entry}, Stop {stop}, Target {target}"
        
        elif bias == "BEARISH" and price_trend in ["DOWNTREND", "CHOPPY"]:
            if structure['swing_high'] and structure['atr']:
                direction = "SHORT"
                entry = structure['last_close']
                stop = structure['swing_high'] + (structure['atr'] * 0.5)
                stop = round(stop, 2)
                risk = stop - entry
                target = round(entry - (risk * 2), 2)
                size = self._calculate_size(confidence)
                reason += f" | Price: {price_trend}, Entry {entry}, Stop {stop}, Target {target}"
        
        else:
            reason += f" | Price trend {price_trend} conflicts with COT bias {bias}. NO TRADE."
        
        # Save to DB
        conn = get_db()
        conn.execute(
            """INSERT INTO signals 
               (timestamp, symbol, bias, confidence, direction, entry_price, stop_price, target_price, size_multiplier, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, self.symbol, bias, confidence, direction, entry, stop, target, size, reason)
        )
        conn.commit()
        conn.close()
        
        return SignalResponse(
            timestamp=now,
            symbol=self.symbol,
            bias=bias,
            confidence=confidence,
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            size_multiplier=size,
            reason=reason
        )
    
    def _calculate_size(self, confidence: int) -> float:
        """Kelly-inspired sizing: higher confidence = larger size"""
        if confidence >= 80:
            return 2.0
        elif confidence >= 65:
            return 1.5
        elif confidence >= 50:
            return 1.0
        else:
            return 0.5

# ============ API ENDPOINTS ============

@app.get("/health")
def health():
    return {"status": "ok", "system": "FlowAlert Institutional", "version": "2.0"}

@app.post("/update-cot")
def update_cot(data: COTUpdate):
    """Manually update COT data (paste from CFTC report weekly)"""
    conn = get_db()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO cot_data 
               (report_date, symbol, commercial_long, commercial_short, 
                noncomm_long, noncomm_short, nonrep_long, nonrep_short, open_interest)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (data.report_date, data.symbol, data.commercial_long, data.commercial_short,
             data.noncomm_long, data.noncomm_short, data.nonrep_long, data.nonrep_short,
             data.open_interest)
        )
        conn.commit()
        
        # Auto-calculate bias after update
        analyzer = COTAnalyzer(data.symbol)
        bias = analyzer.calculate_bias()
        
        return {
            "status": "saved",
            "symbol": data.symbol,
            "report_date": data.report_date,
            "bias": bias['bias'],
            "confidence": bias['confidence']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/ingest-price")
def ingest_price(data: PriceData):
    """NinjaTrader sends recent bars here"""
    conn = get_db()
    now = datetime.utcnow().isoformat()
    for bar in data.bars:
        conn.execute(
            """INSERT INTO price_snapshots (symbol, timestamp, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (data.symbol, bar.time, bar.open, bar.high, bar.low, bar.close, bar.volume)
        )
    conn.commit()
    conn.close()
    return {"status": "saved", "bars": len(data.bars), "symbol": data.symbol}

@app.get("/signal", response_model=SignalResponse)
def get_signal(symbol: str = "MES"):
    """
    Main endpoint for NinjaTrader.
    Returns current signal based on latest COT + recent price data.
    """
    # Get recent price data from DB
    conn = get_db()
    rows = conn.execute(
        """SELECT timestamp as time, open, high, low, close, volume 
           FROM price_snapshots 
           WHERE symbol = ? 
           ORDER BY timestamp DESC LIMIT 50""",
        (symbol,)
    ).fetchall()
    conn.close()
    
    bars = [dict(r) for r in reversed(rows)]
    
    if len(bars) < 20:
        return SignalResponse(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            bias="NEUTRAL",
            confidence=0,
            direction="NONE",
            entry_price=None,
            stop_price=None,
            target_price=None,
            size_multiplier=0,
            reason="Need at least 20 price bars. Send data to /ingest-price first."
        )
    
    cot = COTAnalyzer(symbol)
    price = PriceAnalyzer(bars)
    gen = SignalGenerator(symbol, cot, price)
    
    return gen.generate()

@app.get("/bias/{symbol}")
def get_bias(symbol: str):
    """Quick bias check without generating trade signal"""
    analyzer = COTAnalyzer(symbol)
    return analyzer.calculate_bias()

@app.get("/signals")
def get_signals(symbol: Optional[str] = None, limit: int = 20):
    conn = get_db()
    query = "SELECT * FROM signals WHERE 1=1"
    params = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Simple HTML dashboard"""
    conn = get_db()
    signals = conn.execute("SELECT * FROM signals ORDER BY created_at DESC LIMIT 10").fetchall()
    cot_rows = conn.execute("SELECT * FROM cot_data ORDER BY report_date DESC LIMIT 5").fetchall()
    conn.close()
    
    signals_html = "".join([
        f"<tr><td>{s['timestamp'][:19]}</td><td>{s['symbol']}</td><td>{s['bias']}</td>"
        f"<td>{s['confidence']}%</td><td><b>{s['direction']}</b></td>"
        f"<td>{s['entry_price']}</td><td>{s['stop_price']}</td><td>{s['target_price']}</td></tr>"
        for s in signals
    ])
    
    cot_html = "".join([
        f"<tr><td>{c['report_date']}</td><td>{c['symbol']}</td>"
        f"<td>{c['commercial_long']:,}</td><td>{c['commercial_short']:,}</td>"
        f"<td>{c['noncomm_long']:,}</td><td>{c['noncomm_short']:,}</td></tr>"
        for c in cot_rows
    ])
    
    return f"""
    <html>
    <head><title>FlowAlert Institutional</title>
    <style>
        body {{ font-family: monospace; background: #0a0a0a; color: #00ff88; padding: 20px; }}
        h1 {{ color: #ffd700; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #333; padding: 8px; text-align: left; }}
        th {{ background: #1a1a1a; color: #ffd700; }}
        .bullish {{ color: #00ff88; }}
        .bearish {{ color: #ff4444; }}
    </style></head>
    <body>
        <h1>🏦 FlowAlert Institutional Monitor</h1>
        <h2>Recent Signals</h2>
        <table>
            <tr><th>Time</th><th>Symbol</th><th>Bias</th><th>Conf</th><th>Dir</th><th>Entry</th><th>Stop</th><th>Target</th></tr>
            {signals_html}
        </table>
        <h2>COT Data</h2>
        <table>
            <tr><th>Date</th><th>Symbol</th><th>Comm Long</th><th>Comm Short</th><th>NonComm Long</th><th>NonComm Short</th></tr>
            {cot_html}
        </table>
        <p>Endpoints: <a href="/health" style="color:#ffd700">/health</a> | <a href="/signal?symbol=MES" style="color:#ffd700">/signal?symbol=MES</a> | <a href="/bias/MES" style="color:#ffd700">/bias/MES</a></p>
    </body>
    </html>
    """

@app.post("/report-fill")
def report_fill(signal_id: int, pnl: float):
    """Report back from NT8 when trade closes"""
    conn = get_db()
    conn.execute("UPDATE signals SET executed = 1, pnl = ? WHERE id = ?", (pnl, signal_id))
    conn.commit()
    conn.close()
    return {"status": "updated"}

# ============ MANUAL COT JSON TEMPLATES ============
@app.get("/cot-template/{symbol}")
def cot_template(symbol: str):
    """Returns empty template for manual COT entry"""
    return {
        "report_date": "2026-08-05",
        "symbol": symbol,
        "commercial_long": 0,
        "commercial_short": 0,
        "noncomm_long": 0,
        "noncomm_short": 0,
        "nonrep_long": 0,
        "nonrep_short": 0,
        "open_interest": 0
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
