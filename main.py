"""
FlowAlert Institutional Monitor - PostgreSQL Version (Debug)
"""

import os
import traceback
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="FlowAlert Institutional", version="2.0-pg-debug")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:QhhlQeZPzKzjFibjJhIjsrWPZKoAgtex@postgres.railway.internal:5432/railway")

def get_db():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS cot_data (
            id SERIAL PRIMARY KEY, report_date TEXT, symbol TEXT,
            commercial_long INTEGER, commercial_short INTEGER,
            noncomm_long INTEGER, noncomm_short INTEGER,
            nonrep_long INTEGER DEFAULT 0, nonrep_short INTEGER DEFAULT 0,
            open_interest INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(report_date, symbol)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id SERIAL PRIMARY KEY, timestamp TEXT, symbol TEXT, bias TEXT,
            confidence INTEGER, direction TEXT, entry_price REAL, stop_price REAL,
            target_price REAL, size_multiplier REAL, reason TEXT,
            executed BOOLEAN DEFAULT FALSE, pnl REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id SERIAL PRIMARY KEY, symbol TEXT, timestamp TEXT,
            open REAL, high REAL, low REAL, close REAL, volume BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

COT_SYMBOL_MAP = {
    "MES": "E-MINI S&P 500", "NQ": "NASDAQ-100 STOCK INDEX (MINI)",
    "GC": "GOLD", "ES": "E-MINI S&P 500", "MNQ": "NASDAQ-100 STOCK INDEX (MINI)",
}

class COTUpdate(BaseModel):
    report_date: str; symbol: str; commercial_long: int; commercial_short: int
    noncomm_long: int; noncomm_short: int; nonrep_long: int = 0; nonrep_short: int = 0
    open_interest: int

class OHLCVBar(BaseModel):
    time: str; open: float; high: float; low: float; close: float; volume: int

class PriceData(BaseModel):
    symbol: str; bars: List[OHLCVBar]

class SignalResponse(BaseModel):
    timestamp: str; symbol: str; bias: str; confidence: int; direction: str
    entry_price: Optional[float]; stop_price: Optional[float]; target_price: Optional[float]
    size_multiplier: float; reason: str

class COTAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.cot_name = COT_SYMBOL_MAP.get(symbol, symbol)
    
    def get_recent_cot(self, weeks: int = 4):
        conn = get_db(); c = conn.cursor()
        c.execute("SELECT * FROM cot_data WHERE symbol = %s ORDER BY report_date DESC LIMIT %s", (self.symbol, weeks))
        rows = c.fetchall(); cols = [desc[0] for desc in c.description]
        conn.close()
        return [dict(zip(cols, r)) for r in rows]
    
    def calculate_bias(self) -> Dict[str, Any]:
        data = self.get_recent_cot(weeks=4)
        if len(data) == 0:
            return {"bias": "NEUTRAL", "confidence": 0, "reason": "No COT data"}
        
        latest = data[0]
        comm_net_now = int(latest['commercial_long']) - int(latest['commercial_short'])
        noncomm_net_now = int(latest['noncomm_long']) - int(latest['noncomm_short'])
        oi_now = int(latest['open_interest'])
        
        comm_trend = comm_net_now
        if len(data) >= 2:
            prev = data[1]
            comm_net_prev = int(prev['commercial_long']) - int(prev['commercial_short'])
            comm_trend = comm_net_now - comm_net_prev
        
        noncomm_extreme = abs(noncomm_net_now) / oi_now if oi_now > 0 else 0.0
        
        bias = "NEUTRAL"; confidence = 50; reasons = []
        
        if comm_trend > 0:
            bias = "BULLISH"; confidence += 20
            reasons.append(f"Commercials accumulating (+{comm_trend:,} net)")
        elif comm_trend < 0:
            bias = "BEARISH"; confidence += 20
            reasons.append(f"Commercials distributing ({comm_trend:,} net)")
        
        if noncomm_extreme > 0.25 and noncomm_net_now > 0:
            if bias == "BULLISH": confidence -= 15; reasons.append("Specs overcrowded long (caution)")
            else: bias = "BEARISH"; confidence = 75; reasons.append("Specs extreme long + commercials selling")
        elif noncomm_extreme > 0.25 and noncomm_net_now < 0:
            if bias == "BEARISH": confidence -= 15; reasons.append("Specs overcrowded short (caution)")
            else: bias = "BULLISH"; confidence = 75; reasons.append("Specs extreme short + commercials buying")
        
        if comm_net_now > 0 and comm_trend >= 0: confidence += 10; reasons.append("Commercials net long")
        elif comm_net_now < 0 and comm_trend <= 0: confidence += 10; reasons.append("Commercials net short")
        
        confidence = max(0, min(100, confidence))
        
        return {
            "bias": bias, "confidence": confidence,
            "commercial_net": comm_net_now, "commercial_trend": comm_trend,
            "noncomm_net": noncomm_net_now, "noncomm_extreme_pct": round(float(noncomm_extreme) * 100, 1),
            "open_interest": oi_now, "report_date": latest['report_date'],
            "reason": " | ".join(reasons)
        }

class PriceAnalyzer:
    def __init__(self, bars: List[Dict]):
        self.bars = bars
        self.df = None
        if len(bars) >= 20:
            try:
                self._build_df()
            except Exception as e:
                print(f"PriceAnalyzer build_df error: {e}")
                self.df = None
    
    def _build_df(self):
        # Ensure all numeric columns are Python floats
        clean_bars = []
        for b in self.bars:
            clean_bars.append({
                'time': str(b.get('time', '')),
                'open': float(b.get('open', 0)),
                'high': float(b.get('high', 0)),
                'low': float(b.get('low', 0)),
                'close': float(b.get('close', 0)),
                'volume': int(b.get('volume', 0))
            })
        self.df = pd.DataFrame(clean_bars)
        self.df['ema20'] = self.df['close'].ewm(span=20, adjust=False).mean()
        self.df['ema50'] = self.df['close'].ewm(span=50, adjust=False).mean()
        self.df['atr'] = self._calc_atr(14)
    
    def _calc_atr(self, period: int):
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
        c = float(last['close']); e20 = float(last['ema20']); e50 = float(last['ema50'])
        if c > e20 > e50: return "UPTREND"
        elif c < e20 < e50: return "DOWNTREND"
        return "CHOPPY"
    
    def get_structure(self) -> Dict:
        if self.df is None or len(self.df) < 10:
            return {"swing_high": None, "swing_low": None, "atr": 0.0, "last_close": 0.0}
        
        recent = self.df.tail(20)
        swing_high = float(recent['high'].max())
        swing_low = float(recent['low'].min())
        
        atr_val = self.df['atr'].iloc[-1]
        if pd.isna(atr_val):
            atr = (swing_high - swing_low) * 0.1
        else:
            atr = float(atr_val)
        
        return {
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "atr": round(atr, 2),
            "last_close": round(float(self.df['close'].iloc[-1]), 2)
        }

class SignalGenerator:
    def __init__(self, symbol: str, cot_analyzer: COTAnalyzer, price_analyzer: PriceAnalyzer):
        self.symbol = symbol; self.cot = cot_analyzer; self.price = price_analyzer
    
    def generate(self) -> SignalResponse:
        now = datetime.utcnow().isoformat()
        cot_result = self.cot.calculate_bias()
        bias = cot_result['bias']
        confidence = cot_result['confidence']
        price_trend = self.price.get_trend()
        structure = self.price.get_structure()
        
        direction = "NONE"; entry = None; stop = None; target = None; size = 0.0
        reason = cot_result['reason']
        
        if bias == "BULLISH" and price_trend in ["UPTREND", "CHOPPY"]:
            if structure['swing_low'] and structure['atr'] and structure['atr'] > 0:
                direction = "LONG"
                entry = float(structure['last_close'])
                stop = float(structure['swing_low']) - (float(structure['atr']) * 0.5)
                stop = round(stop, 2)
                risk = entry - stop
                target = round(entry + (risk * 2), 2)
                size = self._calculate_size(confidence)
                reason += f" | Price: {price_trend}, Entry {entry}, Stop {stop}, Target {target}"
        
        elif bias == "BEARISH" and price_trend in ["DOWNTREND", "CHOPPY"]:
            if structure['swing_high'] and structure['atr'] and structure['atr'] > 0:
                direction = "SHORT"
                entry = float(structure['last_close'])
                stop = float(structure['swing_high']) + (float(structure['atr']) * 0.5)
                stop = round(stop, 2)
                risk = stop - entry
                target = round(entry - (risk * 2), 2)
                size = self._calculate_size(confidence)
                reason += f" | Price: {price_trend}, Entry {entry}, Stop {stop}, Target {target}"
        
        else:
            reason += f" | Price trend {price_trend} conflicts with COT bias {bias}. NO TRADE."
        
        # Cast everything to native Python types for PostgreSQL
        entry_f = float(entry) if entry is not None else None
        stop_f = float(stop) if stop is not None else None
        target_f = float(target) if target is not None else None
        size_f = float(size)
        
        conn = get_db(); c = conn.cursor()
        c.execute(
            """INSERT INTO signals (timestamp, symbol, bias, confidence, direction, entry_price, stop_price, target_price, size_multiplier, reason)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (now, self.symbol, bias, confidence, direction, entry_f, stop_f, target_f, size_f, reason)
        )
        conn.commit(); conn.close()
        
        return SignalResponse(
            timestamp=now, symbol=self.symbol, bias=bias, confidence=confidence,
            direction=direction, entry_price=entry_f, stop_price=stop_f,
            target_price=target_f, size_multiplier=size_f, reason=reason
        )
    
    def _calculate_size(self, confidence: int) -> float:
        if confidence >= 80: return 2.0
        elif confidence >= 65: return 1.5
        elif confidence >= 50: return 1.0
        else: return 0.5

@app.get("/health")
def health():
    return {"status": "ok", "system": "FlowAlert Institutional", "version": "2.0-pg-debug"}

@app.post("/update-cot")
def update_cot(data: COTUpdate):
    conn = get_db(); c = conn.cursor()
    try:
        c.execute(
            """INSERT INTO cot_data (report_date, symbol, commercial_long, commercial_short, noncomm_long, noncomm_short, nonrep_long, nonrep_short, open_interest)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (report_date, symbol) DO UPDATE SET
               commercial_long = EXCLUDED.commercial_long, commercial_short = EXCLUDED.commercial_short,
               noncomm_long = EXCLUDED.noncomm_long, noncomm_short = EXCLUDED.noncomm_short,
               nonrep_long = EXCLUDED.nonrep_long, nonrep_short = EXCLUDED.nonrep_short,
               open_interest = EXCLUDED.open_interest, created_at = CURRENT_TIMESTAMP""",
            (data.report_date, data.symbol, data.commercial_long, data.commercial_short,
             data.noncomm_long, data.noncomm_short, data.nonrep_long, data.nonrep_short, data.open_interest)
        )
        conn.commit()
        analyzer = COTAnalyzer(data.symbol)
        bias = analyzer.calculate_bias()
        return {"status": "saved", "symbol": data.symbol, "report_date": data.report_date, "bias": bias['bias'], "confidence": bias['confidence']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/ingest-price")
def ingest_price(data: PriceData):
    conn = get_db(); c = conn.cursor()
    for bar in data.bars:
        c.execute(
            "INSERT INTO price_snapshots (symbol, timestamp, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (data.symbol, bar.time, float(bar.open), float(bar.high), float(bar.low), float(bar.close), int(bar.volume))
        )
    conn.commit(); conn.close()
    return {"status": "saved", "bars": len(data.bars), "symbol": data.symbol}

@app.get("/signal", response_model=SignalResponse)
def get_signal(symbol: str = "MES"):
    try:
        conn = get_db(); c = conn.cursor()
        c.execute(
            "SELECT timestamp as time, open, high, low, close, volume FROM price_snapshots WHERE symbol = %s ORDER BY timestamp DESC LIMIT 50",
            (symbol,)
        )
        rows = c.fetchall(); cols = [desc[0] for desc in c.description]
        conn.close()
        
        bars = [dict(zip(cols, r)) for r in reversed(rows)]
        
        if len(bars) < 20:
            return SignalResponse(
                timestamp=datetime.utcnow().isoformat(), symbol=symbol,
                bias="NEUTRAL", confidence=0, direction="NONE",
                entry_price=None, stop_price=None, target_price=None,
                size_multiplier=0.0, reason=f"Need at least 20 price bars. Have {len(bars)}. Send data to /ingest-price first."
            )
        
        cot = COTAnalyzer(symbol)
        price = PriceAnalyzer(bars)
        gen = SignalGenerator(symbol, cot, price)
        return gen.generate()
    
    except Exception as e:
        error_detail = f"{str(e)}\\n{traceback.format_exc()}"
        print(f"SIGNAL ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/signal-debug")
def signal_debug(symbol: str = "MES"):
    """Debug endpoint - shows raw data without generating signal"""
    try:
        conn = get_db(); c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM price_snapshots WHERE symbol = %s", (symbol,))
        count = c.fetchone()[0]
        
        c.execute(
            "SELECT timestamp as time, open, high, low, close, volume FROM price_snapshots WHERE symbol = %s ORDER BY timestamp DESC LIMIT 5",
            (symbol,)
        )
        rows = c.fetchall(); cols = [desc[0] for desc in c.description]
        conn.close()
        
        recent = [dict(zip(cols, r)) for r in rows]
        
        cot = COTAnalyzer(symbol)
        cot_bias = cot.calculate_bias()
        
        return {
            "symbol": symbol,
            "price_bar_count": count,
            "recent_bars": recent,
            "cot_bias": cot_bias,
            "status": "ok"
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/bias/{symbol}")
def get_bias(symbol: str):
    analyzer = COTAnalyzer(symbol)
    return analyzer.calculate_bias()

@app.get("/signals")
def get_signals(symbol: Optional[str] = None, limit: int = 20):
    conn = get_db(); c = conn.cursor()
    query = "SELECT * FROM signals WHERE 1=1"; params = []
    if symbol: query += " AND symbol = %s"; params.append(symbol)
    query += " ORDER BY created_at DESC LIMIT %s"; params.append(limit)
    c.execute(query, params)
    rows = c.fetchall(); cols = [desc[0] for desc in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT * FROM signals ORDER BY created_at DESC LIMIT 10")
    signals = c.fetchall(); sig_cols = [desc[0] for desc in c.description]
    c.execute("SELECT * FROM cot_data ORDER BY report_date DESC LIMIT 5")
    cot_rows = c.fetchall(); cot_cols = [desc[0] for desc in c.description]
    conn.close()
    
    signals_list = [dict(zip(sig_cols, r)) for r in signals]
    cot_list = [dict(zip(cot_cols, r)) for r in cot_rows]
    
    signals_html = "".join([
        f"<tr><td>{s['timestamp'][:19]}</td><td>{s['symbol']}</td><td>{s['bias']}</td>"
        f"<td>{s['confidence']}%</td><td><b>{s['direction']}</b></td>"
        f"<td>{s['entry_price']}</td><td>{s['stop_price']}</td><td>{s['target_price']}</td></tr>"
        for s in signals_list
    ])
    
    cot_html = "".join([
        f"<tr><td>{c['report_date']}</td><td>{c['symbol']}</td>"
        f"<td>{c['commercial_long']:,}</td><td>{c['commercial_short']:,}</td>"
        f"<td>{c['noncomm_long']:,}</td><td>{c['noncomm_short']:,}</td></tr>"
        for c in cot_list
    ])
    
    return f"""
    <html><head><title>FlowAlert Institutional</title>
    <style>
        body {{ font-family: monospace; background: #0a0a0a; color: #00ff88; padding: 20px; }}
        h1 {{ color: #ffd700; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #333; padding: 8px; text-align: left; }}
        th {{ background: #1a1a1a; color: #ffd700; }}
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
        <p><a href="/health" style="color:#ffd700">/health</a> | <a href="/signal-debug?symbol=NQ" style="color:#ffd700">/signal-debug?symbol=NQ</a> | <a href="/bias/NQ" style="color:#ffd700">/bias/NQ</a></p>
    </body></html>
    """

@app.post("/report-fill")
def report_fill(signal_id: int, pnl: float):
    conn = get_db(); c = conn.cursor()
    c.execute("UPDATE signals SET executed = TRUE, pnl = %s WHERE id = %s", (pnl, signal_id))
    conn.commit(); conn.close()
    return {"status": "updated"}

@app.get("/cot-template/{symbol}")
def cot_template(symbol: str):
    return {
        "report_date": "2026-08-05", "symbol": symbol,
        "commercial_long": 0, "commercial_short": 0,
        "noncomm_long": 0, "noncomm_short": 0,
        "nonrep_long": 0, "nonrep_short": 0, "open_interest": 0
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
