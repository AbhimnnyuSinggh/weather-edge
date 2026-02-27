-- Weather-Edge Bot — Database Schema (PostgreSQL)
-- 11 tables total

-- Station registry: maps cities to ICAO stations with metadata
CREATE TABLE IF NOT EXISTS stations (
    icao TEXT PRIMARY KEY,
    city TEXT NOT NULL,
    country TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    timezone TEXT NOT NULL,
    unit TEXT NOT NULL DEFAULT 'C',
    wunderground_url TEXT,
    is_coastal BOOLEAN DEFAULT FALSE,
    bias_gfs REAL DEFAULT 0.0,
    bias_ecmwf REAL DEFAULT 0.0,
    bias_icon REAL DEFAULT 0.0,
    bias_nws REAL DEFAULT 0.0,
    active BOOLEAN DEFAULT TRUE,
    added_at TIMESTAMP DEFAULT NOW()
);

-- METAR observations: every reading fetched
CREATE TABLE IF NOT EXISTS metar_readings (
    id SERIAL PRIMARY KEY,
    station TEXT NOT NULL REFERENCES stations(icao),
    observed_at TIMESTAMP NOT NULL,
    fetched_at TIMESTAMP DEFAULT NOW(),
    temp_c REAL,
    temp_f REAL,
    dewpoint_c REAL,
    wind_dir INTEGER,
    wind_speed_kt INTEGER,
    wind_gust_kt INTEGER,
    visibility_m REAL,
    t_group_temp_c REAL,
    raw_metar TEXT,
    UNIQUE(station, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_metar_station_date ON metar_readings(station, observed_at DESC);

-- Model forecasts: every forecast fetched for tracking accuracy
CREATE TABLE IF NOT EXISTS model_forecasts (
    id SERIAL PRIMARY KEY,
    station TEXT NOT NULL REFERENCES stations(icao),
    target_date DATE NOT NULL,
    model_name TEXT NOT NULL,
    raw_high_c REAL,
    raw_high_f REAL,
    bias_corrected_c REAL,
    bias_corrected_f REAL,
    fetched_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(station, target_date, model_name, fetched_at)
);
CREATE INDEX IF NOT EXISTS idx_forecast_station_date ON model_forecasts(station, target_date);

-- Market snapshots: Polymarket prices captured periodically
CREATE TABLE IF NOT EXISTS market_snapshots (
    id SERIAL PRIMARY KEY,
    market_id TEXT NOT NULL,
    station TEXT REFERENCES stations(icao),
    target_date DATE,
    bin_label TEXT NOT NULL,
    bin_low REAL,
    bin_high REAL,
    yes_price REAL NOT NULL,
    volume REAL,
    captured_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_snapshot_market ON market_snapshots(market_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_snapshot_station_date ON market_snapshots(station, target_date, captured_at DESC);

-- Trades: every position tracked against wallet
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    station TEXT NOT NULL REFERENCES stations(icao),
    target_date DATE NOT NULL,
    bin_label TEXT NOT NULL,
    side TEXT NOT NULL,
    trade_type TEXT NOT NULL,
    entry_price REAL NOT NULL,
    shares REAL NOT NULL,
    cost REAL NOT NULL,
    confidence_score INTEGER,
    confidence_components JSONB,
    ev_at_entry REAL,
    alert_id TEXT,
    opened_at TIMESTAMP DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    outcome TEXT,
    payout REAL,
    profit_loss REAL,
    actual_high_c REAL,
    actual_high_f REAL,
    winning_bin TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(resolved, station);
CREATE INDEX IF NOT EXISTS idx_trades_station ON trades(station, opened_at DESC);

-- Model accuracy: running EWMA accuracy per model per station per month
CREATE TABLE IF NOT EXISTS model_accuracy (
    id SERIAL PRIMARY KEY,
    station TEXT NOT NULL REFERENCES stations(icao),
    model_name TEXT NOT NULL,
    year_month TEXT NOT NULL,
    ewma_error REAL DEFAULT 0.0,
    ewma_bias REAL DEFAULT 0.0,
    sample_count INTEGER DEFAULT 0,
    weight REAL DEFAULT 0.25,
    last_updated TIMESTAMP DEFAULT NOW(),
    UNIQUE(station, model_name, year_month)
);

-- Daily summary: one row per day for growth tracking
CREATE TABLE IF NOT EXISTS daily_summary (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    starting_capital REAL,
    ending_capital REAL,
    total_deployed REAL,
    trades_taken INTEGER DEFAULT 0,
    trades_won INTEGER DEFAULT 0,
    trades_lost INTEGER DEFAULT 0,
    trades_void INTEGER DEFAULT 0,
    gross_profit REAL DEFAULT 0.0,
    gross_loss REAL DEFAULT 0.0,
    net_pnl REAL DEFAULT 0.0,
    alerts_sent INTEGER DEFAULT 0,
    alerts_skipped INTEGER DEFAULT 0,
    best_trade_pnl REAL,
    worst_trade_pnl REAL
);

-- Alerts log: every alert sent
CREATE TABLE IF NOT EXISTS alerts_log (
    id SERIAL PRIMARY KEY,
    alert_id TEXT UNIQUE NOT NULL,
    station TEXT NOT NULL,
    trade_type TEXT NOT NULL,
    bin_label TEXT,
    side TEXT,
    suggested_price REAL,
    suggested_shares REAL,
    suggested_cost REAL,
    confidence_score INTEGER,
    ev REAL,
    was_acted_on BOOLEAN DEFAULT FALSE,
    actual_outcome TEXT,
    sent_at TIMESTAMP DEFAULT NOW()
);

-- Capital reservations: temporary holds to prevent race conditions
CREATE TABLE IF NOT EXISTS capital_reservations (
    id SERIAL PRIMARY KEY,
    amount REAL NOT NULL,
    reason TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    consumed BOOLEAN DEFAULT FALSE
);

-- Historical time-of-high: stores what hour the daily high typically occurs
CREATE TABLE IF NOT EXISTS time_of_high_stats (
    id SERIAL PRIMARY KEY,
    station TEXT NOT NULL REFERENCES stations(icao),
    month INTEGER NOT NULL,
    hour_local INTEGER NOT NULL,
    frequency REAL NOT NULL,
    sample_count INTEGER DEFAULT 0,
    UNIQUE(station, month, hour_local)
);

-- Max warming rates: historical maximum warming rates per station per month
CREATE TABLE IF NOT EXISTS warming_rates (
    id SERIAL PRIMARY KEY,
    station TEXT NOT NULL REFERENCES stations(icao),
    month INTEGER NOT NULL,
    max_warming_rate_c_per_hr REAL,
    p95_warming_rate_c_per_hr REAL,
    p75_warming_rate_c_per_hr REAL,
    sample_days INTEGER DEFAULT 0,
    UNIQUE(station, month)
);
