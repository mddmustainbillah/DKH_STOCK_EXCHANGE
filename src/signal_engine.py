"""
DSE Sector Signal Engine
Generates monthly Buy / Hold / Sell signals for all DSE sectors.
Combines momentum, relative strength, and an XGBoost ML model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import xgboost as xgb


class DSESectorSignalEngine:
    """
    Trains on historical sector returns and produces a ranked signal table.

    Usage:
        engine = DSESectorSignalEngine(sector_returns)
        engine.train_ml(cutoff_date='2018-12-31')
        signals = engine.generate_signals()
    """

    def __init__(self, sector_returns_df, risk_free_rate=0.075):
        self.sr = sector_returns_df
        self.rf = risk_free_rate / 252
        # Resample daily returns to monthly for feature building
        self.monthly = sector_returns_df.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        self.market = self.monthly.mean(axis=1)   # equal-weight market proxy
        self._ml_model = None
        self._scaler = None
        self._feature_cols = None

    # ── internal helpers ──────────────────────────────────────────────────

    def _momentum(self, series, months):
        """Compound return over the last N months."""
        return series.rolling(months).apply(lambda x: (1 + x).prod() - 1)

    def _relative_strength(self, series, market, months):
        """How much the sector beat (or lagged) the market over N months."""
        return self._momentum(series, months) - self._momentum(market, months)

    def _build_features(self, sector):
        """Build the 8-column feature DataFrame for one sector."""
        s = self.monthly[sector]
        m = self.market
        return pd.DataFrame({
            'mom_1m':           s,
            'mom_3m':           self._momentum(s, 3),
            'mom_6m':           self._momentum(s, 6),
            'mom_12m_minus_1m': self._momentum(s, 12) - s,   # long-term minus noise
            'rs_1m':            self._relative_strength(s, m, 1),
            'rs_3m':            self._relative_strength(s, m, 3),
            'vol_ratio':        s.rolling(3).std() / s.rolling(6).std(),
            'month':            s.index.month,
        })

    # ── public methods ────────────────────────────────────────────────────

    def train_ml(self, cutoff_date):
        """Train XGBoost to predict which sectors will rank in the top-3 next month."""
        monthly_ranks = self.monthly.rank(axis=1, ascending=False)
        top3_label = (monthly_ranks <= 3).shift(-1)   # predict next month's top-3

        feature_cols = [
            'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m_minus_1m',
            'rs_1m', 'rs_3m', 'vol_ratio', 'month'
        ]
        X_list, y_list = [], []

        for s in self.sr.columns:
            feat = self._build_features(s)[feature_cols]
            target = top3_label[s]
            merged = feat.join(target.rename('target')).dropna()
            merged = merged[merged.index <= pd.Timestamp(cutoff_date)]
            if len(merged) < 10:
                continue
            X_list.append(merged[feature_cols])
            y_list.append(merged['target'].astype(int))

        if not X_list:
            return

        X = pd.concat(X_list)
        y = pd.concat(y_list)

        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Up-weight the rare top-3 class to handle imbalance (only 16.7% positives)
        pos_weight = (y == 0).sum() / (y == 1).sum()
        self._ml_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_weight,
            random_state=42, verbosity=0
        )
        self._ml_model.fit(X_scaled, y)
        self._feature_cols = feature_cols

    def generate_signals(self, as_of_date=None):
        """
        Return a DataFrame with Composite Score, Signal, and Confidence for each sector.
        as_of_date defaults to the last date in the data.
        """
        if as_of_date is None:
            as_of_date = self.sr.index[-1]
        as_of = pd.Timestamp(as_of_date)

        rows = []
        for s in self.sr.columns:
            feat = self._build_features(s)
            feat = feat[feat.index <= as_of].tail(1)
            if feat.empty:
                continue

            row = feat.iloc[0]
            # Momentum score: weight 3-month most heavily
            mom = row.get('mom_3m', 0) * 0.5 + row.get('mom_6m', 0) * 0.3 + row.get('mom_1m', 0) * 0.2
            rs = row.get('rs_3m', 0) * 0.6 + row.get('rs_1m', 0) * 0.4

            ml_prob = 0.5   # neutral default when no model is available
            if self._ml_model is not None and self._scaler is not None:
                try:
                    x_scaled = self._scaler.transform(feat[self._feature_cols].values)
                    ml_prob = self._ml_model.predict_proba(x_scaled)[0, 1]
                except Exception:
                    pass

            rows.append({'Sector': s, 'mom_score': mom, 'rs_score': rs, 'ml_prob': ml_prob})

        result = pd.DataFrame(rows).set_index('Sector')

        # Normalise each component to [0, 1] so they combine fairly
        for col in ['mom_score', 'rs_score', 'ml_prob']:
            lo, hi = result[col].min(), result[col].max()
            if hi > lo:
                result[col] = (result[col] - lo) / (hi - lo)

        # Final composite: 35% momentum + 30% relative strength + 35% ML
        result['Composite Score'] = (
            0.35 * result['mom_score'] +
            0.30 * result['rs_score'] +
            0.35 * result['ml_prob']
        ) * 100

        result['Signal'] = pd.cut(
            result['Composite Score'],
            bins=[0, 35, 65, 100],
            labels=['SELL', 'HOLD', 'BUY']
        )

        result['Confidence'] = pd.cut(
            result['Composite Score'].apply(lambda x: abs(x - 50) * 2),
            bins=[0, 33, 66, 100],
            labels=['Low', 'Medium', 'High']
        )

        return result[['Composite Score', 'Signal', 'Confidence']].sort_values(
            'Composite Score', ascending=False
        )