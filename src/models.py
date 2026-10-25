"""Walk-forward forecasting: every model is refit using only data whose targets were already known."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

LGB_REG = dict(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=60,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
LGB_CLF = dict(LGB_REG, n_estimators=300)


def month_starts(index, start):
    idx = index[index >= start]
    return list(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).min())


def walk_forward(df, feats, target, fit_predict, start, horizon, train_from=None, every=1, min_train=400):
    """Refit at the first trading day of every `every`-th month from `start`.

    A row t is used for training only if its target window (t+1 .. t+horizon) ends on or before
    the refit date, so nothing after the refit date leaks into the model. Predictions for each day
    in the following block come from the model fitted at the start of that block.
    """
    pos = pd.Series(np.arange(len(df)), index=df.index)
    refits = month_starts(df.index, start)[::every]
    usable = df[feats].notna().all(axis=1)
    preds = pd.Series(np.nan, index=df.index)
    for i, d in enumerate(refits):
        end = refits[i + 1] if i + 1 < len(refits) else df.index[-1] + pd.Timedelta(days=1)
        cut = pos[d] - horizon                              # last row whose target is known at d
        tr = df.iloc[: cut + 1]
        tr = tr[usable.iloc[: cut + 1] & tr[target].notna()]
        if train_from is not None:
            tr = tr[tr.index >= train_from]
        if len(tr) < min_train:
            continue
        te_mask = (df.index >= d) & (df.index < end) & usable
        if te_mask.sum() == 0:
            continue
        preds[te_mask] = fit_predict(tr[feats].values, tr[target].values, df.loc[te_mask, feats].values)
    return preds


# ---- fit/predict functions ---------------------------------------------------------
def ols(Xtr, ytr, Xte):
    return LinearRegression().fit(Xtr, ytr).predict(Xte)


def lgb_reg(Xtr, ytr, Xte):
    return lgb.LGBMRegressor(random_state=0, **LGB_REG).fit(Xtr, ytr).predict(Xte)


def logit(Xtr, ytr, Xte):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def lgb_clf(Xtr, ytr, Xte):
    return lgb.LGBMClassifier(random_state=0, **LGB_CLF).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def base_rate(Xtr, ytr, Xte):
    return np.full(len(Xte), ytr.mean())
