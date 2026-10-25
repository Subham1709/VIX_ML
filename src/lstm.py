"""A small two-headed LSTM: reads the last 63 days of features and predicts
(1) log realised volatility over the next 21 days and (2) the chance of a 7% fall within 63 days."""
import numpy as np
import pandas as pd
import torch
from torch import nn

SEQ = 63
torch.set_num_threads(4)


class Net(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.head_rv = nn.Linear(hidden, 1)
        self.head_dd = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.drop(self.lstm(x)[0][:, -1])
        return self.head_rv(h).squeeze(-1), self.head_dd(h).squeeze(-1)


def windows(X, rows):
    return np.stack([X[r - SEQ + 1: r + 1] for r in rows]).astype(np.float32)


def fit_predict(X, y_rv, y_dd, tr_rows, te_rows, seed, epochs=40, patience=6):
    """X: full feature matrix (scaled with training statistics); rows index into it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_val = int(len(tr_rows) * 0.15)
    fit_rows, val_rows = tr_rows[: -n_val - 63], tr_rows[-n_val:]   # 63-row gap so val targets don't overlap
    mu, sd = y_rv[fit_rows].mean(), y_rv[fit_rows].std()
    def tensors(rows):
        return (torch.from_numpy(windows(X, rows)), torch.tensor((y_rv[rows] - mu) / sd, dtype=torch.float32),
                torch.tensor(y_dd[rows], dtype=torch.float32))
    xf, rf, df_ = tensors(fit_rows)
    xv, rv, dv = tensors(val_rows)
    net = Net(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xf))
        for b in range(0, len(xf), 128):
            i = perm[b: b + 128]
            prv, pdd = net(xf[i])
            loss = mse(prv, rf[i]) + bce(pdd, df_[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            prv, pdd = net(xv)
            vl = (mse(prv, rv) + bce(pdd, dv)).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        prv, pdd = net(torch.from_numpy(windows(X, te_rows)))
    return prv.numpy() * sd + mu, torch.sigmoid(pdd).numpy()


def walk_forward(df, feats, start, every=3, seeds=(0, 1, 2), horizon=63, log=print):
    """Refit every `every` months on all rows whose 63-day target window has closed."""
    from models import month_starts
    Xraw = df[feats].values
    ok = ~np.isnan(Xraw).any(axis=1)
    first_ok = np.argmax(ok)
    valid_seq = np.zeros(len(df), bool)
    valid_seq[first_ok + SEQ - 1:] = ok[first_ok + SEQ - 1:]
    y_rv, y_dd = df.y_rv.values, df.y_dd.values
    has_y = ~np.isnan(y_rv) & ~np.isnan(y_dd)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    refits = month_starts(df.index, start)[::every]
    out_rv = pd.Series(np.nan, index=df.index)
    out_dd = pd.Series(np.nan, index=df.index)
    for i, d in enumerate(refits):
        end = refits[i + 1] if i + 1 < len(refits) else df.index[-1] + pd.Timedelta(days=1)
        cut = pos[d] - horizon
        tr_rows = np.where(valid_seq[: cut + 1] & has_y[: cut + 1])[0]
        te_rows = np.where((df.index >= d) & (df.index < end) & valid_seq)[0]
        mu, sd = Xraw[tr_rows].mean(0), Xraw[tr_rows].std(0) + 1e-9
        X = np.nan_to_num((Xraw - mu) / sd)
        res = [fit_predict(X, y_rv, y_dd, tr_rows, te_rows, s) for s in seeds]
        out_rv.iloc[te_rows] = np.mean([r[0] for r in res], axis=0)
        out_dd.iloc[te_rows] = np.mean([r[1] for r in res], axis=0)
        log(f"  LSTM refit {d.date()}: trained on {len(tr_rows)} days, predicted {len(te_rows)}")
    return out_rv, out_dd
