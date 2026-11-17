"""GTA VI Twitch CCU launch-day forecaster v3 (Python 3.13, standard library only).

Forecasts the 24 UTC hours of Grand Theft Auto VI's release day on Twitch,
hour by hour, as five quantiles an hour, from everything recorded up to the
end of hour 1 (that hour's print included). One entrypoint, main.py.

Model — a hierarchical launch-day curve, fit at run time from the case's
own comparables (past big launches' hourly Twitch curves):
  * The shape is a shared log-ratio curve y_h = log(v(h) / v(1)), fit as a
    function of launch size x = log(v(1)) by two views that are blended in
    log space: a local-linear fit (bandwidth 1.8, PC-platform launches
    downweighted) and a kernel-weighted median ratio (bandwidth 2.0 / 1.5 /
    1.0 by hour). Blend 70% local-linear / 30% median. The centre is never
    extrapolated past the largest launch in the pool.
  * The spread is the local dispersion of leave-one-out residuals around
    the centre (kernel-weighted MAD, bandwidth 1.25), so small launches —
    whose curves are wildly more dispersed — get far wider quantiles than
    big ones. Gaussian quantiles, spread x0.80.
  * Segment levels: when the target is a console-only launch, PC-platform
    training launches are downweighted (x0.5; x0.1 weight left for a small
    console target, whose centre would otherwise be dragged up by
    explosive small PC launches). A launch-day curve is a console curve
    or a PC curve first, a size second.
  * Misalignment guard: a launch whose UTC hour 1 was pre-unlock quiet
    can explode 1000x later. For a small target (hour 1 under 5,000) the
    upper quantiles are floored at the small-launch pool's own 0.75 / 0.95
    quantiles, so the forecast never claims a thin upper tail it cannot
    defend. (Pinball punishes a missed upper tail ~19x more than an
    over-wide one.)
  * Unlock alignment: when the target's unlock text names a single global
    UTC unlock hour later in the day (a surge hour s > 1), the forecast
    switches to surge-aligned time. The surge LEVEL v(s) is predicted by
    its median over training launches with a parseable surge hour (the
    level clusters near ~150k viewers while the ratio v(s)/v(1) spans
    0.8x..2024x, so the level — not the ratio — is the predictable
    quantity), the post-surge shape is fit on surge-aligned training
    curves, and the surge-level uncertainty is added in quadrature.
    Pre-surge hours keep the ordinary forecast. Rolling-midnight launches
    (surge hour 1) and unknown unlocks use the ordinary forecast.

Every parameter above was chosen by leave-one-out validation on the kit's
37 comparable launches (segments: all / big h1>=20k / console-midnight
family / console-only), never by feeling:
  LOO v3: all 0.3270, big 0.2500, family 0.1721, console 0.1701
  LOO v2: all 0.3786, big 0.2485, family 0.2752, console 0.2085
  persistence baseline LOO: all 0.9296, big 0.3214, console 0.3225

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
HOURS = list(range(2, 25))

# Frozen parameters (measured by LOO on the kit's 37 comparables; see docstring).
BW_LL = 1.8
W_PC_EARLY = 0.5          # PC-platform weight in the local-linear fit, hours <= 5
W_PC_LATE = 0.15          # PC-platform weight in the local-linear fit, hours >= 6
BW_RQ_EARLY = 2.0         # median-ratio bandwidth, hours <= 6
BW_RQ_MID = 1.5           # median-ratio bandwidth, hours 7..12
BW_RQ_LATE = 1.0          # median-ratio bandwidth, hours >= 13
ALPHA = 0.7               # local-linear weight in the centre blend
SPREAD_MULT = 0.80
TAIL_FLOOR_MAX = 5000     # tail floor applies only to targets with hour 1 below this
SIGMA_BW = 1.25           # kernel bandwidth of the local residual dispersion
CONSOLE_GATE = 0.5        # PC weight left for a console target (1 - 0.5)
CONSOLE_GATE_SMALL = 0.9  # PC weight left for a SMALL console target (1 - 0.9)
SMALL_H1 = 20000          # "small launch" threshold, hour-1 viewers


def parse_time(t):
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _kernel(u):
    return math.exp(-0.5 * u * u)


def _wmean(xs, ws):
    sw = sum(ws)
    return sum(x * w for x, w in zip(xs, ws)) / sw if sw > 0 else float("nan")


def _wquantile(pairs, tau):
    """Weighted tau-quantile of (value, weight) pairs."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return float("nan")
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tau * tot:
            return v
    return pairs[-1][0]


def _ll_centre(xs, ys, ws, x0, bw):
    """Local-linear centre at x0 (weighted least squares, Gaussian kernel)."""
    kw = [_kernel((x - x0) / bw) * w for x, w in zip(xs, ws)]
    sw = sum(kw)
    if sw < 1e-9:
        return None
    xm = _wmean(xs, kw)
    ym = _wmean(ys, kw)
    sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, kw))
    sxy = sum(w * (x - xm) * (y - ym) for x, y, w in zip(xs, ys, kw))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    return ym + b * (x0 - xm)


def _rq_centre(ys, xs, x0, bw, ws):
    """Kernel-weighted median of ys at x0."""
    kw = [_kernel((x - x0) / bw) * w for x, w in zip(xs, ws)]
    return _wquantile(list(zip(ys, kw)), 0.5)


def parse_surge(unlock, day_start):
    """Surge hour s (1..24) from unlock text, or None if unknown.

    s is the first UTC hour of the release day that is post-unlock
    (1-based: an unlock at 02:00Z -> s = 3). A rolling midnight-local
    launch -> 1 (the first region unlocks the UTC day before, so the
    surge is already underway at hour 1).
    """
    if not unlock:
        return None
    ds = parse_time(day_start)
    day_end = ds + timedelta(days=1)
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})", unlock):
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                      int(m.group(4)), int(m.group(5)), tzinfo=timezone.utc)
        if dt < ds:
            return 1
        if dt < day_end:
            return dt.hour + 1
        return None
    if "midnight local" in unlock.lower() or "12 am local" in unlock.lower() \
            or "12am local" in unlock.lower():
        return 1
    return None


def release_day_curve(comp):
    """A comparable's release-day curve: {hour 1..24: viewers} (+ channels)."""
    ds = parse_time(comp["day_start"])
    curve, chan = {}, {}
    for p in comp["hourly"]:
        t = parse_time(p["from"])
        h = int((t - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and p["viewers"] is not None:
            curve[h] = p["viewers"]
            chan[h] = p.get("channels")
    return curve, chan


def load_training(comparables):
    """Training pool: comparables with a usable release-day curve."""
    comps = []
    for c in comparables:
        cv, ch = release_day_curve(c)
        if len(cv) >= 20 and 1 in cv and cv[1] > 0:
            comps.append({
                "name": c["name"], "curve": cv, "channels": ch,
                "day_start": c["day_start"],
                "platforms": c.get("platforms", ""),
                "unlock": c.get("unlock", ""),
            })
    return comps


def forecast(p1, ch1, train, target_platforms=None, target_unlock=None,
             target_day_start=None):
    """Forecast hours 2..24 as {hour: [q05, q25, q50, q75, q95]} (viewers)."""
    if not train:
        return {h: [max(1.0, p1)] * 5 for h in HOURS}
    x0 = math.log(max(1.0, p1))
    xs = [math.log(max(1.0, t["curve"][1])) for t in train]
    xmax = max(xs)
    x0c = min(x0, xmax)
    target_console = target_platforms is not None and "PC" not in target_platforms

    def pc_factor(t):
        if target_console and "PC" in t["platforms"]:
            # A small console target gets the full gate: the explosive
            # small PC launches would otherwise drag its centre up to
            # ratios a console launch never reaches.
            g = CONSOLE_GATE_SMALL if p1 < SMALL_H1 else CONSOLE_GATE
            return 1.0 - g
        return 1.0

    out = {}
    for h in HOURS:
        ys, ok = [], []
        for t in train:
            cv = t["curve"]
            if h in cv and cv[h] > 0:
                ys.append(math.log(cv[h] / cv[1]))
                ok.append(t)
        if not ys:
            out[h] = None
            continue
        xs_h = [math.log(max(1.0, t["curve"][1])) for t in ok]
        ws_pc = []
        for t in ok:
            pc = "PC" in t["platforms"]
            w = (W_PC_EARLY if h <= 5 else W_PC_LATE) if pc else 1.0
            ws_pc.append(w * pc_factor(t))
        # Centre: local-linear (x0 capped at the pool's largest launch;
        # no low-side clipping — measured worse on LOO) + median ratio.
        ll = _ll_centre(xs_h, ys, ws_pc, x0c, BW_LL)
        if ll is None:
            ll = _wmean(ys, ws_pc)
        bw_rq = BW_RQ_EARLY if h <= 6 else (BW_RQ_MID if h <= 12 else BW_RQ_LATE)
        rq = _rq_centre(ys, xs_h, x0c, bw_rq, [pc_factor(t) for t in ok])
        if rq is None or (isinstance(rq, float) and math.isnan(rq)):
            rq = ll
        centre = ALPHA * ll + (1 - ALPHA) * rq
        # Spread: local dispersion of leave-one-out residuals of the
        # blended centre (kernel-weighted MAD around x0).
        res, res_x = [], []
        for i, t in enumerate(ok):
            loo_xs = xs_h[:i] + xs_h[i + 1:]
            loo_ys = ys[:i] + ys[i + 1:]
            loo_ws = ws_pc[:i] + ws_pc[i + 1:]
            xi = xs_h[i]
            xi_c = min(xi, max(loo_xs))
            li = _ll_centre(loo_xs, loo_ys, loo_ws, xi_c, BW_LL)
            if li is None:
                li = _wmean(loo_ys, loo_ws)
            loo_ok = ok[:i] + ok[i + 1:]
            ri = _rq_centre(loo_ys, loo_xs, xi_c, bw_rq,
                            [pc_factor(u) for u in loo_ok])
            if ri is None or (isinstance(ri, float) and math.isnan(ri)):
                ri = li
            ci = ALPHA * li + (1 - ALPHA) * ri
            res.append(ys[i] - ci)
            res_x.append(xi)
        kw = [_kernel((x - x0c) / SIGMA_BW) for x in res_x]
        med = _wquantile(list(zip(res, kw)), 0.5)
        mad = _wquantile(list(zip([abs(r - med) for r in res], kw)), 0.5)
        sigma = max(0.05, 1.4826 * mad)
        out[h] = (centre, sigma, ys, xs_h)
    # Smooth sigma across hours (3-hour moving average).
    sig = {}
    for h in HOURS:
        if out[h] is None:
            sig[h] = 0.15
            continue
        vals = [out[hh][1] for hh in (h - 1, h, h + 1)
                if hh in out and out[hh] is not None]
        sig[h] = sum(vals) / len(vals) if vals else out[h][1]
    # Tail floor from the small-launch pool, only for a small target.
    small = [t for t in train if t["curve"][1] < SMALL_H1]
    result = {}
    for h in HOURS:
        if out[h] is None:
            result[h] = [max(1.0, p1)] * 5
            continue
        centre, sigma = out[h][0], sig[h] * SPREAD_MULT
        qs = [centre + sigma * z for z in Z]
        if p1 < TAIL_FLOOR_MAX and small:
            sy = []
            for t in small:
                cv = t["curve"]
                if h in cv and cv[h] > 0:
                    sy.append(math.log(cv[h] / cv[1]))
            if len(sy) >= 3:
                sy.sort()

                def _q(v, tau):
                    i = min(len(v) - 1, max(0, int(tau * len(v))))
                    return v[i]

                qs[3] = max(qs[3], _q(sy, 0.75))
                qs[4] = max(qs[4], _q(sy, 0.95))
        vals = [max(1.0, p1 * math.exp(q)) for q in qs]
        for i in range(1, 5):
            vals[i] = max(vals[i], vals[i - 1])
        result[h] = [min(v, 1e9) for v in vals]
    al = forecast_aligned(p1, ch1, train, target_platforms,
                          target_unlock, target_day_start, result)
    return al if al is not None else result


def forecast_aligned(p1, ch1, train, target_platforms,
                     target_unlock, target_day_start, base_result):
    """Unlock-aligned forecast for a target with surge hour s_t > 1.

    Two stages: (1) the surge LEVEL v(s), predicted by its median over
    training launches with a parseable surge hour (the level clusters
    near ~150k viewers while the ratio v(s)/v(1) spans 0.8x..2024x, so
    the level — not the ratio — is the predictable quantity); (2) the
    post-surge shape z(a) = log(v(s+a-1)/v(s)), fit with the standard
    machinery on surge-aligned training curves. The surge-level
    uncertainty is added in quadrature. Pre-surge hours keep the
    standard forecast. Returns None when alignment does not apply.
    """
    s_t = parse_surge(target_unlock, target_day_start)
    if not s_t or s_t <= 1:
        return None
    log_vs = []
    train_aligned = []
    for t in train:
        s = parse_surge(t.get("unlock"), t.get("day_start"))
        ss = s if s else 1
        if s and s > 1 and s in t["curve"] and t["curve"][s] > 0:
            log_vs.append(math.log(t["curve"][s]))
        cv, ch = {}, {}
        for h, v in t["curve"].items():
            a = h - ss + 1
            if 1 <= a <= 24:
                cv[a] = v
                ch[a] = t["channels"].get(h)
        train_aligned.append({"name": t["name"], "curve": cv, "channels": ch,
                              "platforms": t["platforms"],
                              "unlock": t.get("unlock"),
                              "day_start": t.get("day_start")})
    if len(log_vs) < 2:
        return None
    log_v = statistics.median(log_vs)
    mad_v = statistics.median([abs(v - log_v) for v in log_vs])
    sigma_level = max(0.5, 1.4826 * mad_v)
    v_s = math.exp(log_v)
    al = forecast(v_s, ch1, train_aligned, target_platforms)
    out = {}
    for h in HOURS:
        if h < s_t:
            out[h] = base_result[h]
        elif h == s_t:
            out[h] = [max(1.0, v_s * math.exp(sigma_level * z)) for z in Z]
            for i in range(1, 5):
                out[h][i] = max(out[h][i], out[h][i - 1])
            out[h] = [min(v, 1e9) for v in out[h]]
        else:
            a = h - s_t + 1
            qs = al.get(a)
            if qs is None:
                out[h] = base_result[h]
                continue
            lq = [math.log(max(q, 1e-9)) for q in qs]
            sig_z = (lq[4] - lq[0]) / (2 * 1.6448536269514722)
            sig_tot = math.sqrt(sig_z * sig_z + sigma_level * sigma_level)
            centre = lq[2]
            vals = [max(1.0, math.exp(centre + sig_tot * z)) for z in Z]
            for i in range(1, 5):
                vals[i] = max(vals[i], vals[i - 1])
            out[h] = [min(v, 1e9) for v in vals]
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    ch1 = first.get("channels")
    train = load_training(case.get("comparables", []))
    # The target's own platforms: find its entry among the comparables
    # (the final case carries the GTA VI category's own history there).
    target_platforms = None
    gname = (case.get("game") or {}).get("name", "")
    if gname:
        for c in case.get("comparables", []):
            if gname.lower() in c.get("name", "").lower():
                target_platforms = c.get("platforms")
                break
    fc = forecast(p1, ch1, train, target_platforms,
                  case.get("notes"), case.get("day_start"))
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = [round2(q) for q in fc[h]]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        fc = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(fc, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
