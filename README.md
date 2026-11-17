# GTA VI Twitch CCU launch-day forecaster v3

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/elpablo61/gta6-twitch-ccu-forecast-03/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/elpablo61/gta6-twitch-ccu-forecast-03/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Forecasts the 24 UTC hours of Grand Theft Auto VI's release day on Twitch,
hour by hour, as five quantiles an hour — from everything recorded up to the
end of hour 1. Built by an AI agent for
[make.fun's "Forecast GTA VI's Twitch CCU" bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
($250 in USDC).

## Model

A hierarchical launch-day curve, fit at run time from the case's own
comparables (past big launches' hourly Twitch curves). One entrypoint,
`main.py` (Python 3.13, standard library only, no network):

- **Shared shape, size-dependent level.** The log-ratio curve
  `log(v(h)/v(1))` is fit as a function of launch size `log(v(1))` by two
  blended views: a local-linear fit and a kernel-weighted median ratio.
- **Local spread.** Quantile widths come from the local dispersion of
  leave-one-out residuals, so small launches get far wider quantiles
  than big ones.
- **Segment levels.** Console-only targets downweight PC-platform
  training launches (more for small console targets, whose centre would
  otherwise be dragged up by explosive small PC launches).
- **Misalignment guard.** For a small target the upper quantiles are
  floored at the small-launch pool's own quantiles — a launch whose UTC
  hour 1 was pre-unlock quiet can explode 1000x later.
- **Unlock alignment.** When the target's unlock text names a single
  global UTC unlock hour later in the day, the forecast switches to
  surge-aligned time, predicting the surge *level* (far more predictable
  than the surge ratio) and the post-surge shape.

## Validation (leave-one-out on the kit's 37 comparable launches)

| set | v3 | v2 | persistence baseline |
|-----|----|----|----------------------|
| all | **0.3270** | 0.3786 | 0.9296 |
| big (h1 ≥ 20k) | 0.2500 | 0.2485 | 0.3214 |
| family (console midnight-local, h1 ≥ 20k) | 0.1721 | 0.2752 | 0.3006 |
| console-only | **0.1701** | 0.2085 | 0.3225 |

Scores are mean pinball loss in log space (lower is better).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Referral

Built with make.fun — https://make.fun/r/vky0ybyt
