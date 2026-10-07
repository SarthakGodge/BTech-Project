# Live-traffic demo

Replays held-out flows through the trained detector one at a time, exactly the way it would
see traffic from a live interface. Useful for a viva or a project demo: you can watch
`P(Pre-Attack)` climb as reconnaissance enters the rolling window, several flows before the
attack itself starts.

## Quick start (no server needed)

```bash
cd ~/BTech-Project
export IDS_DATA_ROOT=/data
python demo/replay.py --local --scenario escalation
```

Output, one line per flow:

```
  flow   true label     P(No Attack) P(Pre-Attack) P(Attack)   decision      action
    26   benign            0.927        0.059       0.015    No Attack    log       correct
    27   portscan          0.910        0.074       0.016    No Attack    log       wrong
    31   portscan          0.385        0.571       0.044    No Attack    log       wrong
    33   portscan          0.239        0.700       0.061    Pre-Attack   ALERT     correct
    35   portscan          0.070        0.819       0.110    Pre-Attack   ALERT     correct
```

and a summary with how many flows passed before the first correct detection.

## The real serving path

```bash
# terminal 1
export IDS_SAVED_MODEL=saved_model_weights_s1
export IDS_THRESHOLDS=thresholds_weights_s1.json
uvicorn serve.app:app --port 8000

# terminal 2
python demo/replay.py --api http://localhost:8000 --scenario escalation
```

Here the flows go over HTTP and the server keeps the 20-flow buffer, applies the saved scaler
and runs the model — the same code path a CICFlowMeter feed would use.

## Options

| Option | Meaning |
|---|---|
| `--scenario escalation` | benign, then reconnaissance, then attack (clearest for a demo) |
| `--scenario contiguous` | one real test block replayed in its original order |
| `--scenario portscan` \| `infiltration` \| `attack` \| `benign` | one traffic type only, after 25 benign flows |
| `--block N` | which block to replay with `--scenario contiguous` |
| `--n 120` | how many flows |
| `--rate 5` | flows per second, to make it watchable |
| `--tag weights_s1` | which trained model (local mode) |
| `--no-scale` | decide by raw argmax, ignoring the validation-tuned class scales |
| `--quiet` | print only alerts and the summary |
| `--list-blocks` | show which test blocks contain Pre-Attack or Attack traffic |

## How a decision is made

1. **Buffer.** The last 20 flow records are kept in order. Nothing is predicted until 20 have
   arrived, which is why the first 19 lines say `buffering`.
2. **Scale.** The 35 selected features are transformed with the *saved* scaler — signed
   `log(1+|x|)` then standardisation — using the statistics fitted on training data only. Fitting
   a new scaler here would shift every feature and ruin the model's calibration.
3. **Predict.** The (1, 20, 35) tensor goes through the CNN encoder and the LSTM aggregator,
   producing three probabilities that sum to 1.
4. **Apply class scales.** The probabilities are multiplied by the per-class scales chosen on the
   validation set (`thresholds_<tag>.json`) and the largest wins. This is why a class can lead on
   raw probability and still not be chosen — use `--no-scale` to see the unadjusted decision.
5. **Act.** `No Attack` → log, `Pre-Attack` → alert, `Attack` → block. In `serve/app.py` a
   decision below `CONFIDENCE_THRESHOLD` (default 0.60) is downgraded to a log entry.

## Honest limits

* Flows come from the held-out test split of CIC-IDS2017 / CSE-CIC-IDS2018, not from your
  network. Performance on a different network will be worse — our cross-dataset experiment
  showed transfer to an unseen capture fails badly.
* On real traffic, CICFlowMeter cannot emit a record until the flow ends or times out (120 s by
  default), so detection lags the traffic by up to that much. Model inference itself is a few
  milliseconds.
* Port-scan recall is high (over 0.99); infiltration recall is around 0.5, so roughly half of the
  subtler reconnaissance is missed.
* At production traffic volumes, even a small false-positive rate on benign windows produces many
  alerts per day. The confidence threshold and class scales are the knobs for that trade-off.
