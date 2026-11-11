# Privacy versus Integrity in Federated Keyboard Learning

**A governance audit and threat map, with runnable offline probes.**
Authors: Aaron Sam and Joey Manivong (M.S. in Artificial Intelligence, San José State University).
Course project, Track A: Governance Audit and Threat Mapping.

## What this is
Cross-device federated learning (FL) is how phone keyboards learn next-word prediction without uploading what you type. It is one of the best-documented privacy-preserving ML deployments, which makes it a good audit target. This repository audits that architecture class and tests the central claim of the project:

> **Secure aggregation (a privacy control) and server-side update inspection (an integrity control) are in structural tension.** Hiding individual updates from the server protects users, and also removes the server's ability to spot a malicious update.

This repository holds the **evidence** for the audit (`src/`, `tests/`, `results/`). Everything is offline and synthetic. No real device, user data, or live system is touched, and the findings are about an architecture class and the published literature, not about any company's product.

## Quick start
```bash
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
make test        # 29 unit tests, ~10 s
make smoke       # end-to-end check of all three probes, ~20 s
make quick       # 2 seeds, a few minutes on a laptop CPU
make full        # 3 seeds, the scale behind results/preliminary (about an hour on 2 cores)
# also: python3 scripts/run_dose.py, scripts/summarize.py, scripts/build_register.py
```
Outputs (CSVs and PNG figures) go to `results/`. Fixed seeds make runs repeatable on the same library versions.

## Layout
| Path | Contents |
|---|---|
| `src/pvi/` | Simulator: synthetic non-IID data, LSTM model, FedAvg with clipping, DP noise and secure-aggregation visibility, attacks, defenses, metrics, accounting |
| `src/pvi/probes/` | The three probes |
| `tests/` | Unit tests, including the rho-to-epsilon conversion check |
| `scripts/` | `run_all.py`, `make_figures.py`, `build_register.py` |
| `results/preliminary/` | Committed CSVs and figures from the 3-seed run |

## The three probes
1. **Backdoor sweep** (`probes/backdoor.py`): trigger and edge-case backdoors against open, clipped, DP, secure-aggregation and inspecting defenses.
2. **Canary exposure** (`probes/canary.py`): memorization of a planted random sequence versus DP noise.
3. **Secagg versus inspection** (`probes/tension.py`): detection AUC when updates are visible, against membership-inference AUC from the same visibility.

## Honest scope
* **Surrogate system.** A ~30k-parameter LSTM on synthetic Markov-chain text with 30 clients. Effect sizes do not transfer to production. The direction of each comparison is the claim, and with three seeds the confidence intervals are wide.
* **Secure aggregation is modeled by visibility,** not implemented cryptographically.
* **DP accounting is conservative:** rho = T / (2 z^2), no subsampling credit. Real deployments use DP-FTRL and tighter accounting [9], [10].
* **Not simulated:** the malicious-server inconsistent-model attack [26] and gradient inversion. They are scored from the literature.
* **Flower.** The abstract names Flower. The probes use the in-repo simulator because it can model secagg visibility and norm-matched attackers directly. `src/pvi/flower_adapter.py` plugs the same model and training step into Flower for the honest-client path; it is not exercised by the tests.
* **Data.** Synthetic by default. `data.load_leaf_shakespeare` loads the LEAF Shakespeare partition [30] if you download it yourself.

## Reproducing the headline numbers
`make full` regenerates `results/full/`; the committed `results/preliminary/` is the run reported in the audit.

## License
MIT. See `LICENSE`.
