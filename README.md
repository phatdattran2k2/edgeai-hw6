# edgeai-hw6 — Team 陳發達 & 楊瑋竣

_I4210 AI 實務專題, Tatung University_
_Tran Phat Dat (陳發達) & Yang Wei-Chun (楊瑋竣)_

[![CI](https://github.com/phatdattran2k2/edgeai-hw6/actions/workflows/ci.yml/badge.svg)](https://github.com/phatdattran2k2/edgeai-hw6/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/phatdattran2k2/edgeai-hw6)](https://github.com/phatdattran2k2/edgeai-hw6/releases)

---

## Operations

### Quickstart

```bash
git clone https://github.com/phatdattran2k2/edgeai-hw6.git
cd edgeai-hw6
pdm install
pytest tests/ --ignore=tests/integration --cov=src --cov-fail-under=90
```

Expected: 43 passed, coverage ≥ 98%.

### How to deploy a new release

1. Ensure all CI checks pass on `main`.
2. Create an annotated tag:
   ```bash
   git tag -a v1.0.0 -m "Release notes here"
   git push --tags
   ```
3. Go to GitHub → Actions → the triggered `Deploy` workflow run.
4. Click **Review deployments** → **Approve and deploy** (production environment gate).
5. The workflow re-tags the image as `v1.0.0 / v1.0 / v1 / latest`, SSHes into the Jetson, sets nvpmodel, restarts the compose stack, and polls `/healthz` until 3 consecutive successes.

### How to roll back

**Symptoms that warrant a rollback:**

- `/healthz` returning non-`healthy` status for >60 s after deploy
- `docker ps` showing the inference container in a restart loop
- mAP drop >5% detected in post-deploy monitoring
- MQTT topic silent for >2 min after deploy

**Rollback command (run from the Jetson or via SSH):**

```bash
time bash deploy/rollback.sh
```

Expected wall time: <30 s.

**How to find the previous tag:**

```bash
cat /var/lib/edgeai-hw6/deployed.txt          # current tag
cat /var/lib/edgeai-hw6/deployed.txt.history  # previous tags
gh release list                                # all semver tags
```

**If rollback also fails (two broken tags):**

1. SSH into Jetson manually.
2. Identify the last known-good tag from `deployed.txt.history`.
3. Run: `IMAGE_TAG=<known-good-tag> docker compose -f deploy/docker-compose.yml up -d --force-recreate`
4. Verify `/healthz` manually: `curl http://localhost:8000/healthz`

**Team communication template:**

```
[INCIDENT] EdgeAI deploy issue — <timestamp>
Status: Rolling back from <bad-tag> to <previous-tag>
Impact: Inference service restarting, MQTT detections may be delayed ~30s
Action: bash deploy/rollback.sh (ETA <30s)
Update at: <timestamp + 5min>
```

---

## Architecture

### Pipeline Diagram

```mermaid
graph LR
    A[git push / PR] --> B[lint<br/>Ruff]
    B --> C[test<br/>pytest ≥90% cov]
    B --> D[security-scan<br/>Bandit + pip-audit]
    C --> E[build<br/>QEMU ARM64 → GHCR]
    D --> E
    E --> F[integration-test<br/>Jetson runner]
    G[git tag v*.*.*] --> H[deploy.yml<br/>manual approval]
    H --> I[re-tag image<br/>semver]
    I --> J[deploy.sh<br/>Jetson SSH]
    J --> K[nvpmodel switch]
    K --> L[compose up]
    L --> M[/healthz poll]
    M -->|fail| N[rollback.sh]
    M -->|pass| O[deployed ✓]
```

### Per-stage rationale

**lint (Ruff):** Catches style, import order, and common bugs before any code runs. Ruff is 10–100× faster than flake8, so it adds <15 s to the pipeline. Runs on every push and PR.

**test (pytest + coverage gate):** Runs all unit tests on a free x86 runner. The `--cov-fail-under=90` flag means a PR that deletes tests or adds untested code cannot merge. Ultralytics is lazy-imported so the suite runs without a GPU. Runs on every push and PR.

**security-scan (Bandit + pip-audit):** Bandit scans Python source for known-bad patterns (hardcoded secrets, subprocess misuse). pip-audit cross-references `requirements.txt` against the PyPA Advisory Database. Both must pass. Runs on every push and PR.

**build (Docker Buildx + QEMU):** Produces a multi-arch ARM64 image via QEMU emulation on the GitHub-hosted x86 runner and pushes it to GHCR tagged `sha-<short>`. Only runs after both test and security-scan pass. QEMU is slow (~8–12 min) but avoids needing a second Jetson as a build machine.

**integration-test (Jetson runner):** Pulls the just-built image, starts the inference container with `--runtime nvidia`, subscribes to the MQTT topic, and asserts at least one detection message arrives within 30 s. Only runs on pushes to `main` (not PRs) to avoid exhausting the single Jetson runner. Provides the "does it actually run on hardware" gate that x86 CI cannot.

**deploy (tag-triggered):** Triggered only on annotated `v*.*.*` tags. Pauses for a human reviewer on the `production` GitHub environment, then re-tags the SHA image with semver aliases, sets nvpmodel, restarts the compose stack, and runs a 3-consecutive-success healthcheck. Rolls back automatically if the healthcheck fails.

### What we explicitly chose NOT to do

- **Kubernetes / K3s at this stage:** The single-Jetson scope does not justify the operational overhead of a cluster orchestrator. The fleet-readiness section documents the path forward.
- **Build on the Jetson runner:** Would be 6× faster than QEMU but ties up the runner during the 8-min build, blocking integration tests. We accepted the QEMU penalty to keep the runner free.
- **TRT engine baked into the Docker image:** Engine files are hardware-specific (TRT version + driver). Baking the engine would mean a new image per Jetson SKU. We defer compilation to first container start and cache in a named volume (`lab12-models`).
- **Slack/Teams notification on deploy:** Out of scope for HW6; the GitHub deployment status and commit annotation serve as the audit trail.

---

## Optimization (INT8 vs FP16)

### Comparison Table

| Precision | Size (MB) | mAP@50 | Latency (ms/frame) | Notes                                     |
| --------- | --------- | ------ | ------------------ | ----------------------------------------- |
| FP16      | 7.0       | 0.3322 | ~6.2               | TensorRT half-precision, baseline         |
| INT8      | 4.0       | 0.3332 | ~6.6               | Calibrated with 500 representative frames |

Evaluation: `split=test`, `imgsz=320`, `batch=1`, Jetson Orin Nano 8 GB, JetPack 6 / TRT 10.7.

Delta: INT8 mAP@50 = **+0.0010 vs FP16** (INT8 is marginally _better_, well within the ≤2 pt threshold). Engine size: **4.3× smaller** (7 MB → 4 MB).

### Production recommendation

We would ship **INT8** to production. The calibration result shows no accuracy regression — INT8 is marginally better here because the activation histogram calibration effectively acts as a mild regularizer on this small test split. The 4× size reduction matters for fleet deployment: smaller images pull faster over the landfill site's LTE uplink, and the reduced memory footprint leaves headroom for future model additions (e.g., a secondary smoke-classification head).

The chosen production power mode is **15W** (nvpmodel ID=2, Original Orin Nano). At 15W the Jetson sustains the inference loop at ~57 FPS on the 320×320 input — well above the 10 FPS minimum for the target 30 s alert window. Dropping to 7W cuts FPS below 20 and increases per-frame latency enough to risk missing a short-duration smoke event.

### What didn't fit

**Distillation and pruning** were out of scope for HW6. Both require retraining from scratch with a teacher model or iterative pruning loops — roughly 20–30 additional GPU-hours on the construction-safety dataset. The INT8 result already meets the ≤2 pt accuracy gate, so pursuing distillation would be premature optimization for the current deployment target.

**Mixed-precision (FP16 layers + INT8 layers):** TensorRT supports per-layer precision overrides, but diagnosing which layers benefit requires profiling with `trtexec --profilingVerbosity=detailed`. We ran a single full-INT8 calibration pass; layer-wise tuning is a production-hardening step for a future sprint.

**Larger calibration sets:** We used 500 randomly sampled frames from the Lab 9 training split. A production calibration run would include frames from the actual landfill deployment environment (different lighting, dust, seasonal variation) — this is the highest-leverage improvement available if the INT8 mAP ever drops below threshold.

---

## Scaling to a Fleet

### How deploy.sh would change for N Jetsons

The current `deploy.sh` targets one Jetson because it runs _on_ the Jetson via a self-hosted runner. For N devices:

```bash
# Naive approach (dangerous — see below)
for JETSON_IP in 192.168.1.10 192.168.1.11 192.168.1.12; do
  ssh nvidia@$JETSON_IP "cd ~/edgeai-hw6 && bash deploy/deploy.sh $TAG"
done
```

The SSH key per device would be stored as separate GitHub secrets (`JETSON_SSH_KEY_01`, `JETSON_SSH_KEY_02`, etc.) and the workflow matrix would fan out one job per device.

### Why naive `for jetson in ...; do deploy; done` is dangerous

A sequential loop with no rollback coordination means:

- If device 2 fails its healthcheck and rolls back while devices 3–10 are still on the old tag, the fleet is split across two versions indefinitely.
- A bad image that crashes on device 1 will still be deployed to all remaining devices before anyone notices.
- No way to pause mid-fleet if a canary reveals a latency regression.

The safe pattern is a **rolling deploy with a canary**:

1. Deploy to 1 device (canary). Wait for healthcheck + 60 s of MQTT traffic.
2. If canary passes: deploy to the next N/4 devices in parallel.
3. After each wave: check aggregate MQTT message rate. If it drops >20%, halt and roll back the wave.
4. Tag the fleet state in a per-device state file (e.g. `/var/lib/edgeai-hw6/fleet_tag.json`).

**Per-device tag pinning:** Each Jetson should report its running tag to a central registry (SQLite + Flask or a simple MQTT retain message). The deploy workflow reads this registry before deploying and skips devices already on the target tag (idempotent deploys).

**Drift detection:** A nightly GitHub Actions scheduled job can SSH into each device, read `/var/lib/edgeai-hw6/deployed.txt`, and compare against the latest release tag. Any device more than 1 release behind triggers a Slack alert.

### Concrete tool recommendation: Ansible

For a fleet of 5–20 Jetsons at a landfill site, **Ansible** is the right tool:

- **Why it fits:** Ansible is agentless (SSH only), has native rolling-update support (`serial: 1` or `serial: 25%`), and its playbook YAML is readable by non-DevOps team members. The existing `deploy.sh` maps directly to an Ansible task with `command` or `shell`. Ansible's `delegate_to` and `when` conditionals handle the canary logic without writing custom orchestration code.
- **Dominant downside:** Ansible requires a control node with network access to all Jetsons simultaneously. At a landfill with intermittent LTE, a playbook can stall mid-fleet if a device goes offline during the run. The mitigation is `--forks 1 --timeout 30` and an explicit `ignore_errors: false` so the playbook stops rather than silently skipping unreachable devices.

---

## Reflections

### Tran Phat Dat (陳發達)

In HW6 I was primarily responsible for Part 0 (INT8 calibration on the Jetson, including collecting the 500-frame calibration dataset, running `calibrate_int8.py`, and measuring the FP16 vs INT8 mAP delta), Part A (refactoring `inference_node.py` to be testable, writing `mqtt_publisher.py`, `test_inference.py`, `test_mqtt.py`, `test_accuracy.py`, `test_healthcheck.py`, and demonstrating both the coverage-gate and accuracy-gate demo PRs), Part B (replacing the Lab 12 ci.yml with the five-stage dependency graph), and Part C (writing `test_jetson_e2e.py` and debugging the integration test end-to-end).

The most challenging technical problem was getting the Jetson self-hosted runner to successfully checkout the repository during CI. The runner would fail with `fatal: could not read Username for 'https://github.com': terminal prompts disabled` because the git credential helper was configured for the `jetson` user's interactive shell but not for the systemd service context under which the runner executes. The fix was to write the PAT to `/root/.git-credentials` and set `credential.helper=store` at the system level (`sudo git config --system`), then restart the runner service.

What I learned that I did not know before: the snapshot-testing pattern for the accuracy gate. Rather than running `model.val()` in CI (which requires the TRT engine and the Jetson), committing `accuracy_baseline.json` shifts the expensive measurement to a one-time Jetson run and makes the gate fast, free, and PR-blocking on the x86 runner. Any re-calibration that regresses mAP by >2 pts is caught the moment the updated JSON lands in a PR diff.

What I would do differently next time: write `test_healthcheck.py` before writing `healthcheck.py`, not after. I added `src/healthcheck.py` to fix a Dockerfile COPY error, then discovered the file was pulling coverage down to 69% because no tests existed yet. Writing the tests first (TDD) would have avoided that emergency coverage recovery sprint and the associated chain of fix commits that cluttered the PR history.

### Yang Wei-Chun (楊瑋竣)

_[Wei-Chun to fill in after completing Part D, E, F — must address: (1) which parts you worked on with specific file names, (2) the most challenging technical problem solved with symptom/fix detail, (3) one concrete transferable thing learned, (4) one concrete change you would make next time. Length: 150–250 words.]_

---

## Submission Evidence

Repo: <https://github.com/phatdattran2k2/edgeai-hw6>
Submission tag: `submission-final`
Released tag: `v1.0.0` _(to be created in Part D)_
GHCR image: `ghcr.io/phatdattran2k2/edgeai-hw6:v1.0.0` _(Part D)_

### Part 0 — INT8 Calibration (10 pts)

- Engine produced via real calibration →
  `best_int8.engine` in repo (size: 4 MB)
- INT8 mAP drop ≤ 2 pts →
  `calibration/accuracy_baseline.json` shows fp16=0.3322 int8=0.3332 Δ=−0.0010 (INT8 higher)
- Comparison table + production recommendation →
  README §"Optimization (INT8 vs FP16)" above

### Part A — Tests + Coverage + Accuracy Gates (15 pts)

- 6+ tests in test_inference → `tests/test_inference.py` (20 tests collected)
- 4+ tests in test_mqtt → `tests/test_mqtt.py` (8 tests collected)
- Coverage ≥90% gate + demo PR →
  green run: <https://github.com/phatdattran2k2/edgeai-hw6/actions/runs/26328982952>;
  demo coverage PR (red→green): <https://github.com/phatdattran2k2/edgeai-hw6/pull/2> (closed)
- htmlcov artifact uploaded → `evidence/htmlcov-artifact.png` _(screenshot to be added)_
- Accuracy gate + demo PR →
  demo accuracy PR (red→green): closed branch `demo/accuracy-gate-failing`
  fail run: <https://github.com/phatdattran2k2/edgeai-hw6/actions/runs/26329118098>;
  pass run: <https://github.com/phatdattran2k2/edgeai-hw6/actions/runs/26329314772>

### Part B — Five-Stage Workflow Graph (15 pts)

- 5 jobs with correct needs graph → `.github/workflows/ci.yml`
- bandit + pip-audit both run →
  green security-scan job: <https://github.com/phatdattran2k2/edgeai-hw6/actions/runs/26328982952>
- integration-test runs on jetson →
  `ci.yml`: `runs-on: [self-hosted, linux, arm64, jetson]`
- Workflow runs green end-to-end on main →
  <https://github.com/phatdattran2k2/edgeai-hw6/actions/runs/26328982952>

### Part C — Integration Test on Jetson (15 pts)

- Test pulls per-commit image →
  `tests/integration/test_jetson_e2e.py` (`test_image_is_per_commit_sha_tagged`)
- `--runtime nvidia` + model-cache volume →
  same file, fixture `inference_container`
- MQTT message within 30 s →
  same file, `test_inference_publishes_mqtt_within_window`
- Cleanup on failure → same file, fixture uses `yield` + explicit stop/rm
- Job runs green on main push →
  <https://github.com/phatdattran2k2/edgeai-hw6/actions/runs/26328982952>
  screenshot: `evidence/integration-test-passed.png`

### Part D — Tag-Triggered Deploy (20 pts)

- deploy.yml triggers on v*.*._ tags → `.github/workflows/deploy.yml` _(Part D — Wei-Chun)\*
- production environment with required reviewer →
  screenshot: `evidence/production-env-settings.png` _(Part D)_
- Re-tags as v1.0.0 / v1.0 / v1 / latest →
  green deploy run: _(Part D)_
- deploy.sh: pull → compose up → healthcheck → rollback-on-fail →
  `deploy/deploy.sh` _(Part D)_
- healthcheck.sh: 3 consecutive successes within 60 s →
  `deploy/healthcheck.sh` _(Part D)_
- deploy.sh sets nvpmodel →
  screenshot: `evidence/deploy-log-nvpmodel.png` _(Part D)_
- /healthz reports power_mode from live nvpmodel -q →
  `evidence/healthz-curl.png` _(Part D)_

### Part E — Rollback Under 30 s (5 pts)

- rollback.sh runs end-to-end <30 s →
  recording: `evidence/rollback-demo.cast` _(Part E — Wei-Chun)_
- State file maintains current + previous tag →
  recording shows `cat /var/lib/edgeai-hw6/deployed.txt` before + after _(Part E)_
- Rollback procedure (symptoms / command / recovery / comms) →
  README §"Operations" → "How to roll back" above

### Part F — Documentation & Fleet-Readiness (15 pts)

- All sections present in this README →
  §Architecture, §Optimization, §Scaling to a Fleet,
  §Operations, §Reflections, §Submission Evidence

### Code Quality (5 pts)

- Headers, ruff clean, secrets-free →
  confirmed by green lint + security-scan jobs above:
  <https://github.com/phatdattran2k2/edgeai-hw6/actions/runs/26328982952>
