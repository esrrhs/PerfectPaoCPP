# PerfectPaoCPP

[![CI](https://github.com/esrrhs/-PerfectPaoCPP/actions/workflows/ci.yml/badge.svg)](https://github.com/esrrhs/-PerfectPaoCPP/actions/workflows/ci.yml)
[![Language](https://img.shields.io/badge/language-C%2B%2B20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](https://github.com/esrrhs/-PerfectPaoCPP)

**PerfectPaoCPP** is an end-to-end self-play reinforcement learning system for **Paodekuai (跑得快, Run Fast)** — the Hunan 3-player 16-card shedding game — written natively in **pure C++20** and powered by **LibTorch**.

The framework is adapted from [PerfectDouCPP](https://github.com/esrrhs/PerfectDouCPP) (DouDizhu, PerfectDou lineage). The card engine, features, reward design and evaluation loop are re-built around the Paodekuai rule set; the PPO / perfect-information-distillation training core is preserved.

[**English**](#perfectpaocpp) | [**中文说明**](#perfectpaocpp-中文版)

---

## The Game (docs/RULES.md)

Paodekuai, Hunan 3-player variant (the rules used by Tencent's 欢乐跑得快):

- **Suit-less 48-card deck**: one deck minus the jokers, three 2s and one A. 3..K have four copies each, **A only three, 2 only one** (the largest solo).
- **3 symmetric players, 16 cards each, everyone for themselves**; first to shed all cards wins. No bidding, no kitty.
- **First lead**: one of the four 3s is designated at random; its holder leads and the first hand must contain a 3.
- **Must-beat (有牌必管)**: when following, pass is legal **only if nothing beats the table — bombs included** in the mandatory-beat set.
- **Patterns**: solo, pair, bare triple, triple+two loose kickers, straight (≥5), pair chain (≥2), triple chain (≥2), plane (triple chain + one solo/pair wing per trio), bomb.
- **Report-single (报单)**: a player holding one card is reported; leading a solo into a reported next player must top with the largest solo.
- **Zero-sum scoring**: the winner collects each loser's remaining-card count (doubled for a 16-card shutdown), normalized by 16.

## What changed vs the DouDizhu framework

| | PerfectDouCPP (DouDizhu) | PerfectPaoCPP (Paodekuai) |
| :--- | :--- | :--- |
| Seats | landlord / down / up (asymmetric teams) | 3 fully symmetric seats |
| Policy models | one Actor+Critic per seat (**3 models**) | **one shared model** with relative-seat features |
| Pass | always a legal action | **illegal when a beat exists** (must-beat) |
| Deck | 54 cards with jokers, 15 ranks | 48 suit-less cards, 13 ranks, A×3 / 2×1 |
| Move types | 15 (rocket, four+two, ...) | 11 (no jokers/rocket/four-with-two; pair chains ≥2, bare triples) |
| Terminal reward | ADP ±2·2^bombs camp scores | zero-sum winner-collects, /16 normalized |
| Oracle shaping | landlord vs peasants min-step diff | **per-seat** n(s) − min(n(others)) step diff |
| League | peasant team binds one snapshot | **one random seat** plays a snapshot **or the strengthened rule agent**, two learn |
| Baseline eval | external DouZero TCP server | **built-in PIMC + Max^N MCTS AI** (DouZero-style anchor) + PlanAgent |

## Architecture & Training Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│              1. Paodekuai Engine & Feature Engineering              │
│  48-card suit-less engine (must-beat / first-3 / report-single)     │
│  11×13 card matrices + relative-seat node features (1188-d)         │
│  5-step×3-move LSTM history (429-d) · per-seat Oracle min-steps     │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│            2. Multi-threaded Self-Play Rollout (CPU)                │
│  ONE shared policy for all seats → all waiting games batched        │
│  together in a single forward pass regardless of seat               │
│  Stratified league (recent pool + geometric archive), 1 hist seat   │
│  Per-seat Oracle difference dense rewards                           │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│        3. Zero-sum Settlement & GAE (γ=1.0, λ=0.95)                 │
│  Winner collects losers' remaining cards (shutdown doubled), /16    │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│      4. Perfect Information Distillation + PPO (GPU/MPS/CPU)        │
│  Omniscient critic (both opponents' hands + min-steps) → GAE        │
│  Clipped L_CLIP · decoupled Adam · masked softmax · KL early stop   │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│        5. Weight Sync · Full Checkpointing · Resume                 │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│        6. Built-in RuleAgent Evaluation (async, CSV, snapshots)     │
│  3 games per deck, policy occupies every seat · greedy argmax       │
│  tools/plot_eval.py curve rendering                                 │
└────────────────────────────────────────────────────────────────────┘
```

## Opponents & Evaluation (DouZero-style)

Three built-in opponents form an evaluation ladder (all operating on the same
engine, so rules always match exactly):

| Opponent | Algorithm | Cost/decision | Role |
| :--- | :--- | :--- | :--- |
| `RuleAgent` | cheap greedy heuristics | ~µs | PIMC fallback, sanity anchor |
| `PlanAgent` | Oracle min-steps planning with tactical guards (finish-now, bomb preservation, low-rank beats, holding the lone 2) | ~µs | fast league opponent + training anchor |
| `PimcAi` | **PIMC + Max^N UCB MCTS**: determinize the two hidden hands from the unseen pool, search each world with planning rollouts, aggregate root win-rates | ~7 ms (8x100) / ~50 ms (16x400) / ~180 ms (48x800) | the main evaluation anchor |

Training league mixes pure self-play, historical snapshots, and **rule-agent
seats** (`--rule-opponent-prob`, default 0.15) so every update contains fresh
games against a fixed non-drifting anchor. Evaluation mirrors the
PerfectDouCPP↔DouZero flow: during training the current policy is deep-copied
every N updates and plays 3 seat-rotated games per deck against two PIMC AIs
(`--pimc-every/--pimc-decks/--pimc-worlds/--pimc-iters`), appending to
`eval_vs_pimc.csv`, saving rolling snapshots, and a second lightweight loop
does the same against `PlanAgent` (`eval_vs_rule.csv`). Both CSVs render with
`tools/plot_eval.py`.

## Results

Two 2000-update CPU runs (≈1M games each, ~80 min on an 8-core laptop):

- `baselines/trained_v1/` — pure self-play + snapshot league, evaluated
  against the cheap greedy RuleAgent: **WP 0.561** (±0.009, 3000 games).
- `baselines/trained_v2/` — same budget with the rule-agent league
  (`--rule-opponent-prob 0.15`), evaluated against the PIMC ladder:

| Opponent (policy seat vs two opponents) | trained_v2 @2000 | trained_v3 @6000 | games (v3) | chance |
| :--- | :--- | :--- | :--- | :--- |
| PlanAgent (Oracle-planning rules) | 0.427 ±0.016 | **0.468** ±0.016 | 900 | 0.333 |
| PIMC light (8 worlds × 100 iters) | 0.363 ±0.028 | **0.383** ±0.028 | 300 | 0.333 |
| PIMC standard (16 × 400) | 0.310 ±0.027 | **0.345** ±0.019 | 600 | 0.333 |
| PIMC heavy (48 × 800) | 0.328 ±0.035 | **0.340** ±0.027 | 300 | 0.333 |

`trained_v3` = `trained_v2` continued to 6000 updates (~3M games) with the
late-stage curriculum (oracle shaping 0.2 → 0.05). Every tier improved: the
policy now clearly beats the light search tier, edges past the standard/heavy
tiers (0.34-0.35 over 300-600 games, training-time evals averaged ≈0.35 in the
second half), and dominates the planning rule agent. For reference trained_v1
(pure self-play league) scores 0.320 vs PIMC light and 0.427 vs PlanAgent.
Strength ladder of the built-in AIs themselves: PIMC standard beats PlanAgent
0.42 / PlanAgent beats greedy 0.49 (one seat each), greedy beats random 0.68.

Training throughput on the same 8-core M1 Pro: ~1.9 s per 512-game update vs
~8.8 s for PerfectDouCPP (2.3-4.6x faster per game at equal games/update:
single shared policy, smaller features, shorter games). The MPS backend works
(`--backend mps`) but gives no speedup at this model scale — CPU is
recommended on Apple Silicon.

```
$ ./build/perfectpao_eval --ckpt baselines/trained_v2 --decks 100 --opponent pimc --worlds 16 --iters 400
vs PIMC(16x400) | decks 100 games 300 | WP 0.3100 ADP +0.0427 (random-chance WP = 0.333)
```

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The 28-test Google Test suite covers pattern detection (including the
sub-chain plane edge cases), the must-beat generator, first-move-3 /
report-single constraints, zero-sum payoffs, the Oracle DP/DFS solver
against hand-checked values, feature/action encodings, GAE/PPO numerics,
NaN fallbacks, checkpoint serialization, rollout with oracle rewards, and
the end-to-end RuleAgent evaluation loop with snapshot pruning.

### Prerequisites
- C++20 compiler (g++ ≥ 11, clang ≥ 13, or MSVC 2022)
- CMake ≥ 3.16
- [LibTorch](https://pytorch.org/get-started/locally/) (a CPU `pip install torch`
  works: point `CMAKE_PREFIX_PATH` at `torch.utils.cmake_prefix_path`)
- Python 3.8+ (only for `tools/plot_eval.py`)

## Training Guide

```bash
# Standard CPU self-play training with periodic RuleAgent evaluation
./build/perfectpao_train --updates 100000 --games 768 --threads 16 --out ckpt \
    --eval-every 200 --eval-decks 300

# GPU (CUDA) / Apple Silicon (MPS)
./build/perfectpao_train --backend cuda --updates 50000 --games 768 --threads 16 --out ckpt
./build/perfectpao_train --backend mps  --updates 50000 --games 512 --threads 8  --out ckpt

# Quick sanity run
./build/perfectpao_train --updates 300 --games 128 --threads 8 --out ckpt_debug

# Evaluate a checkpoint against the PIMC AI (DouZero-style anchor)
./build/perfectpao_eval --ckpt ckpt --decks 100 --opponent pimc --worlds 16 --iters 400 --threads 8
# ...or against the planning rule agent
./build/perfectpao_eval --ckpt ckpt --decks 300 --opponent rule

# Render win-rate curves
python tools/plot_eval.py --csv ckpt/eval_vs_rule.csv ckpt/eval_vs_pimc.csv \
    --labels "vs PlanAgent" "vs PIMC" --out ckpt/eval_curves.png
```

Training log (win chance is 1/3 for three symmetric players; watch the
eval WP against RuleAgent instead — see the Results section above):

```
upd  100 | wall 4.0s learn 3.1s games 512 mb 2048 ep 2/2 | wins 170/171/171 margin 0.412 ... | train_n 24064
[eval u100] vs RuleAgent games 450 | WP 0.412 ADP +0.020
```

### Practical tips

1. **Oracle reward curriculum**: keep `--oracle-reward 1 --oracle-coef 0.2`
   early; lower to `0.05`/`0` late to focus on the terminal zero-sum score.
2. **Seat symmetry is free data**: every game contributes three on-policy
   transitions streams into the single shared policy.
3. **Resume**: rerun the original command with `--resume <DIR>` appended;
   deck streams derive deterministically from `seed + update * 7919`.

## Command Line Arguments Reference

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--backend <B>` | `cpu` | Hardware backend (`cuda` / `mps` / `cpu`) |
| `--out <DIR>` | `ckpt_long` | Output directory (weights, optimizer, logs) |
| `--updates N` | 100000 | Total PPO update steps |
| `--games N` | 768 | Games collected per update |
| `--threads N` | CPU cores | Concurrent self-play rollout threads |
| `--seed <S>` | 1 | Global PRNG seed |
| `--hidden H` | 256 | MLP hidden width |
| `--lstm-hidden H` | 128 | LSTM hidden size |
| `--epochs N` | 2 | PPO epochs per batch |
| `--mb N` | 2048 | PPO minibatch size |
| `--lr N` | 3e-4 | Adam learning rate |
| `--ent N` | 0.03 | Policy entropy coefficient |
| `--clip N` | 0.2 | PPO ratio clip ε |
| `--target-kl N` | 0.03 | KL early-stop threshold |
| `--gamma / --lambda` | 1.0 / 0.95 | GAE parameters |
| `--oracle-reward` / `--oracle-coef` | 1 / 0.2 | Per-seat Oracle difference reward |
| `--pool-size N` | 16 | League rolling pool size |
| `--archive-size N` | 16 | League geometric archive size |
| `--pool-every K` | 20 | League snapshot cadence |
| `--historical-prob` | 0.2 | Probability a game faces a snapshot |
| `--snapshot-every K` | 50 | Checkpoint cadence |
| `--resume DIR` | — | Resume from a checkpoint directory |
| `--eval-every N` | 200 | RuleAgent evaluation cadence (0 = off) |
| `--eval-decks D` | 200 | Decks per evaluation (×3 games) |
| `--eval-max-snapshots N` | 0 | Rolling snapshot limit (0 = unlimited) |

## Directory Structure

```
docs/RULES.md   The authoritative Paodekuai rule set this engine implements
src/pdk/        Suit-less card engine, 11 move types, must-beat generator,
                report-single / first-3 constraints, Oracle solver, features
src/torch_nn/   LibTorch Actor / Critic (shared policy), PPO learner, bridges
src/algo/       Batched single-policy rollout, GAE, RuleAgent evaluation
src/pdk/pimc_ai.*        PIMC + Max^N MCTS AI (determinization, UCB tree,
                         planning rollouts, cross-world aggregation)
src/pdk/rule_agent.h     greedy RuleAgent + strengthened PlanAgent
src/algo/eval_pimc.*     parallel DouZero-style evaluation vs the PIMC AI
src/            Training entry train.cpp, standalone eval eval_main.cpp
baselines/      trained_v1 / v2 / v3 policies + eval curves
tools/          plot_eval.py
tests/          Google Test suite (34 tests)
```

## Lineage

- **[PerfectDouCPP](https://github.com/esrrhs/PerfectDouCPP)** — the sibling DouDizhu
  project this framework was adapted from (PPO + perfect information
  distillation, league training, industrial numerics).
- **[DouZero](https://github.com/kwai/DouZero)** (ICML 2021) and
  **[PerfectDou](https://github.com/Vincent-Guan/PerfectDou)** (NeurIPS 2022) —
  the underlying training-paradigm heritage.

---

# PerfectPaoCPP 中文版

**PerfectPaoCPP** 是一套纯 C++20 + LibTorch 的**跑得快**（湖南三人 16 张玩法）端到端自对弈强化学习训练与评测系统。框架直接改编自斗地主项目 [PerfectDouCPP](https://github.com/esrrhs/PerfectDouCPP)，针对跑得快玩法重写了牌局引擎、特征编码、奖励设计与基线评测，保留了 PPO + 完美信息蒸馏的训练内核。

## 玩法（详见 docs/RULES.md）

- 无花色 48 张牌：去掉大小王、3 张 2、1 张 A；3~K 各四张、**A 仅三张、2 仅一张**（最大单张）。
- **三人对称、各自为战**，每人 16 张，先出完者胜；无叫牌、无底牌。
- 首出：随机指定场上 4 张 3 中的一张，持有者先出，首手必须含 3。
- **有牌必管**：跟牌时只有确实无牌可压才能过牌（**炸弹也算必须压的牌**）。
- 牌型：单张、对子、三张、三带二（两张散牌即可）、顺子（≥5）、连对（≥2）、三顺（≥2）、飞机带翅膀、炸弹。
- **报单**：下一家剩 1 张时，领出单张必须顶手中最大单张。
- 零和计分：赢家收两家剩余张数（16 张全关翻倍），除以 16 归一。

## 相对斗地主框架的关键改动

1. **对称单模型**：三个座位完全对称，一套 Actor/Critic + 相对座位特征（自己/上家/下家）通吃三席，每局产出三份在轨样本，数据效率 ×3；rollout 时所有等待决策的牌局**不分座位合并成一个大 batch** 前向。
2. **有牌必管约束**：合法动作空间由引擎强制生成——能压必压（含炸弹），pass 只在无牌可压时出现；这也是策略学习的核心难点。
3. **按座位 Oracle 奖励**：adv(s) = n(s) − min(n(其他两家))，出牌者获得 (lastAdv − curAdv)×coef 的步级密集塑形。
4. **零和终局结算**：/16 归一的收分制替代 ADP 炸弹翻倍。
5. **内置规则 Agent 评测**：没有跑得快版 DouZero，用带牌理的启发式 Agent 作锚点基线（随机策略 ≈ 0.24 WP，规则 Agent 单席对随机 ≈ 0.68 WP），每副牌模型轮换三个座位、贪心 argmax 决策、批量化对弈、CSV + 快照 + 自动淘汰。

## 快速开始

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure        # 28 项单测全绿

./build/perfectpao_train --updates 2000 --games 512 --threads 8 --out ckpt \
    --eval-every 100 --eval-decks 150             # CPU 训练（8 核约 70 分钟）
./build/perfectpao_eval --ckpt ckpt --decks 500   # 独立评测
python tools/plot_eval.py --csv ckpt/eval_vs_rule.csv --out ckpt/eval_curve.png
```

## 训练结果

`baselines/trained_v1/` 为 2000 轮 CPU 训练（约 100 万局自对弈）的模型，
对内置规则 Agent **3000 局评测胜率 0.561**（随机策略 ≈0.24、规则 Agent
单席对随机 ≈0.68，三人对称随机水位 1/3）：

```
$ ./build/perfectpao_eval --ckpt baselines/trained_v1 --decks 1000
vs RuleAgent | decks 1000 games 3000 | WP 0.5610 ADP +0.3599 (random-chance WP = 0.333)
```

## 参考文献

1. **DouZero**: Daochen Zha, et al. *Mastering DouDizhu with Self-Play Deep Reinforcement Learning*, ICML 2021.
2. **PerfectDou**: Yang Guan, et al. *PerfectDou: Dominating DouDizhu with Perfect Information Distillation*, NeurIPS 2022.
3. **PerfectDouCPP**: 完美信息蒸馏斗地主训练系统的 C++ 工程实现（本项目框架来源）。
