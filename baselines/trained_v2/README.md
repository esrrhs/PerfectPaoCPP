# Trained policy v2 (2000 PPO updates, ~1M games, rule-agent league)
#
# League: 65% pure self-play + 20% historical snapshots + 15% PlanAgent seats
# (vs v1 which had no rule-agent opponents).
#
# Evaluation (policy occupies each seat in turn; chance = 0.333):
#   vs PIMC 8x100  (light)    : WP 0.363 +-0.028  (300 games)
#   vs PIMC 16x400 (standard) : WP 0.310 +-0.027  (300 games)
#   vs PIMC 48x800 (heavy)    : WP 0.328 +-0.035  (180 games)
#   vs PlanAgent              : WP 0.427 +-0.016  (900 games)
#   v1 comparison: vs PIMC light 0.320, vs PlanAgent 0.427
#
# Evaluate: ./build/perfectpao_eval --ckpt baselines/trained_v2 --decks 100 \
#             --opponent pimc --worlds 16 --iters 400 --threads 8
# Resume:    ./build/perfectpao_train <original args> --resume baselines/trained_v2
