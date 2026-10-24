# Trained policy v3 (6000 PPO updates, ~3M games)
#
# trained_v2 continued for 4000 more updates with the late-stage curriculum
# (oracle shaping coefficient 0.2 -> 0.05) under the same rule-agent league.
#
# Evaluation (policy occupies each seat in turn; chance = 0.333):
#   vs PIMC 8x100  (light)    : WP 0.383 +-0.028  (300 games)
#   vs PIMC 16x400 (standard) : WP 0.345 +-0.019  (600 games)
#   vs PIMC 48x800 (heavy)    : WP 0.340 +-0.027  (300 games)
#   vs PlanAgent              : WP 0.468 +-0.016  (900 games)
#   vs v2@2000:               light 0.363 / standard 0.310 / heavy 0.328 / plan 0.427
#
# Evaluate: ./build/perfectpao_eval --ckpt baselines/trained_v3 --decks 200 \
#             --opponent pimc --worlds 16 --iters 400 --threads 8
# Resume:    ./build/perfectpao_train <original args> --resume baselines/trained_v3
