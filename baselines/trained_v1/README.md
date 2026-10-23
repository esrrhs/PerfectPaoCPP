# Trained policy checkpoint (2000 PPO updates, 1.02M games self-play)
# Eval vs built-in RuleAgent over 3000 games: WP 0.561 ADP +0.360 (random-chance WP 0.333)
# Restore: ./build/perfectpao_train <original args> --resume baselines/trained_v1
# Evaluate: ./build/perfectpao_eval --ckpt baselines/trained_v1 --decks 1000
