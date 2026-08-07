# Reinforcement Learning — Bandits & Fitted Q-Iteration

Course projects (Centrale Lille / Master Data Science): multi-armed / cascading bandits and Fitted Q-Iteration with neural networks.

## Contents

| File | Topic |
|------|--------|
| `Bandits_TP.ipynb` | Cascading bandits, UCB / Thompson sampling |
| `MDP_Qlearning_TP.ipynb` | MDP & tabular Q-learning |
| `FittedQLearning_TP.ipynb` | Fitted Q-Iteration |
| `sekou_kaba.ipynb` | Deep RL / FQI notebook (graded work) |
| `SOLUTION.md` | Written notes |
| `regret1.png`, `wals.png`, `curve1.png` | Example regret / performance plots |

## Highlighted result

On the cascading-bandits experiment, CascadeUCB improved click rate by **~65%** vs a random ranking baseline (see notebook / plots).

## Stack

NumPy · PyTorch · Gymnasium (where used) · Matplotlib · Jupyter

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Bandits_TP.ipynb
```

## Note

This is **coursework**, kept public because it shows RL foundations (bandits → MDP → FQI) with clear experiments — complementary to production ML repos.
