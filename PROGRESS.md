# Ayush & Era: ARENA progress

Every section's exercise folder has an `ayush/` and an `era/` folder, each holding that person's copy of the
exercise notebook:

```
chapter2_rl/exercises/part1_intro_to_rl/
├── ayush/2.1_Intro_to_RL.ipynb     ← Ayush works here
├── era/2.1_Intro_to_RL.ipynb       ← Era works here
├── 2.1_Intro_to_RL_exercises.ipynb   (upstream original, don't edit)
├── 2.1_Intro_to_RL_solutions.ipynb
├── tests.py  utils.py  solutions.py  (shared)
```

## Rules

- **Only edit files in your own folder.** Since neither of us touches the other's files, merges never conflict.
- **Leave the upstream files alone** (`*_exercises.ipynb`, `tests.py`, `utils.py`, `instructions/`). This keeps
  `git fetch upstream && git merge upstream/main` clean.
- **In the progress table below, tick your own column only.**
- Clear notebook outputs before committing, to keep diffs readable.

```bash
git pull
git add -A '*/ayush/*'      # or '*/era/*'
git commit -m "ayush: 2.1 policy iteration"
git push
```

## Progress

☐ not started · ◐ in progress · ☑ done

| Section | Ayush | Era |
|---|:-:|:-:|
| **Chapter 0: Fundamentals** | | |
| 0.0 Prerequisites | ☐ | ☐ |
| 0.1 Ray Tracing | ☐ | ☐ |
| 0.2 CNNs & ResNets | ☐ | ☐ |
| 0.3 Optimization | ☐ | ☐ |
| 0.4 Backprop | ☐ | ☐ |
| 0.5.1 VAEs | ☐ | ☐ |
| 0.5.2 GANs | ☐ | ☐ |
| **Chapter 1: Transformer Interpretability** | | |
| 1.1 Transformer from Scratch | ☐ | ☐ |
| 1.2 Intro to Mech Interp | ☐ | ☐ |
| 1.3.1 Linear Probes | ☐ | ☐ |
| 1.3.2 Function Vectors & Model Steering | ☐ | ☐ |
| 1.3.3 Interpretability with SAEs | ☐ | ☐ |
| 1.3.4 Activation Oracles | ☐ | ☐ |
| 1.4.1 Indirect Object Identification | ☐ | ☐ |
| 1.4.2 SAE Circuits | ☐ | ☐ |
| 1.5.1 Balanced Bracket Classifier | ☐ | ☐ |
| 1.5.2 Grokking & Modular Arithmetic | ☐ | ☐ |
| 1.5.3 OthelloGPT | ☐ | ☐ |
| 1.5.4 Toy Models of Superposition & SAEs | ☐ | ☐ |
| **Chapter 2: Reinforcement Learning** | | |
| 2.1 Intro to RL | ◐ | ☐ |
| 2.2.1 DQN | ☐ | ☐ |
| 2.2.2 VPG | ☐ | ☐ |
| 2.3 PPO | ☐ | ☐ |
| 2.4.1 RLHF | ☐ | ☐ |
| 2.4.2 RLVR | ☐ | ☐ |
| 2.5 MCTS & AlphaZero | ☐ | ☐ |
| **Chapter 3: LLM Evals** | | |
| 3.1 Intro to Evals | ☐ | ☐ |
| 3.2 Dataset Generation | ☐ | ☐ |
| 3.3 Running Evals with Inspect | ☐ | ☐ |
| 3.4 LLM Agents | ☐ | ☐ |
| 3.5 AI Control | ☐ | ☐ |
| **Chapter 4: Alignment Science** | | |
| 4.1 Emergent Misalignment | ☐ | ☐ |
| 4.2 Science of Misalignment | ☐ | ☐ |
| 4.3 Interpreting Reasoning Models | ☐ | ☐ |
| 4.4 LLM Psychology & Persona Vectors | ☐ | ☐ |
| 4.5 Investigator Agents | ☐ | ☐ |

2.4.3 RLAIF has no folders because upstream it's still a placeholder with no exercises.
