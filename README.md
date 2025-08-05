# Learning and Robotics Challenge

This repository contains PyTorch implementations of two deep reinforcement learning algorithms (PPO, VPG) for various environments including CartPole, Pendulum, and Panda robotic tasks.

Success: mild --> still struggles to converge consistenly, but don't we all?

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- Panda-gym
- TensorBoard
- Other dependencies as specified in `setup.py`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lr_challenge.git
cd lr_challenge
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Available Scripts

The repository contains various training scripts organized by environment:

### CartPole Environment
```bash
python scripts/cartpole/cartpole_ppo_train.py
```

### Pendulum Environment
```bash
python scripts/pendulum/pendulum_ppo_train.py
```

### Panda Robot Environment
```bash
python scripts/panda/pnp_pposb3_train.py  # Pick and Place task
python scripts/panda/panda_slide_ppo_train.py  # Sliding task
```

## Monitoring Training with TensorBoard

Training progress can be monitored using TensorBoard. The training scripts automatically log metrics to the `runs` directory.

To start TensorBoard:

1. Activate your virtual environment if not already activated
2. Run TensorBoard:
```bash
tensorboard --logdir runs
```
3. Open your web browser and navigate to `http://localhost:6006`

TensorBoard will display training metrics including:
- Reward statistics
- Policy loss
- Value loss
- Other relevant metrics

## Project Structure

```
lr_challenge/
├── lr_challenge/
│   ├── algorithms/      # RL algorithm implementations
│   ├── learning/        # Core learning utilities
│   └── ...
├── scripts/            # Training scripts
│   ├── cartpole/
│   ├── pendulum/
│   └── panda/
└── runs/               # TensorBoard logs and saved models
```

## License

MIT License - See LICENSE file for details
