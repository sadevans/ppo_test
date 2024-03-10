# ppo_test
PPO implementation for InvertedPendulumEnv

# The best state
Наиболее похожее на ТЗ состояние:

<video controls src="videos/best.mp4" title="Title"></video>

Такого состояния удалось достичь при функции наград вида:
```python
    theta = np.mod(ob[1], 2*np.pi) # [0; 2*pi]
    theta = (theta - 2*np.pi) if theta > np.pi else theta # [-pi; pi]
    reward = -(theta**2 + 0.1*ob[3]**2 + 2*ob[0]**2)
    if abs(theta) < 0.1:
        reward += 0.1 * np.cos(theta) - 0.1*ob[3]**2
```

# Evaluation
Fast evaluation in google colab - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sadevans/ppo_test/blob/main/fast_eval.ipynb)

