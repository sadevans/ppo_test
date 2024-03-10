# ppo_test
PPO implementation for InvertedPendulumEnv

# Задание

С помощью алгоритма Proximal Policy Optimization для системы маятник на тележке решить задачу подъема маятника из нижнего положения в верхнее с последующей стабилизацией.

Ожидаем имплементацию всего пайплайна PPO с использованием только пакета pytorch. При этом можно ориентироваться на готовые решения.

# Решение
Для реализации алгоритма PPO использовалась оригинальная [статья](https://arxiv.org/pdf/1707.06347.pdf) и [книга с теорией](https://github.com/FortsAndMills/RL-Theory-book/blob/main/RL_Theory_Book.pdf)

Также ориентировалась на [курс ШАДА](https://github.com/yandexdataschool/Practical_RL/tree/master)
Посмотрела также реализацию в Pytorch, но там странно считаются advantages - не стала на нее ориентироваться.


# Наилучший результат
Наиболее похожее на ТЗ состояние:

https://github.com/sadevans/ppo_test/assets/82286355/8b88d8b5-9485-4e6a-b3b1-22f4b7765c3c

Такого состояния удалось достичь при функции наград вида:
```python
    theta = np.mod(ob[1], 2*np.pi) # [0; 2*pi]
    theta = (theta - 2*np.pi) if theta > np.pi else theta # [-pi; pi]
    reward = -(theta**2 + 0.1*ob[3]**2 + 2*ob[0]**2)
    if abs(theta) < 0.1:
        reward += 0.1 * np.cos(theta) - 0.1*ob[3]**2
```

# Evaluation
Быстро протестировать в google colab - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sadevans/ppo_test/blob/main/fast_eval.ipynb)

