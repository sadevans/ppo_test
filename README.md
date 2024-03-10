# ppo_test
PPO implementation for InvertedPendulumEnv

# Задание

С помощью алгоритма Proximal Policy Optimization для системы маятник на тележке решить задачу подъема маятника из нижнего положения в верхнее с последующей стабилизацией.

Ожидаем имплементацию всего пайплайна PPO с использованием только пакета pytorch. При этом можно ориентироваться на готовые решения.

# Решение
Для реализации алгоритма PPO использовалась оригинальная [статья](https://arxiv.org/pdf/1707.06347.pdf) и [книга с теорией](https://github.com/FortsAndMills/RL-Theory-book/blob/main/RL_Theory_Book.pdf)

Также ориентировалась на [курс ШАДА](https://github.com/yandexdataschool/Practical_RL/tree/master).

Посмотрела также реализацию в Pytorch, но там странно считаются advantages - не стала на нее ориентироваться.

## Достижение наилучшего результата
В процессе решения менялась функция наград в классе InvertedPendulumEnv.

Были протестированы различные функции. Их вид и результат тестирование представлены ниже.

### Вариант №1

```python
theta = np.mod(ob[1], 2*np.pi) # [0; 2*pi]
theta = (theta - 2*np.pi) if theta > np.pi else theta # [-pi; pi]
if abs(ob[0]) > 0.8:
  out_of_bound = 1
else:
  out_of_bound = 0
x = abs(ob[0]) # ob[0] in [-1.1, 1.1]
x_change_reward = -x ** 2
reward += 0.3 * x_change_reward
if abs(theta) < 0.1:
  reward += 10
else:
  reward = 2.5 * np.cos(theta) - 0.01*(ob[3])**2 - 0.1*a[0]**2 - 10*out_of_bound
```

https://github.com/sadevans/ppo_test/assets/82286355/8aed1d2d-e8c6-42a0-8979-ceee6f1fae61

### Вариант №2
```python
reward = 0
theta = np.mod(ob[1], 2*np.pi) # [0; 2*pi]
theta = (theta - 2*np.pi) if theta > np.pi else theta # [-pi; pi]      
if abs(ob[0]) > 0.9:
  out_of_bound = 1
else:
  out_of_bound = 0

reward = np.cos(theta) - 0.001*(ob[3])**2 - 0.01*a[0]**2 - 1*out_of_bound
```

https://github.com/sadevans/ppo_test/assets/82286355/f632cb42-92ad-47c3-9223-02addfdbe6b7

### Вариант №3
```python
reward = 0
theta = np.mod(ob[1], 2*np.pi) # [0; 2*pi]
theta = (theta - 2*np.pi) if theta > np.pi else theta # [-pi; pi]

if abs(theta) > 0.9:
reward -= 1
coef_velocity = 1
else:
print(theta, ob[2], ob[3])
reward += np.exp(1 - abs(theta))
coef_velocity = -(abs(theta) - 1)

if abs(theta) < 0.2:
reward += 100/(0.3*ob[0]**2)

if abs(ob[0]) > 0.8:
reward -= 10

swing_up =  1 - abs(theta) / np.pi
reward += swing_up + coef_velocity*abs(ob[3])**2 - 0.4*abs(a[0])/3 + ob[0] * 0.2*abs(ob[2]) # более плавно но перелетает все равно
```


https://github.com/sadevans/ppo_test/assets/82286355/2f55b9ad-cbe3-41af-acd0-755e3e6a9608


# Наилучший результат
Наиболее похожее на ТЗ состояние:

https://github.com/sadevans/ppo_test/assets/82286355/8b88d8b5-9485-4e6a-b3b1-22f4b7765c3c

Маятник смог подняться в вертикальное состяние и немного его удержать, проехавшись. 

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

