# %%
import numpy as np

# %%
data = np.array([[1, 2], [3, 4], [5, 6]])
print(data)

# %%
data.ndim

# %%
data.shape

# %%
data.size

# %%
data.dtype

# %%
data.nbytes

# %%
np.array([1, 2, 3, 4], dtype=np.int)

# %%
np.array([1, 2, 3, 4], dtype=np.float)

# %%
np.array([1, 2, 3, 4], dtype=np.complex)

# %%
data = np.array([1, 2, 3], dtype=np.float)
data.dtype
# %%
data = np.array(data, dtype=np.int)
data.dtype
# %%
data

# %%
d1 = np.array([1, 2, 3], dtype=float)
d2 = np.array([1, 2, 3], dtype=complex)
d1 + d2

# %%
(d1 + d2).dtype

# %%
np.sqrt(np.array([-1, 0, 1]))

# %%
np.sqrt(np.array([-1, 0, 1], dtype=complex))

# %%
np.zeros((2, 3))

# %%
np.ones((3, 4))

# %%
np.arange(0.0, 10, 1)

# %%
np.linspace(0, 10, 11)

# %%
np.logspace(0, 2, 4)

# %%
x1 = 4.4 * np.ones(10)
x1
# %%
x2 = np.full(10, 4.4)
x2

# %%
x = np.array([-1, 0, 1])
y = np.array([-2, 0, 2])
X, Y = np.meshgrid(x, y)
X
# %%
Y

# %%
Z = (X + Y) ** 2
Z
# %%
np.empty(3, dtype=np.float)

# %%


def f(x):
    y = np.ones_like(x)
    return y


f1 = f([[1, 2],
        [3, 4],
        [5, 6]])
f1

# %%
np.identity(4)

# %%
np.eye(3, k=1)

# %%
np.eye(3, k=-1)

# %%
np.diag(np.arange(0, 20, 5))

# %%
