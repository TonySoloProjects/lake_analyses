# fun with numpy
import numpy as np

def f(x,y):
    print(f'x=\n{x}')
    print(f'y=\n{y}')
    return x+y

z = np.fromfunction(f,(4,3))

print(f'z=\n{z}')
