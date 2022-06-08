import os
import os


if __name__ == '__main__':
    dir = os.listdir('tf')

    json = [f for f in dir if os.path.isfile(f'tf/{f}')]
    print(json)