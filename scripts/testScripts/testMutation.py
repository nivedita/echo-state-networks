import numpy as np
def changeToNonZero(array):
    array[array == 0] = 1.0

if __name__ == "__main__":
    array = np.zeros((2,2))
    print(array)

    changeToNonZero(array)

    print(array)


    # Yay! Arrays are mutable !!!!!!!!!!