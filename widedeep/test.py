
def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / \
        filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled
    
import numpy as np
print(np.arange(6).reshape([2,3][::-1]))

print( np.arange(6).reshape([2,3][::-1]).T )

row_sort = np.arange(6).reshape([2,3][::-1]).T.ravel()



col_sort = np.arange(4).reshape(
                [2,2][::-1]).T.ravel()

print(row_sort)
print(col_sort)

#jacobi = np.mat(np.zeros((6, 4)))
jacobi = np.mat(np.arange(24).reshape([6,4]))
print(jacobi)
print(jacobi[row_sort, :][:, col_sort])