import heapq
import random
import collections
from functools import lru_cache
from LC.LCMetaPractice import ListNode, TreeNode


def toeplitz_matrix(matrix):
    m, n = len(matrix), len(matrix[0])
    for i in range(m - 1):
        for j in range(n - 1):
            if matrix[i][j] != matrix[i - 1][j - 1]:
                return False
    return True
