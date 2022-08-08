import heapq
import random
import collections
from functools import lru_cache
from LC.LCMetaPractice import ListNode, TreeNode


def word_search(board, word):
    found = [False]
    m, n, k = len(board), len(board[0]), len(word)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(start, x, y):
        if found[0]:
            return
        if start == k:
            found[0] = True
            return
        if x < 0 or x >= m or y < 0 or y >= n:
            return
        temp = board[x][y]
        board[x][y] = '#'
        if temp != word[start]:
            return
        for dx, dy in directions:
            dfs(start + 1, x + dx, y + dy)
        board[x][y] = temp

    for r in range(m):
        for c in range(n):
            if found[0]:
                return True
            dfs(0, r, c)
    return found[0]


def valid_palindrome_3(s, k):
    if s == s[::-1]:
        return True
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j - 1:
                if s[i] == s[j]:
                    dp[i][j] = 0
                else:
                    dp[i][j] = 1
            else:
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1
    return dp[0][n - 1] <= k


def partition_equal_subset_sum(nums):
    n, total_sum = len(nums), sum(nums)

    @lru_cache
    def recursive(total, i=0):
        if total == 0:
            return True
        if i >= n or total < 0:
            return False
        return recursive(total - nums[i], i + 1) or recursive(total, i + 1)

    return total_sum & 1 == 0 and recursive(total_sum // 2)


def validate_binary_tree_nodes(n, left_child, right_child):
    root, child_nodes = 0, set(left_child + right_child)
    for i in range(n):
        if i not in child_nodes:
            root = i

    queue, visited = collections.deque([root]), set()
    while queue:
        current = queue.popleft()
        if current in visited:
            return False
        visited.add(current)
        if left_child[current] != -1:
            queue.append(left_child[current])
        if right_child[current] != -1:
            queue.append(right_child[current])
    return len(visited) == n


def largest_bst_subtree(root):
    pass


def number_of_connected_components_in_an_undirected_graph(n, edges):
    graph = collections.defaultdict(list)
    result, visited = 0, set()

    def build_graph(u, v):
        graph[u].append(v)
        graph[v].append(u)

    def dfs(vertex):
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor)

    for edge in edges:
        build_graph(edge[0], edge[1])

    for vertex in range(n):
        if vertex not in visited:
            dfs(vertex)
            result += 1
    return result


def odd_even_linked_list(head):
    if not head or not head.next:
        return head
    odd, even, even_head = head, head.next, head.next

    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    odd.next = even_head
    return head


def flatten_binary_tree_to_linked_list(root):
    current = root
    while current:
        if current.left is not None:
            p = current.left
            while p.right is not None:
                p = p.right
            p.right = current.right
            current.right = current.left
            current.left = None
        current = current.right


def toeplitz_matrix(matrix):
    m, n = len(matrix), len(matrix[0])
    for i in range(m - 1):
        for j in range(n - 1):
            if matrix[i][j] != matrix[i - 1][j - 1]:
                return False
    return True
