import heapq
import random
import collections
from functools import lru_cache
from LC.LCMetaPractice import ListNode, TreeNode


def smallest_subtree_with_all_the_deepest_nodes(root):
    # same as lowest common ancestor of the deepest leaves
    if not root:
        return None
    parent, queue, last_level = {}, collections.deque([root]), []
    while queue:
        last_level = []
        for _ in range(len(queue)):
            current = queue.popleft()
            last_level.append(current)
            if current.left:
                queue.append(current.left)
                parent[current.left] = current
            if current.right:
                queue.append(current.right)
                parent[current.right] = current

    while len(last_level) > 1:
        parent_nodes = set()
        for node in last_level:
            parent_nodes.add(parent[node])
        last_level = list(parent_nodes)
    return last_level[0]


def intersection_of_three_sorted_arrays1(set1, set2, set3):
    set1, set2, set3 = set(set1), set(set2), set(set3)
    return list(set1.intersection(set2).intersection(set3))


def intersection_of_three_sorted_arrays2(arr1, arr2, arr3):
    result, i, j, k = [], 0, 0, 0
    while i != len(arr1) and j != len(arr2) and k != len(arr3):
        if arr1[i] == arr2[j] == arr3[k]:
            result.append(arr1[i])
            i, j, k = i + 1, j + 1, k + 1
        else:
            current = max(arr1[i], arr2[j], arr3[k])
            while i != len(arr1) and arr1[i] < current:
                i += 1
            while j != len(arr2) and arr2[j] < current:
                j += 1
            while k != len(arr3) and arr3[k] < current:
                k += 1
        return k


def stickers_to_spell_words():
    pass


def count_and_say(n):
    pass


def maximum_average_subtree(root):
    if not root or not root.children:
        return None
    result = [float('-inf'), root]

    def dfs(node):
        if not node.children:
            return [node.val, 1]
        current_value, current_count = node.val, 1
        for child in node.children:
            child_value, child_count = dfs(child)
            current_value += child_value
            current_count += child_count
        current_average = current_value / float(current_count)

        if current_average > result[0]:
            result[0], result[1] = current_average, node
        return [current_average, current_count]

    dfs(root)
    return result[0]


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
