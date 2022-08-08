import heapq
import random
import collections
from functools import lru_cache
from LC.LCMetaPractice import ListNode, TreeNode


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
