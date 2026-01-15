import random
from collections import defaultdict, Counter, deque
from itertools import zip_longest

from leetcode.utils import ListNode


def accounts_merge(accounts):
    # https://leetcode.com/problems/accounts-merge
    graph = defaultdict(list)  # adjacency list representation of a graph

    # If there are links in the provided accounts then a bidirectional graph is formed.
    # If not then, they are not added to the graph like "Mary" in the problem statement.
    for account in accounts:
        for i in range(2, len(account)):
            graph[account[i - 1]].append(account[i])
            graph[account[i]].append(account[i - 1])

    visited = set()

    def dfs(email, emails_list_per_person):
        visited.add(email)
        emails_list_per_person.append(email)
        for new_email in graph[email]:
            if new_email not in visited:
                dfs(new_email, emails_list_per_person)

    result = []

    for account in accounts:
        name = account[0]
        first_email = account[0]
        if first_email not in visited:
            all_emails = []
            dfs(first_email, all_emails)
            result.append([name] + sorted(all_emails))

    return result


def convert_bst_to_sorted_dll(root):
    # https://algo.monster/liteproblems/426
    # https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list
    if not root:
        return None

    list_head, previous_node = None, None

    def dfs(node):
        nonlocal list_head, previous_node
        if not node:
            return

        dfs(node.left)

        if previous_node:  # If we’ve already visited a node before (last), connect:
            previous_node.right = node
            node.left = previous_node
        else:  # If this is the very first node, it becomes head.
            list_head = node
        previous_node = node

        dfs(node.right)

    dfs(root)

    list_head.left = previous_node
    previous_node.right = list_head
    return list_head


def lowest_common_ancestor_of_a_binary_tree_iii(root, p, q):
    # https://zhenchaogan.gitbook.io/leetcode-solution/leetcode-1650-lowest-common-ancestor-of-a-binary-tree-iii
    # https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/description/
    # Same constraints as above
    # Base case: if root is None, or root is one of the targets
    if root is None or root == p or root == q:
        return root
    # We search both subtrees.
    # l can be either None (not found) or the node (p or q or an ancestor).
    # r behaves the same.
    left = lowest_common_ancestor_of_a_binary_tree_iii(root.left, p, q)
    right = lowest_common_ancestor_of_a_binary_tree_iii(root.right, p, q)
    # If both sides return something, it means:
    # p is in one branch
    # q is in the other branch
    # → so root is their lowest common ancestor.
    if left and right:
        return root
    return left or right


def toeplitz_matrix(matrix):
    # https://leetcode.com/problems/toeplitz-matrix
    m, n = len(matrix), len(matrix[0])

    for i in range(m - 1):
        for j in range(n - 1):
            if matrix[i][j] != matrix[i + 1][j + 1]:
                return False

    return True


def custom_sort_string(order, s):
    # https://leetcode.com/problems/custom-sort-string
    s_counts = Counter(s)

    result = []

    for char in order:
        if char in s_counts:
            result.append(char * s_counts[char])
            s_counts.pop(char)

    for char, count in s_counts.items():
        result.append(char * count)

    return "".join(result)


def insert_into_a_circular_linked_list(head, insert_val):
    # https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list
    # https://www.geeksforgeeks.org/sorted-insert-for-circular-linked-list/
    new_node = ListNode(insert_val)

    # Case 1: empty list
    if not head:
        new_node.next = new_node
        return new_node

    current = head
    while True:
        # Case 2.1: Normal insertion between two nodes
        if current.val <= insert_val <= current.next.val:
            break

        # Case 2.2: At the boundary (max -> min transition)
        if current.val > current.next.val:
            if insert_val >= current.val or insert_val <= current.next.val:
                break

        # Move forward and break after completing full cycle
        current = current.next
        if current == head:
            break

    # Insert new_node between current and current.next
    new_node.next = current.next
    current.next = new_node
    return head


def making_a_large_island(grid):
    # https://leetcode.com/problems/making-a-large-island
    """
    1. Paint each island different color and save their areas in a dictionary.
    2. Now go through each 0 cell and see if all island surrounding it can be added and see if that makes it the largest
    """
    n = len(grid)
    directions = [(-1, 0), (0, 1), (0, 1), (-1, 0)]
    area_dict = {}
    visited = [[0] * n for _ in range(n)]

    def calculate_area_of_each_island_using_bfs(r, c, island_id):
        queue = deque([(r, c)])
        grid[r][c] = island_id
        area = 1

        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 1:
                    grid[nx][ny] = island_id
                    area += 1
                    queue.append((x, y))

        return area

    island_id = 2
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                area_dict[island_id] = calculate_area_of_each_island_using_bfs(i, j, island_id)
                island_id += 1

    if not area_dict:
        return -1

    max_area = max(area_dict.values())

    for i in range(n):
        for j in range(n):
            if grid[i][j] == 0:
                visited = set()
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < j and grid[ni][nj] > 1:
                        visited.add(grid[ni][nj])
                new_area = 1  # flipped cell
                for island_id in visited:
                    new_area += area_dict[island_id]
                max_area = max(max_area, new_area)

    return max_area


def merge_string_alternatively(word1, word2):
    # https://leetcode.com/problems/merge-strings-alternately
    """
    result = []

    for char1, char2 in zip_longest(word1, word2):
        if char1:
            result.append(char1)
        if char2:
            result.append(char2)

    return "".join(result)
    """
    m, n = len(word1), len(word2)
    k = min(m, n)
    result = []

    for i in range(k):
        result.append(word1[k] + word2[k])

    return "".join(result) + word1[k:] + word2[k:]
