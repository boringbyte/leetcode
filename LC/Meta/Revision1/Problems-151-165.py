import collections
from functools import lru_cache


def smallest_subtree_with_all_the_deepest_nodes(root):
    # same as lowest common ancestor of the deepest leaves
    # https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/discuss/479076/Python-Recursive-and-Iterative-Solution
    if not root:
        return None
    parent, queue, last_level = {root: None}, collections.deque([root]), []
    while queue:
        last_level = []
        for _ in range(len(queue)):
            child = queue.popleft()
            last_level.append(child)
            if child.left:
                queue.append(child.left)
                parent[child.left] = child
            if child.right:
                queue.append(child.right)
                parent[child.right] = child

    while len(last_level) > 1:
        parent_nodes = set()
        for node in last_level:
            parent_nodes.add(parent[node])
        last_level = list(parent_nodes)
    return last_level[0]


def intersection_of_three_sorted_arrays1(arr1, arr2, arr3):
    arr1, arr2, arr3 = set(arr1), set(arr2), set(arr3)
    return list(arr1.intersection(arr2).intersection(arr3))


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


def stickers_to_spell_words(stickers, target):
    # TODO: This might be wrong
    count, result, n, memo = collections.Counter(target), [float('inf')], len(target), collections.defaultdict(int)

    def dfs(sofar, i):
        if i == n:
            result[0] = sofar
        elif memo[target[i]] >= count[target[i]]:
            dfs(sofar, i + 1)
        elif sofar + 1 < result[0]:
            for sticker in stickers:
                if target[i] in sticker:
                    for s in sticker:
                        memo[s] += 1
                    dfs(sofar + 1, i + 1)
                    for s in sticker:
                        memo[s] -= 1
    dfs(0, 0)
    return result[0] if result[0] < float('inf') else -1


def count_and_say(n):
    # https://leetcode.com/problems/count-and-say/discuss/201832/It's-a-good-question-let-me-explain-it.
    def count(s):
        c, counter, result = s[0], 1, ''
        for char in s[1:]:
            if char == c:
                counter += 1
            else:
                result = result + str(counter) + c
                c, counter = char, 1
        result = result + str(counter) + c
        return result

    if n == 1:
        return '1'
    return count(count_and_say(n - 1))


def maximum_average_subtree(root):
    # https://leetcode.com/discuss/interview-question/349617
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
    return result[1]


def word_search1(board, word):
    found = [False]
    m, n, k = len(board), len(board[0]), len(word)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(idx, x, y):
        if found[0]:
            return
        if idx == k:
            found[0] = True
            return
        if x < 0 or x >= m or y < 0 or y >= n or board[x][y] != word[idx]:
            return
        board[x][y], temp = '#', board[x][y]
        for dx, dy in directions:
            dfs(idx + 1, x + dx, y + dy)
        board[x][y] = temp

    for r in range(m):
        for c in range(n):
            if found[0]:
                return True
            dfs(0, r, c)
    return found[0]


def word_search2(board, word):
    # This is a TLE solution
    m, n, k = len(board), len(board[0]), len(word)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(idx, x, y):
        if idx == k:
            return True
        if x < 0 or x >= m or y < 0 or y >= n or board[x][y] != word[idx]:
            return False
        board[x][y], temp = '#', board[x][y]
        if any(dfs(idx + 1, x + dx, y + dy) for dx, dy in directions):
            return True
        board[x][y] = temp
        return False

    for r in range(m):
        for c in range(n):
            if dfs(0, r, c):
                return True
    return False


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
    # https://leetcode.com/problems/partition-equal-subset-sum/discuss/1624939/C%2B%2BPython-5-Simple-Solutions-w-Explanation-or-Optimization-from-Brute-Force-to-DP-to-Bitmask
    n, total_sum = len(nums), sum(nums)

    @lru_cache
    def recursive(total, i=0):
        if total == 0:
            return True
        if i >= n or total < 0:
            return False
        return recursive(total - nums[i], i + 1) or recursive(total, i + 1)

    return total_sum % 2 == 0 and recursive(total_sum // 2)


def validate_binary_tree_nodes1(n, left_child, right_child):
    # https://leetcode.com/problems/validate-binary-tree-nodes/discuss/939381/Python%3A-clean-BFS-96-faster-TimeComplexity%3A-O(n)-Space-Complexity%3A-O(n)
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


def validate_binary_tree_nodes2(n, left_child, right_child):
    # comment of
    # https://leetcode.com/problems/validate-binary-tree-nodes/discuss/939381/Python%3A-clean-BFS-96-faster-TimeComplexity%3A-O(n)-Space-Complexity%3A-O(n)
    in_degree = [0] * n
    for l, r in zip(left_child, right_child):
        if l > -1:
            in_degree[l] += 1
        if r > -1:
            in_degree[r] += 1
        if in_degree[l] > 1 or in_degree[r] > 1:
            return False

    queue = collections.deque(node for node, degree in enumerate(in_degree) if degree == 0)

    if len(queue) > 1:
        return False

    while queue:
        current = queue.popleft()
        for child in left_child[current], right_child[current]:
            if child != -1:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    return sum(in_degree) == 0


class SubTreeInfo:
    """
    min, max: stores the minimum and the maximum value rooted under the current node
    `min`, `max` fields are relevant only if `isBST` flag is true
    size: stores size of the largest BST rooted under the current node
    isBST: true if the binary tree rooted under the current node is a BST
    """
    def __init__(self, min, max, size, is_bst):
        self.min = min
        self.max = max
        self.size = size
        self.is_bst = is_bst


def largest_bst_subtree(root):
    # https://www.techiedelight.com/find-size-largest-bst-in-binary-tree/
    if root is None:
        return SubTreeInfo(float('inf'), -float('inf'), 0, True)
    l, r = largest_bst_subtree(root.left), largest_bst_subtree(root.right)
    if l.is_bst and r.is_bst and (l.max < root.val < r.min):
        current = SubTreeInfo(min(root.val, l.min, r.min),
                              max(root.val, l.max, r.max),
                              l.size + 1 + r.size, True)
    else:
        current = SubTreeInfo(0, 0, max(l.size, r.size), False)
    return current.size


def number_of_connected_components_in_an_undirected_graph(n, edges):
    # https://www.goodtecher.com/leetcode-323-number-of-connected-components-in-an-undirected-graph/
    graph = collections.defaultdict(list)
    result, visited = 0, set()

    def build_graph(u, v):
        graph[u].append(v)
        graph[v].append(u)

    def dfs(i):
        visited.add(i)
        for neighbor in graph[i]:
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
        odd = odd.next  # similar to head = head.next
        even.next = odd.next
        even = even.next
    odd.next = even_head
    return head


def flatten_binary_tree_to_linked_list(root):
    # https://leetcode.com/problems/flatten-binary-tree-to-linked-list/discuss/1208004/Extremely-Intuitive-O(1)-Space-solution-with-Simple-explanation-Python
    current = root
    while current:
        if current.left:
            p = current.left
            while p.right:
                p = p.right
            p.right = current.right
            current.right = current.left
            current.left = None
        current = current.right


def toeplitz_matrix(matrix):
    # https://leetcode.com/problems/toeplitz-matrix/discuss/516366/Python-Follow-Up-1-with-Explanation-and-Diagrams
    m, n = len(matrix), len(matrix[0])
    for i in range(m - 1):
        for j in range(n - 1):
            if matrix[i][j] != matrix[i + 1][j + 1]:
                return False
    return True


def problem1():
    # https://leetcode.com/discuss/interview-question/2048705/Meta-or-Phone-Screenor-MLE-or-Remote
    arr = ['a', 'b', 'c', 'd', 'e']
    l, n = 0, len(arr)
    while l <= n - 3:
        print(arr[l: l + 3])
        l += 1


def problem2():
    # https://leetcode.com/discuss/interview-question/2048705/Meta-or-Phone-Screenor-MLE-or-Remote
    arr = ['a', 'b', 'c', 'd', 'e']
    l, n = 0, len(arr)
    while l <= n - 4:
        temp1 = arr[l: l + 4]
        for i in range(1, 4):
            temp = temp1[:]
            temp.pop(i)
            print(temp)
        l += 1


def valid_word_abbreviation(word, abbr):
    # https://massivealgorithms.blogspot.com/2016/10/leetcode-408-valid-word-abbreviation.html
    size, count, loc = len(word), 0, 0
    for char in abbr:
        if char.isdigit():
            if char == '0' and count == 0:
                return False
            count = count * 10 + int(char)
        else:
            loc = loc + count
            count = 0
            if loc >= size or word[loc] != char:
                return False
            loc += 1
    return loc + count == size
