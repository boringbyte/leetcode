import heapq
import collections
from LC.LCMetaPractice import ListNode, TreeNode


def buildings_with_an_ocean_view1(heights):
    result, n = [], len(heights)
    for i in range(n - 1, -1, -1):
        if not result or heights[i] > heights[result[-1]]:
            result.append(i)
    return result[::-1]


def buildings_with_an_ocean_view2(heights):
    result = []
    for i, height in enumerate(heights):
        while result and heights[result[-1]] <= height:
            result.pop()
        result.append(i)
    return result


def closest_binary_search_tree_value(root, target):
    gap, result = float('inf'), float('inf')
    while root:
        if abs(root.val - target) < gap:
            gap = abs(root.val - target)
            result = root.val
        if root.val == target:
            break
        elif target < root.val:
            root = root.left
        else:
            root = root.right
    return result


def remove_linked_list_elements(head, target):
    dummy = ListNode(-1)
    dummy.next, prev = head, dummy
    while head:
        if head.val != target:
            prev = head
        else:
            prev.next = head.next
        head = head.next
    return dummy.next


def string_to_integer():
    pass


def build_graph_from_binary_tree(root):
    graph = collections.defaultdict(list)

    def dfs(child, parent):
        if parent:
            graph[child].append(parent)
        if child.left:
            graph[child].append(child.left)
            dfs(child.left, child)
        if child.right:
            graph[child].append(child.right)
            dfs(child.right, child)

    dfs(root, None)
    return graph


def all_nodes_distance_k_in_binary_tree1(root, target, k):
    visited, result = {target}, []
    graph = build_graph_from_binary_tree(root)

    def dfs(node, distance):
        if distance == 0:
            result.append(node.val)
        else:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, distance - 1)

    dfs(target, k)
    return result


def all_nodes_distance_k_in_binary_tree2(root, target, k):
    visited, result, queue = set(), [], collections.deque([(target, 0)])
    graph = build_graph_from_binary_tree(root)
    while queue:
        current, distance = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if k == distance:
            result.append(current.val)
        elif distance < k:
            for child in graph[current]:
                queue.append((child, distance + 1))
    return result


def number_of_islands(grid):
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    m, n, result = len(grid), len(grid[0]), 0

    def dfs(x, y):
        if 0 <= x < m and 0 <= y < n and grid[x][y] == '1':
            grid[x][y] == '0'
            for dx, dy in directions:
                dfs(x + dx, y + dy)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
                result += 1
    return result


def regular_expression_matching(s, p):
    pass


def is_valid_number(s):
    s = s.strip()
    met_dot = met_e = met_digit = False
    for i, char in enumerate(s):
        if char in '+-':
            if i > 0 and s[i - 1].lower() != 'e':
                return False
        elif char == '.':
            if met_dot or met_e:
                return False
            met_dot = True
        elif char.lower() == 'e':
            if met_e or not met_digit:
                return False
            met_e, met_digit = True, False
        elif char in '0123456789':
            met_digit = True
        else:
            return False
    return met_digit


def longest_substring_without_repeating_characters(s):
    n, result, seen, left = len(s), 1, {}, 0
    if n == 0:
        return 0
    for right, char in enumerate(s):
        if char in seen:
            left = max(left, seen[char] + 1)
        result = max(result, right - left + 1)
        seen[char] = right
    return result


def length_of_longest_substring_without_repeating_characters(s):
    n, result, seen, left = len(s), 1, {}, 0
    if n == 0:
        return 0
    for right, char in enumerate(s):
        if char in seen:
            left = max(left, seen[char] + 1)
        result = max(result, right - left + 1)
        seen[char] = right
    return result


def top_k_frequent_elements(nums, k):
    counts, freq_dict, result = collections.Counter(nums), collections.defaultdict(list), []

    for num, freq in counts.items():
        freq_dict[freq].append(num)

    for freq in reversed(range(len(nums) + 1)):
        result.extend(freq_dict[freq])
        if len(result) >= k:
            return result[:k]
    return result[:k]


def search_in_rotated_sorted_array():
    pass


def largest_island(grid):
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    m, n, next_color = len(grid), len(grid[0]), 2
    component_size = collections.defaultdict(int)

    def paint(x, y, color):
        if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
            grid[x][y] = color
            component_size[color] += 1
            for dx, dy in directions:
                paint(x+dx, y+dy, color)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                paint(i, j, next_color)
                next_color += 1

    result = max(component_size.values() or [0])
    for x in range(m):
        for y in range(n):
            if grid[x][y] > 0:
                continue
            neighbor_colors = set()
            for dx, dy in directions:
                nx, ny = x+dx, y+dy
                if nx < 0 or nx == m or ny < 0 or ny == n or grid[nx][ny] == 0:
                    continue
                neighbor_colors.add(grid[nx][ny])
            size_formed = 1  # Start with 1, which is matrix[r][c] when turning from 0 into 1
            for color in neighbor_colors:
                size_formed += component_size[color]
            result = max(result, size_formed)
    return result


def three_sum(nums):
    result, negatives, positives, zeros = set(), [], [], []

    for num in nums:
        if num > 0:
            positives.append(num)
        elif num < 0:
            negatives.append(num)
        else:
            zeros.append(num)

    N, P = set(negatives), set(positives)

    if len(zeros) >= 3:
        result.add((0, 0, 0))

    if zeros:
        for num in P:
            if -num in N:
                result.add((-num, 0, num))

    for i in range(len(negatives)):
        for j in range(i + 1, len(negatives)):
            target = -1 * (negatives[i] + negatives[j])
            if target in P:
                result.add((negatives[i], negatives[j], target))

    for i in range(len(positives)):
        for j in range(i + 1, len(positives)):
            target = -1 * (positives[i] + positives[j])
            if target in N:
                result.add((positives[i], positives[j], target))

    return result


def simplify_path(path):
    stack = []
    for element in path.split("/"):
        if stack and element == '..':
            stack.pop()
        elif element not in ['.', '', '..']:
            stack.append(element)
    return '/' + '/'.join(stack)


def group_shifted_strings(strings):
    # https://baihuqian.github.io/2018-07-26-group-shifted-strings/
    # https://techyield.blogspot.com/2020/10/group-shifted-strings-python-solution.html
    if len(strings) == 0:
        return []
    groups = collections.defaultdict(list)
    for s in strings:
        key = []
        for i in range(len(s) - 1):
            key.append((26 + ord(s[i + 1]) - ord(s[i])) % 26)
        groups[tuple(key)].append(s)
    return groups.values()


def insert_into_a_sorted_circular_linked_list(head, node):
    # https://www.geeksforgeeks.org/sorted-insert-for-circular-linked-list/
    current = head
    if current is None:
        node.next = node
        head = node
    elif current.val >= node.val:
        while current.next != head:
            current = current.next
        current.next = node
        node.next = head
        head = node
    else:
        while current.next != head and current.next.val < node.val:
            current = current.next
        node.next = current.next
        current.next = node


def longest_substring_with_at_most_k_distinct_characters(s, k):
    counter = collections.Counter()
    left, result = 0, 0
    for right, char in enumerate(s):
        counter[char] += 1
        while len(counter) > k:
            counter[s[left]] -= 1
            if counter[s[left]] == 0:
                del counter[s[left]]
            left += 1
        result = max(result, right - left + 1)
    return result


def combination_sum(candidates, target):
    def backtrack(current, path, k):
        if current == 0:
            result.append(path[:])
        if current < 0 or k >= n:
            return
        for i in range(k, n):
            chosen = candidates[i]
            backtrack(current - chosen, path + [chosen], i)

    result, n = [], len(candidates)
    backtrack(target, [], 0)
    return result


def read_n_characters_given_read4_ii_call_multiple_times():
    pass


def balance_a_binary_search_tree(root):
    def dfs(node):
        if not node:
            return []
        return dfs(node.left) + [node.val] + dfs(node.right)

    values = dfs(root)

    def build(l, r):
        if l > r:
            return None
        mid = (l + r) // 2
        node = TreeNode(values[mid])
        node.left = build(l, mid - 1)
        node.right = build(mid + 1, r)
        return node
    return build(0, len(values) - 1)


def remove_all_adjacent_duplicates_in_string(s):
    stack = []
    for char in s:
        if stack and stack[-1] == char:
            stack.pop()
        else:
            stack.append(char)
    return ''.join(stack)


def word_ladder_length(begin_word, end_word, word_list):
    if end_word not in word_list or not begin_word \
            or not end_word or not word_list \
            or len(begin_word) != len(end_word):
        return 0
    n, hashmap = len(begin_word), collections.defaultdict(list)
    for word in word_list:
        for i in range(n):
            hashmap[word[:i] + '*' + word[i+1:]].append(word)

    queue, visited = collections.deque([(begin_word, 1)]), {[begin_word]}
    while queue:
        current_word, level = queue.popleft()
        for i in range(n):
            intermediate_word = current_word[:i] + '*' + current_word[i+1:]
            for word in hashmap[intermediate_word]:
                if word == end_word:
                    return level+1
                if word not in visited:
                    visited.add(word)
                    queue.append((word, level+1))
    return 0


def two_sum(nums, target):
    hashmap = {num: i for i, num in enumerate(nums)}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in hashmap and i != hashmap[diff]:
            return [i, hashmap[diff]]


def max_area(grid):
    m, n, result = len(grid), len(grid[0]), 0
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(x, y, area):
        if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
            area, grid[x][y] = area + 1, 0
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                area = dfs(nx, ny, area)
        return area

    for x in range(m):
        for y in range(n):
            result = max(result, dfs(x, y, 0))
    return result


class RandomPickIndex:
    def __init__(self, nums):
        pass

    def pick(self, target):
        pass


