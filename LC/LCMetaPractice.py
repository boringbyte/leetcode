import heapq
import collections
import itertools
import string
from queue import PriorityQueue


class RandomPointerNode:
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random


class ListNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None


class DLLNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.prev = None


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word):
        node = self.root
        for char in word:
            node = node.children[char]
        node.is_word = True

    def search(self, word):
        n = len(word)

        def dfs(node, i):
            if i == n:
                return node.is_word
            if word[i] == '.':
                for child in node.children:
                    if dfs(child, i + 1): return True
            if word[i] in node.children:
                return dfs(node.children[word[i]], i + 1)
            return False

        return dfs(self.root, 0)


class LRUNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None
        self.prev = None


class LRUCache1:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head, self.tail = LRUNode(0, 0), LRUNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        p, n = node.prev, node.next
        p.next, n.prev = n, p

    def _add(self, node):
        p = self.tail.prev
        p.next = node
        node.prev = p
        self.tail.prev = node
        node.next = self.tail

    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.val
        return -1

    def set(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        current = LRUNode(key, value)
        self._add(current)
        self.cache[key] = current
        if len(self.cache) > self.capacity:
            node = self.head.next
            self._remove(node)
            del self.cache[node.key]


class BinaryTreeCodec:
    pass


def binary_tree_vertical_order_traversal_bfs(root):
    if root is None:
        return
    queue, hashmap = collections.deque([(root, 0)]), {}
    while queue:
        current, distance = queue.popleft()
        hashmap.setdefault(distance, []).append(current.val)
        if current.left:
            queue.append((current.left, distance - 1))
        if current.right:
            queue.append((current.right, distance + 1))

    for key in sorted(hashmap.keys()):
        print(hashmap.get(key))


def binary_tree_vertical_order_traversal_pre(root):
    hashmap = {}

    def dfs(node, distance):
        if node is None:
            return
        hashmap.setdefault(distance, []).append(node.val)
        dfs(node.left, distance - 1)
        dfs(node.right, distance + 1)

    dfs(root, 0)
    for value in hashmap.values():
        print(value)


def max_consecutive_ones_3(nums, k):
    i, j, n, result, n_zeros = 0, 0, len(nums), 0, 0
    for j in range(n):
        if nums[j] == 0:
            n_zeros += 1
        if n_zeros > k:
            if nums[i] == 0:
                n_zeros -= 1
            i += 1
        if n_zeros <= k:
            result = max(result, j - i + 1)
    return result


def remove_all_adjacent_duplicates_2(s, k):
    stack = []  # [char, freq]
    for char in s:
        if stack and stack[-1][0] == char:
            stack[-1][1] += 1
        else:
            stack.append([char, 1])
        if stack[-1][1] == k:
            stack.pop()
    return ''.join(char * freq for char, freq in stack)


def diagonal_traversal_of_matrix(matrix):
    m, n, hashmap = len(matrix), len(matrix[0]), {}
    for i in range(m):
        for j in range(n):
            hashmap.setdefault(i + j, []).append(matrix[i][j])

    result, direction = [], -1
    for value in hashmap.values():
        result.extend(value[::direction])
        direction *= -1
    return result


def is_complete_binary_tree_bfs(root):
    queue = collections.deque([root])
    have_null = False
    while queue:
        current = queue.popleft()
        if not current:
            have_null = True
            continue
        if have_null:
            return False
        queue.append(current.left)
        queue.append(current.right)
    return True


def permute(nums):
    result, n = [], len(nums)
    visited = [0] * n

    def backtrack(sofar):
        if len(sofar) == n:
            result.append(sofar[:])
        else:
            for i in range(n):
                if visited[i] != 1:
                    chosen, visited[i] = nums[i], 1
                    backtrack(sofar + [chosen])
                    visited[i] = 0

    backtrack(sofar=[])
    return result


def build_adjacency_list(n, edges):
    adj_list = [[] for _ in range(n)]
    for c1, c2 in edges:
        adj_list[c2].append(c1)
    return adj_list


def course_schedule(num_courses, prerequisites):
    adj_list = build_adjacency_list(num_courses, prerequisites)
    in_degree = [0] * num_courses
    for v1, v2 in prerequisites:
        in_degree[v1] += 1

    queue = collections.deque([])
    for v in range(num_courses):
        if in_degree[v] == 0:
            queue.append(v)
    count, top_order = 0, []
    while queue:
        v = queue.popleft()
        top_order.append(v)
        count += 1
        for des in adj_list[v]:
            in_degree[des] -= 1
            if in_degree[des] == 0:
                queue.append(des)

        if count != num_courses:
            return False
        else:
            return True


def reverse_linked_list(head):
    prev = None
    while head:
        current = head
        head = head.next
        current.next = prev
        prev = current
    return prev


def verify_alien_dictionary(words, order):
    def check_order(word1, word2):
        for char1, char2 in zip(word1, word2):
            if order_map[char1] != order_map[char2]:
                return order_map[char1] < order_map[char2]
        return len(word1) <= len(word2)

    order_map = {char: i for i, char in enumerate(order)}
    return all(check_order(word1, word2) for word1, word2 in zip(words, words[1:]))


def minimum_remove_to_make_valid_parenthesis_method1(s):
    in_valid, stack = set(), []
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                in_valid.add(i)
    for i in stack:
        in_valid.add(i)

    return ''.join([char for i, char in enumerate(s) if i not in in_valid])


def minimum_remove_to_make_valid_parenthesis_method2(s):
    s, stack = list(s), []
    for i, char in s:
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                s[i] = ''
    while stack:
        s[stack.pop()] = ''
    return ''.join(s)


def k_closest_points_method1(points, k):
    points = sorted(points, key=lambda x: x[0] ** 2 + x[1] ** 2)
    return points[:k]


def squared_distance(points):
    return points[0] ** 2 + points[1] ** 2


def k_closest_points_method2(points, k):
    heap = [(-squared_distance(points[i]), i) for i in range(k)]
    heapq.heapify(heap)
    for i in range(k, len(points)):
        distance = -squared_distance(points[i])
        if distance > heap[0][0]:
            heapq.heappushpop(heap, (distance, i))
    return [points[i] for (_, i) in heap]


def product_of_array_except_self(nums):
    n = len(nums)
    prefix, suffix = [1] * n, [1] * n
    for i in range(1, n):
        prefix[i] = prefix[i - 1] * nums[i - 1]
    for i in range(n - 2, -1, -1):
        suffix[i] = suffix[i + 1] * nums[i + 1]
    return [a * b for a, b in zip(prefix, suffix)]


def valid_palindrome_2(s):
    def check_palindrome(s, i, j):
        while i < j:
            if s[i] != s[j]:
                return False
            i, j = i + 1, j - 1
        return True

    i, j = 0, len(s) - 1
    while i < j:
        if s[i] != s[j]:
            return check_palindrome(s, i + 1, j) or check_palindrome(s, i, j - 1)
        i, j = i + 1, j - 1
    return True


def sub_array_sum_equals_k(nums, k):
    pass


def string_to_digit(num):
    return ord(num) - ord('0')


def add_strings(num1, num2):
    n1, n2, carry, result = len(num1) - 1, len(num2) - 1, 0, []
    while n1 >= 0 or n2 >= 0 or carry:
        digit1 = string_to_digit(num1[n1]) if n1 >= 0 else 0
        digit2 = string_to_digit(num2[n2]) if n2 >= 0 else 0
        carry, digit = divmod(digit1 + digit2 + carry, 10)
        result.append(str(digit))
        n1, n2 = n1 - 1, n2 - 1
    return ''.join(result[::-1])


def add_binary_strings(a, b):
    n1, n2, carry, result = len(a) - 1, len(b) - 1, 0, []
    while n1 >= 0 or n2 >= 0 or carry:
        digit1 = string_to_digit(a[n1]) if n1 >= 0 else 0
        digit2 = string_to_digit(b[n2]) if n2 >= 0 else 0
        carry, digit = divmod(digit1 + digit2 + carry, 2)
        result.append(str(digit))
        n1, n2 = n1 - 1, n2 - 1
    return ''.join(result[::-1])


def merge_intervals(intervals):
    intervals, result = sorted(intervals, key=lambda x: x[0]), []
    for interval in intervals:
        if not result or result[-1][-1] < interval[0]:
            result.append(interval)
        else:
            result[-1][-1] = max(result[-1][-1], interval[-1])
    return result


def is_valid_palindrome(s):
    def clean_string(s):
        s, t, valid_chars = s.lower(), '', set(string.ascii_lowercase + '0123456789')
        for char in s:
            if char in valid_chars:
                t += char
        return t

    s = clean_string(s)
    l, r = 0, len(s) - 1
    while l < r:
        if s[l] != s[r]:
            return False
        l, r = l + 1, r - 1
    return True


class SparseVector:
    def __init__(self, nums):
        self.hashmap = {i: val for i, val in enumerate(nums) if val}

    def dot_product(self, vec):
        if len(self.hashmap) > len(vec.hashmap):
            self, vec = vec, self
        return sum(val * vec.hashmap[i] for i, val in self.hashmap.items() for i in vec.hashmap)


def range_sum_BST(root, low, high):
    result = [0]

    def dfs(node):
        if node:
            if low <= node.val <= high:
                result[0] += node.val
            if node.val > low: dfs(node.left)
            if node.val < high: dfs(node.right)

    dfs(root)
    return result[0]


def right_side_view_binary_tree(root):
    if not root:
        return []
    result, queue = [], collections.deque([root])
    while queue:
        for i in range(len(queue)):
            current = queue.popleft()
            if i == 0:
                result.append(current.val)
            if current.right:
                queue.append(current.right)
            if current.left:
                queue.append(current.left)
    return result


def least_common_ancestor_iterative(root, p, q):
    stack, parent = [root], {root: None}
    while p not in parent and q not in parent:
        current = stack.pop()
        if current.left:
            parent[current.left] = current
            stack.append(current.left)
        if current.right:
            parent[current.right] = current
            stack.append(current.right)
    ancestors = set()
    while p:
        ancestors.add(p)
        p = parent[p]
    while q not in ancestors:
        q = parent[q]
    return q


def least_common_ancestor_recursive(root, p, q):
    if root is None or p == root or q == root:
        return root
    left = least_common_ancestor_recursive(root.left, p, q)
    right = least_common_ancestor_iterative(root.right, p, q)
    if left and right:
        return root
    return left or right


def first_bad_version(n):
    def is_bad_version(x):
        return x > 0

    lo, hi = 0, n
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if is_bad_version(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def binary_search_tree_to_dll(root):
    if root is None:
        return root
    dummy = prev = DLLNode(-1)
    stack, current = [], root
    while True:
        if current is not None:
            stack.append(current)
            current = current.left
        elif stack:
            current = stack.pop()
            node = DLLNode(current.val)
            prev.next = node
            node.prev = prev
            prev = node
            current = current.right
        else:
            break
    return dummy.next


def accounts_merge_dfs(accounts):
    n, result = len(accounts), []
    email_accounts_map, visited_accounts = collections.defaultdict(list), [False] * n
    for i, account in enumerate(accounts):
        for j in range(1, len(account)):
            email = account[j]
            email_accounts_map[email].append(i)

    def dfs(i, emails):
        if visited_accounts[i]:
            return
        visited_accounts[i] = True
        for j in range(1, len(accounts[i])):
            email = accounts[i][j]
            emails.add(email)
            for neighbor in email_accounts_map[email]:
                dfs(neighbor, emails)

    for i, account in enumerate(accounts):
        if visited_accounts[i]:
            continue
        name, emails = accounts[0], set()
        dfs(i, emails)
        result.append([name] + sorted(emails))
    return result


def accounts_merge_dfs2(accounts):
    graph, visited, result = collections.defaultdict(list), set(), []
    for account in accounts:
        for i in range(2, len(account)):
            graph[account[i]].append(account[i - 1])
            graph[account[i - 1]].append(account[i])

    def dfs(email):
        visited.add(email)
        emails = [email]
        for edge in graph[email]:
            if edge not in visited:
                emails.extend(dfs(edge))
        return emails

    for account in accounts:
        if account[1] not in visited:
            result.append([account[0] + sorted(dfs(account[1]))])
    return result


def remove_invalid_parenthesis(s):
    pass


def merge_k_lists_pq(lists):
    dummy = current = ListNode()
    pq = PriorityQueue()
    for l in lists:
        if l:
            pq.put((l.val, l))
    while not pq.empty():
        val, node = pq.get()
        current.next = ListNode(val)
        current = current.next
        node = node.next
        if node:
            pq.put((node.val, node))
    return dummy.next


def merge(list1, list2):
    dummy = current = ListNode()
    while list1 and list2:
        if list1.val < list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next
    current.next = list1 or list2
    return dummy.next


def merge_k_list_divide_conquer(lists):
    if not lists:
        return None
    if len(lists) == 1:
        return lists[0]
    mid = len(lists) // 2
    list1, list2 = merge_k_list_divide_conquer(lists[:mid]), merge_k_list_divide_conquer(lists[mid:])
    return merge(list1, list2)


def interval_list_intersections(list1, list2):
    i, j, n1, n2, result = 0, 0, len(list1), len(list2), []
    while i < n1 and j < n2:
        lo = max(list1[0], list2[0])
        hi = min(list1[1], list2[1])
        if lo <= hi:
            result.append([lo, hi])
        if list1[i][1] < list2[j][1]:
            i += 1
        else:
            j += 1
    return result


def squares_of_sorted_array(nums):
    n = len(nums)
    result, i, j, k = [None] * n, 0, n - 1, n - 1
    while i <= j:
        l, r = nums[i] ** 2, nums[j] ** 2
        result[k] = max(l, r)
        if l >= r:
            i += 1
        else:
            j -= 1
        k -= 1
    return result


def making_largest_island(grid):
    pass


def three_sum(nums):
    pass  # Check three pointer solution as well


def simplify_path(path):
    stack = []
    for e in path.split('/'):
        if stack and e == '..':
            stack.pop()
        elif e not in ['.', '', '..']:
            stack.append(e)
    return '/' + '/'.join(stack)


def group_shifting_string(s):  # Didn't understand
    if len(s) == 0:
        return []

    def get_diff_string(s):
        shift = ''
        for x, y in zip(s, s[1:]):
            diff = (ord(x) - ord(y))
            if diff < 0:
                diff += 26
            shift += chr(diff + ord('a'))
        return shift

    hashmap, n = collections.defaultdict(list), len(s)
    for i in range(n):
        diff_str = get_diff_string(s[i])
        hashmap[diff_str].append(i)
    return hashmap.values()


def sorted_circular_insert(head, node):
    # https://www.geeksforgeeks.org/sorted-insert-for-circular-linked-list/
    current = head
    if current is None:
        node.next = node
        head = node
    elif current.data >= node.data:
        while current.next != head:
            current = current.next
        current.next = node.next
        node.next = head
        head = node
    else:
        while current.next != head and current.next.data < node.data:
            current = current.next
        node.next = current.next
        current.next = node


############


def island_perimeter(grid):
    m, n, result = len(grid), len(grid[0]), 0
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0:
                continue
            result += 4
            for di, dj in directions:
                x, y = i + di, j + dj
                if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
                    result -= 1
    return result


def path_sum2_dfs_stack(root, target_sum):
    # replace stack with queue to make it BFS solution
    if root is None:
        return []
    result, stack = [], [(root, root.val, [root.val])]
    while stack:
        current, total, path = stack.pop()
        if not current.left and not current.right and total == target_sum:
            result.append(path)
        if current.right:
            value = current.right.val
            stack.append((current.right, total + value, path + [value]))
        if current.left:
            value = current.left.val
            stack.append((current.left, total + value, path + [value]))


def is_monotonic(nums):
    inc = dec = True
    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]:
            inc = False
        if nums[i] < nums[i + 1]:
            dec = False
    return inc or dec


def container_with_most_water(heights):
    result, l, r = 0, 0, len(heights) - 1
    while l < r:
        area = min(heights[l], heights[r]) * (r - l)
        result = max(result, area)
        if heights[l] < heights[r]:
            l += 1
        else:
            r -= 1
    return result


def connected_components_undirected_graph(n, edges):
    graph, visited, result = collections.defaultdict(list), [0] * n, 0
    components = [-1] * n
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    def dfs(at):
        visited[at] = True
        components[at] = result
        neighbours = graph[at]
        for neighbour in neighbours:
            if not visited[neighbour]:
                dfs(neighbour)

    for i in range(n):
        if not visited[i]:
            result += 1
            dfs(i)
    return result, components


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


def flatten_binary_tree_to_linked_list_morris(root):
    current = root
    while current:
        if current.left:
            runner = current.left
            while runner.right:
                runner = runner.right
            runner.right = current.right
            current.right = current.left
            current.left = None
        current = current.right


def toeplitz_matrix(matrix):
    m, n = len(matrix), len(matrix[0])
    for i in range(m - 1):
        for j in range(n - 1):
            if matrix[i][j] != matrix[i + 1][j + 1]:
                return False
    return True


def buildings_with_an_ocean_view1(heights):
    n, result, prev_max = len(heights), [heights[-1]], heights[-1]
    for i in range(n - 2, -1, -1):
        if heights[i] > prev_max:
            result.append(heights[i])
        prev_max = max(prev_max, heights[i])
    return result[::-1]


def buildings_with_an_ocean_view2(heights):
    stack = []
    for i, height in enumerate(heights):
        while stack and heights[stack[-1]] <= height:
            stack.pop()
        stack.append(i)
    return stack


def closest_binary_search_tree_value(root, target):
    gap = result = float('inf')
    while root:
        if abs(root.val - target) < gap:
            gap = abs(root.val - target)
            result = root.val
        if target == root.val:
            break
        elif target < root.val:
            root = root.left
        else:
            root = root.right
    return result


def remove_linked_list_elements1(head, val):
    current = head
    new_head = dummy = ListNode(-1)
    while current:
        if val != current.val:
            dummy.next = ListNode(current.val)
            dummy = dummy.next
        current = current.next
    return new_head.next


def remove_linked_list_elements2(head, val):
    dummy = ListNode(-1)
    prev, dummy.next, current = dummy, head, head
    while current:
        if current.val != val:
            prev = current
        else:
            prev.next = current.next
        current = current.next
    return dummy.next


def all_nodes_distance_k_binary_tree(root, target, k):
    graph = collections.defaultdict(list)

    def build_graph(node, parent):
        if parent:
            graph[node].append(parent)
        if node.left:
            graph[node].append(node.left)
            build_graph(node.left, node)
        if node.right:
            graph[node].append(node.right)
            build_graph(node.right, node)

    visited, result = {target}, []
    build_graph(root, None)

    def dfs(node, distance):
        if distance == 0:
            result.append(node.val)
        else:
            visited.add(node)
            for neighbour in graph[node]:
                if neighbour not in visited:
                    dfs(neighbour, distance - 1)

    dfs(target, k)
    return result


def number_of_island(grid):
    m, n, result = len(grid), len(grid[0]), 0
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(x, y):
        if 0 <= x < m and 0 <= y < n and grid[x][y] == '1':
            grid[x][y] = '0'
            for dx, dy in directions:
                dfs(x + dx, y + dy)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
                result += 1
    return result


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


# 111
def is_palindrome_linked_list(head):
    prev, slow, fast = None, head, head

    while fast and fast.next:
        fast = fast.next.next
        temp = slow.next
        slow.next = prev
        prev = slow
        slow = temp

    if fast:
        slow = slow.next

    while slow:
        if prev.val != slow.val:
            return False
        prev = prev.next
        slow = slow.next
    return True


# 118
def range_sum_query_2d(matrix, row1, col1, row2, col2):
    m, n = len(matrix), len(matrix[0])
    prefix_sum = [[0] * (n+1) for _ in range(m+1)]
    for r in range(1, m+1):
        for c in range(1, n+1):
            prefix_sum[r][c] = \
                prefix_sum[r-1][c] + \
                prefix_sum[r][c-1] + \
                prefix_sum[r-1][c-1] + \
                matrix[r-1][c-1]

    r1, c1, r2, c2 = row1+1, col1+1, row2+1, col2+1
    return prefix_sum[r2][c2] - prefix_sum[r2][c1-1] - prefix_sum[r1-1][c2] + prefix_sum[r1-1][c1-1]


def reverse_a_number1(x):
    sign = 1 if x > 0 else -1
    digit = sign * int(str(abs(x))[::-1])
    max_value = 2**31  # for a 32 bit machine
    return digit if -max_value - 1 < digit < max_value else 0


def reverse_a_number2(x):
    sign = 1 if x > 0 else -1
    result, x, int32_max = 0, sign*x, 2**32-1
    if x < 0:
        int32_max += 1

    while x:
        if result > int32_max // 10:
            return 0
        digit = x % 10
        result *= 10

        if int32_max - result < digit:
            return 0
        result += digit
        x //= 10
    return sign * result


def sum_root_to_leaf_numbers1(root):
    def dfs(node, value):
        if node is None:
            return 0
        value = value * 10 + node.val
        if node.left is None and node.right is None:
            return value
        return dfs(node.left, value) + dfs(node.right, value)
    return dfs(root, 0)


def sum_root_to_leaf_numbers2(root):
    stack, result = [(root, 0)], 0
    while len(stack):
        current, value = stack.pop()
        value = value * 10 + current.val
        if current.left is None and current.right is None:
            result += value
        if current.right:
            stack.append((current.right, value))
        if current.left:
            stack.append((current.left, value))
    return result


def sum_root_to_leaf_numbers3(root):
    queue, result = collections.deque((root, 0)), 0
    while len(queue):
        current, value = queue.popleft()
        value = value * 10 + current.val
        if current.left:
            queue.append((current.left, value))
        if current.right:
            queue.append((current.right, value))
        if current.left is None and current.right is None:
            result += value
    return result
