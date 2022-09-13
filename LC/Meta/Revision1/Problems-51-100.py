import heapq
import collections
import random
from functools import lru_cache
from LC.LCMetaPractice import ListNode, TreeNode, RandomPointerNode


def closest_binary_search_tree_value(root, target):
    difference = result = float('inf')
    while root:
        if abs(root.val - target) < difference:
            difference = abs(root.val - target)
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
            grid[x][y] = '0'
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


def top_k_frequent_elements(nums, k):
    counts, freq_dict, result = collections.Counter(nums), collections.defaultdict(list), []

    for num, freq in counts.items():
        freq_dict[freq].append(num)

    for freq in reversed(range(len(nums) + 1)):
        result.extend(freq_dict[freq])
        if len(result) >= k:
            return result[:k]
    return result[:k]


def search_in_rotated_sorted_array(nums, target):
    # https://leetcode.com/problems/search-in-rotated-sorted-array/discuss/14437/Python-binary-search-solution-O(logn)-48ms
    if not nums:
        return -1
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if target == nums[mid]:
            return mid
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target <= nums[mid]:
                hi = mid + 1
            else:
                lo = mid - 1
        else:
            if nums[lo] <= target <= nums[mid]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


def largest_island(grid):
    # https://leetcode.com/problems/making-a-large-island/discuss/1375992/C%2B%2BPython-DFS-paint-different-colors-Union-Find-Solutions-with-Picture-Clean-and-Concise
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    m, n, next_color = len(grid), len(grid[0]), 2
    component_size = collections.defaultdict(int)

    def paint(x, y, color):
        if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
            grid[x][y] = color
            component_size[color] += 1
            for dx, dy in directions:
                paint(x + dx, y + dy, color)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                paint(i, j, next_color)
                next_color += 1

    result = max(component_size.values() or [0])

    for x in range(m):
        for y in range(n):
            if grid[x][y] > 0:  # We are concerned about grids with no color, so skipping anything with colors
                continue
            neighbor_colors = set()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx == m or ny < 0 or ny == n or grid[nx][ny] == 0:  # We are concerned about grids with colors only to make a large island
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
    for word in strings:
        key = []
        for i in range(len(word) - 1):
            key.append((26 + ord(word[i + 1]) - ord(word[i])) % 26)
        groups[tuple(key)].append(word)
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
    # https://shareablecode.com/snippets/longest-substring-with-at-most-k-distinct-characters-python-solution-leetcode-Q8vt-7cbC
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
    def backtrack(current, sofar, k):
        if current == 0:
            result.append(sofar[:])
        if current < 0 or k >= n:
            return
        for i in range(k, n):
            chosen = candidates[i]
            backtrack(current - chosen, sofar + [chosen], i)

    result, n = [], len(candidates)
    backtrack(target, [], 0)
    return result


def read():
    return 0


class ReadNCharacters:
    # https://poopcode.com/read-n-characters-given-read4-ii-call-multiple-times-leetcode-challenge-python-solution/
    def __init__(self):
        self.buffer = [''] * 4
        self.offset = 0
        self.buffer_size = 0

    def read_n_characters_given_read4_ii_call_multiple_times(self, buf, n):
        pos, eof = 0, False
        while not eof and pos < n:
            if self.buffer_size == 0:
                self.buffer_size = read(self.buffer)
                eof = self.buffer_size < 4
            byte = min(n - pos, self.buffer_size)
            for i in range(byte):
                buf[pos + i] = self.buffer[self.offset + i]
            self.offset = (self.offset + byte) % 4
            self.buffer_size -= byte
            pos += byte
        return pos


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


def word_ladder(begin_word, end_word, word_list):
    # https://leetcode.com/problems/word-ladder/discuss/346920/Python3-Breadth-first-search
    if end_word not in word_list or not begin_word \
            or not end_word or not word_list \
            or len(begin_word) != len(end_word):
        return 0
    n, hashmap = len(begin_word), collections.defaultdict(list)
    for word in word_list:
        for i in range(n):
            hashmap[word[:i] + '*' + word[i + 1:]].append(word)

    queue, visited = collections.deque([(begin_word, 1)]), {[begin_word]}
    while queue:
        current_word, level = queue.popleft()
        for i in range(n):
            intermediate_word = current_word[:i] + '*' + current_word[i + 1:]
            for word in hashmap[intermediate_word]:
                if word == end_word:
                    return level + 1
                if word not in visited:
                    visited.add(word)
                    queue.append((word, level + 1))
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
    # https://leetcode.com/problems/random-pick-index/discuss/88153/Python-reservoir-sampling-solution.
    def __init__(self, nums):
        self.nums = nums

    def pick(self, target):
        result, count = None, 0
        for i, num in enumerate(self.nums):
            if num == target:
                count += 1
                chance = random.randint(1, count)
                if chance == count:
                    result = i
        return result


def subsets(nums):
    result, n = [], len(nums)

    def backtrack(sofar, k):
        result.append(sofar[:])
        for i in range(k, n):
            chosen = nums[i]
            backtrack(sofar + [chosen], i + 1)

    backtrack([], 0)
    return result


def palindrome_permutation(s):
    # https://cheonhyangzhang.gitbooks.io/leetcode-solutions/content/266-palindrome-permutation.html
    counter, n = collections.Counter(s), len(s)
    odd_count = 0
    for key in counter.keys():
        if counter[key] % 2 == 1:
            odd_count += 1
    if odd_count == 1 or odd_count % 2 == 0:
        return True
    else:
        return False


def minimum_add_to_make_parentheses_valid1(s):
    left, right = 0, 0
    for char in s:
        if char == '(':
            right += 1
        elif right > 0:
            right -= 1
        else:
            left += 1
    return left + right


def minimum_add_to_make_parentheses_valid2(s):
    right, stack = 0, []
    for char in s:
        if char == '(':
            stack.append(char)
        elif stack and char == ')':
            stack.pop()
        else:
            right += 1
    return right + len(stack)


def remove_nth_from_end1(head, n):
    length, current = 0, head
    while current:
        current = current.next
        length += 1

    current = head
    for _ in range(1, length - n):
        current = current.next

    current.next = current.next.next
    return head


def remove_nth_from_end2(head, n):
    slow = fast = head
    for _ in range(n):
        fast = fast.next
    if not fast:
        return head.next
    while fast.next:
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next
    return slow


class RandomizedSet:
    def __init__(self):
        self.nums, self.positions = [], {}

    def insert(self, val):
        if val not in self.positions:
            self.positions[val] = len(self.nums)
            self.nums.append(val)
            return True
        return False

    def remove(self, val):
        if val in self.positions:
            idx, last = self.positions[val], self.nums[-1]
            self.nums[idx], self.positions[last] = last, idx
            self.nums.pop()
            self.positions.pop(val, 0)
            return True
        return False

    def get_random(self):
        return random.choice(self.nums)


def multiply_strings(num1, num2):
    m, n = len(num1), len(num2)
    result = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            value = (ord(num1[i]) - ord('0')) * (ord(num2[j]) - ord('0'))
            p1, p2 = i + j, i + j + 1
            total = value + result[p2]
            result[p1] += total // 10
            result[p2] = total % 10
    final_result = []
    for value in result:
        if value == 0 and len(final_result) == 0:
            continue
        else:
            final_result.append(str(value))
    return '0' if len(final_result) == 0 else ''.join(final_result)


def add_operators(s, target):
    result, n = [], len(s)

    def backtrack(sofar, k, result_sofar, prev):
        if k == len(s):
            if result_sofar == target:
                result.append(sofar[:])
                return
        for i in range(k, n):
            if i > k and s[i] == '0':  # leading zero number skipped
                break
            num = int(s[k:i + 1])
            if i == 0:
                backtrack(i + 1, sofar + str(num), result_sofar + num,
                          num)  # First num, pick it without adding any operator
            else:
                backtrack(i + 1, sofar + '+' + str(num), result_sofar + num, num)
                backtrack(i + 1, sofar + '-' + str(num), result_sofar - num, -num)
                backtrack(i + 1, sofar + '*' + str(num), result_sofar - prev + prev * num, prev * num)

    backtrack('', 0, 0, 0)
    return result


def longest_increasing_path_matrix_dfs(matrix):
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    m, n, result = len(matrix), len(matrix[0]), 0

    @lru_cache
    def dfs(x, y):
        local_result = 1
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[x][y]:
                local_result = max(local_result, dfs(nx, ny) + local_result)
        return local_result

    for x in range(m):
        for y in range(n):
            result = max(result, dfs(x, y))
    return result


def copy_list_with_random_pointer1(head: RandomPointerNode):
    hashmap, prev, current = {}, None, head
    while current:
        if current not in hashmap:
            hashmap[current] = RandomPointerNode(current.val, current.next, current.random)
        if prev:
            prev.next = hashmap[current]
        else:
            head = hashmap[current]

        random_node = current.random
        if random_node:
            if random_node not in hashmap:
                hashmap[random_node] = RandomPointerNode(random_node.val, random_node.next, random_node.random)
            hashmap[current].random = hashmap[random_node]
        prev, current = hashmap[current], current.next
    return head


class NestedIterator:
    # https://leetcode.com/problems/flatten-nested-list-iterator/discuss/1156268/JS-Python-Java-C%2B%2B-or-Simple-Queue-Solution-w-Explanation
    def __init__(self, nested_list):
        self.data = []
        self.flatten(nested_list)

    def flatten(self, lst):
        for el in lst:
            if el.isInteger():
                self.data.append(el.getInteger())
            else:
                self.flatten(el.getList())

    def next(self):
        return self.data.pop(0)

    def has_next(self):
        return len(self.data)


def find_peak_element(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > mid[mid + 1]:
            r = mid
        else:
            l = mid + 1
    return l


def basic_calculator_2(s):
    def update(op, value):
        if op == '+': stack.append(value)
        if op == '-': stack.append(-value)
        if op == '*': stack.append(stack.pop() * value)
        if op == '/': stack.append(int(stack.pop()) / value)

    i, num, stack, sign = 0, 0, [], '+'

    while i < len(s):
        if s[i].isdigit():
            num = num * 10 + int(s[i])
        elif s[i] in '+-*/':
            update(sign, num)
            num, sign = 0, s[i]
        elif s[i] == '(':
            num, j = basic_calculator_2(s[i + 1:])
            i = i + j
        elif s[i] == ')':
            update(sign, num)
            return sum(stack), i + 1
        i += 1
    update(sign, num)
    return sum(stack)


def add_two_numbers(l1, l2):
    dummy = current = ListNode(0)
    carry = 0
    while l1 or l2 or carry:
        if l1:
            carry += l1.val
            l1 = l1.next
        if l2:
            carry += l2.val
            l2 = l2.next
        carry, rem = divmod(carry, 10)
        current.next = ListNode(rem)
        current = current.next
    return dummy.next


def is_graph_bipartite(graph):
    def dfs(start):
        if loop[0]:
            return
        for neighbor in graph[start]:
            if distance[neighbor] >= 0 and distance[neighbor] == distance[start]:
                loop[0] = True
            elif distance[neighbor] < 0:
                distance[neighbor] = distance[start] ^ 1
                dfs(neighbor)

    n = len(graph)
    loop, distance = [False], [-1] * n

    for i in range(n):
        if loop[0]:
            return False
        if distance[i] == -1:
            distance[i] = 0
            dfs(i)
    return True


def recyclable_and_low_fat_products():
    pass  # This is sql


def three_sum_closest(nums, target):
    nums, n = sorted(nums), len(nums)
    if n < 3:
        return
    result = sum(nums[:3])
    for i in range(n - 2):
        l, r = i + 1, n - 1
        local_result = nums[i] + nums[l] + nums[r]
        if abs(result - target) > abs(local_result - target):
            result = local_result
        if local_result == target:
            return target
        if local_result > target:
            r -= 1
        else:
            l += 1
    return result


def maximum_swap(num):
    s = list(str(num))
    n = len(s)
    for i in range(n - 1):
        if s[i] < s[i + 1]:
            break
    else:
        return num
    max_idx, max_val = i + 1, s[i + 1]
    for j in range(i + 1, n):
        if max_val <= s[j]:
            max_idx, max_val = j, s[j]
    left_idx = i
    for j in range(i + 1):
        if s[j] < max_val:
            left_idx = j
            break
    s[max_idx], s[left_idx] = s[left_idx], s[max_idx]
    return int(''.join(s))


def find_k_closest_elements(arr, k, x):
    # https://leetcode.com/problems/find-k-closest-elements/discuss/1311191/C%2B%2BJavaPython-Binary-Search-With-Diagram-explain-Clean-and-Concise
    n, i = len(arr), 0
    l, r = 0, n - k
    while l <= r:
        mid = l + (r - l) // 2
        if mid + k == n or x - arr[mid] <= arr[mid + k] - x:
            i = mid
            r = mid - 1
        else:
            l = mid + 1
    return arr[i: i + k]


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
    # https://leetcode.com/problems/max-consecutive-ones-iii/discuss/1304883/C%2B%2BPython-3-solutions-Easy-to-understand-Clean-and-Concise
    n, result, l, zeros = len(nums), 0, 0, 0
    for r in range(n):
        if nums[r] == 0:
            zeros += 1
        while zeros > k:
            if nums[l] == 0:
                zeros -= 1
            l += 1
        result = max(result, r - l + 1)
    return result


def remove_all_adjacent_duplicates_in_string_2(s, k):
    stack = []
    for char in s:
        if stack and stack[-1][0] == char:
            stack[-1][1] += 1
        else:
            stack.append([char, 1])
        if stack[-1][1] == k:
            stack.pop()
        return ''.join(char * freq for char, freq in stack)


class MedianFinder:
    def __init__(self):
        self.min_heap, self.max_heap = [], []

    def add_num(self, num):
        heapq.heappush(self.max_heap, -num)
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def find_median(self):
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2


def shortest_distance_from_all_buildings(grid):
    # https://blog.csdn.net/qq_37821701/article/details/108906696
    # https://github.com/ShiqinHuo/LeetCode-Python/blob/master/Python/shortest-distance-from-all-buildings.py
    # Need to verify this solution
    m, n = len(grid), len(grid[0])
    no_buildings, result = 0, float('inf')

    def bfs(x, y):
        visited, buildings, current_steps = set(), set(), 0
        queue = collections.deque([(x, y, 0)])
        visited.add((i, j))
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        while queue:
            x, y, step = queue.popleft()
            if grid[x][y] == 1 and (i, j) not in buildings:
                current_steps += step
                buildings.add((i, j))
            if len(buildings) == no_buildings:
                break
            if grid[x][y] != 1:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and (x, y) not in visited and grid[nx][ny] != 2:
                        queue.append((nx, ny, step + 1))
                        visited.add((nx, ny))
            return current_steps if len(buildings) == no_buildings else -1

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                no_buildings += 1

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0:
                current_steps = bfs(i, j)
                if current_steps != -1 and current_steps < result:
                    result = current_steps
    return result if result != float('inf') else -1


def beast_time_to_buy_and_sell1(prices):
    current_max, result, n = 0, 0, len(prices)
    for i in range(1, n):
        current_max += prices[i] - prices[i - 1]
        if current_max < 0:
            current_max = 0
        result = max(current_max, result)
    return result


def beast_time_to_buy_and_sell2(prices):
    current_max, result, n = 0, 0, len(prices)
    for i in range(1, n):
        current_max += prices[i] - prices[i - 1]
        current_max = max(current_max, 0)
        result = max(current_max, result)
    return result


def daily_temperatures(temperatures):
    # https://leetcode.com/problems/daily-temperatures/discuss/1574808/C%2B%2BPython-3-Simple-Solutions-w-Explanation-Examples-and-Images-or-2-Monotonic-Stack-Approaches
    result, stack = [0] * len(temperatures), []
    for i, temp in enumerate(temperatures):
        while stack and temp > temperatures[stack[-1]]:
            result[stack[-1]] = i - stack[-1]
            stack.pop()
        stack.append(i)
    return result


def valid_parentheses(s):
    if len(s) % 2 == 0:
        return False
    hashmap, stack = {'(': ')', '[': ']', '{': '}'}, []
    for char in s:
        if char in hashmap:
            stack.append(char)
        else:
            if stack and hashmap[stack[-1]] == char:
                stack.pop()
            else:
                return False
    return len(stack) == 0
