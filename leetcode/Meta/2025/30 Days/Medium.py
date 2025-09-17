import bisect
import random
import heapq
from collections import OrderedDict, defaultdict, deque, Counter
from leetcode.LCMetaPractice import ListNode, TreeNode, RandomPointerNode


def simplify_path(path):
    # https://leetcode.com/problems/simplify-path
    stack, components = [], path.split('/')
    for component in components:
        if component in ['.', '']:
            continue
        elif component == '..':
            if stack:
                stack.pop()
        else:
            stack.append(component)
    return '/' + '/'.join(stack)


class LRUCache1:

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}

    def get(self, key):
        if key not in self.cache:
            return -1
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            self.cache.pop(next(iter(self.cache)))  # For ordered dict use self.cache.popitem(last=False)
            # oldest_key = next(iter(self.cache))
        self.cache[key] = value


class LRUCache2:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = []  # list of (key, value) pairs

    def get(self, key: int) -> int:
        # Search linearly (O(n))
        for i, (k, v) in enumerate(self.cache):
            if k == key:
                # Move to most recently used (end of list)
                self.cache.append(self.cache.pop(i))
                return v
        return -1  # not found

    def put(self, key: int, value: int) -> None:
        # Check if key already exists
        for i, (k, v) in enumerate(self.cache):
            if k == key:
                # Update value and move to MRU
                self.cache.pop(i)
                self.cache.append((key, value))
                return

        # If at capacity, evict least recently used (front of list)
        if len(self.cache) == self.capacity:
            self.cache.pop(0)

        # Insert new key-value at MRU position
        self.cache.append((key, value))


class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache3:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node
        self.head = Node(0, 0)  # dummy head
        self.tail = Node(0, 0)  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        prev, nxt = node.prev, node.next
        prev.next, nxt.prev = nxt, prev

    def _add(self, node):
        prev, nxt = self.tail.prev, self.tail
        prev.next = nxt.prev = node
        node.prev, node.next = prev, nxt

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self._add(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            # remove from the front
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]


def maximum_swap(num):
    # https://leetcode.com/problems/maximum-swap/discuss/846837/Python-3-or-Greedy-Math-or-Explanations
    s = list(str(num))  # num = 2736, result = 7236
    n = len(s)

    # Track the rightmost max digit and positions to swap
    max_idx = n - 1
    left, right = -1, -1  # left tracks the smallest and right tracks the largest

    # Traverse from right to left
    for i in range(n - 2, -1, -1):
        if s[i] > s[max_idx]:
            max_idx = i
        elif s[i] < s[max_idx]:
            left, right = i, max_idx

    if left == -1:  # already largest
        return num

    # Swap and return
    s[left], s[right] = s[right], s[left]
    return int("".join(s))


def find_peak_element_1(nums):
    # This might work but it is O(n)
    n = len(nums)
    if n == 1 or nums[0] >= nums[1]:
        return 0
    if nums[n - 1] >= nums[n - 2]:
        return n - 1

    for i in range(1, n - 1):
        if nums[i - 1] < nums[i] > nums[i + 1]:
            return i
    return -1


def find_peak_element_2(nums):
    # https://leetcode.com/problems/find-peak-element
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid + 1] < nums[mid]:
            right = mid
        else:
            left = mid + 1
    return left


class SparseVector1:
    # https://zhenchaogan.gitbook.io/leetcode-solution/leetcode-1570-dot-product-of-two-sparse-vectors
    def __init__(self, nums):
        self.hashmap = {i: val for i, val in enumerate(nums) if val}

    def dot_product(self, vec):
        if len(self.hashmap) > len(vec.hashmap):
            self, vec = vec, self
        return sum(val * vec.hashmap[key] for key, val in self.hashmap.items() if key in vec.hashmap)


class SparseVector2:
    # This is what I used in my last interview
    def __init__(self, nums):
        self.linked_list = [[i, val] for i, val in enumerate(nums) if val]

    def dot_product(self, vec):
        if len(self.linked_list) > len(vec.linked_list):
            self, vec = vec, self
        result = i = j = 0
        n1, n2 = len(self.linked_list), len(vec.linked_list)
        while i < n1 and j < n2:
            if self.linked_list[i][0] == vec.linked_list[j][0]:
                result += self.linked_list[i][1] * vec.linked_list[j][1]
                i, j = i + 1, j + 1
            elif self.linked_list[i][0] < vec.linked_list[j][0]:
                i += 1
            else:
                j += 1
        return result


def kth_largest_number_in_an_array_1(nums, k):
    # https://leetcode.com/problems/kth-largest-element-in-an-array
    """
    O(n log n)
    """
    return sorted(nums)[-k]


def kth_largest_number_in_an_array_2(nums, k):
    """
    O(n log k)
    """
    heap = []
    for num in nums:
        heapq.heappush(heap, num)  # O(log k)
        if len(heap) > k:
            heapq.heappop(heap)  # removes the smaller elements
    return heap[0]  # or heapq.heappop(heap)


def kth_largest_number_in_an_array_3(nums, k):  # TLE
    """
    Average time: O(n)
    Worst case: O(n^2) (though rare due to random pivot).
    """
    n = len(nums)
    k = n - k   # Convert kth largest to kth smallest index
                # Example: [3,2,1,5,6,4], k=2 → we want index n-k=4 (value=5)

    def swap (i, j):
        nums[i], nums[j] = nums[j], nums[i]

    def partition(left, right, p_index):
        pivot = nums[p_index]     # pick pivot
        swap(p_index, right)      # move pivot to the end temporarily
        p_index = left            # new pivot index starts at left
        for i in range(left, right):
            if nums[i] <= pivot:  # move smaller elements to left
                swap(i, p_index)
                p_index += 1
        swap(p_index, right)      # place pivot back in its correct position
        return p_index            # return final pivot index


    def quick_select(left, right):
        p_index = random.randint(left, right)       # random pivot
        p_index = partition(left, right, p_index)

        if k == p_index:                            # found the kth element
            return
        elif k < p_index:                           # search left side
            quick_select(left, p_index - 1)
        else:                                       # search right side
            quick_select(p_index + 1, right)

    quick_select(0, n - 1)
    return nums[k]                                  # kth smallest → kth largest


def kth_largest_number_in_an_array_4(nums, k):
    # Convert kth largest → kth smallest index
    # Example: for nums = [3,2,1,5,6,4], k = 2 (2nd largest),
    #          len(nums) - k = 6 - 2 = 4 → 4th smallest (0-based index)
    k = len(nums) - k

    def quick_select(left, right):
        # Choose pivot as the rightmost element
        pivot = nums[right]
        low, high = left, right

        # Partition the array around the pivot
        while low <= high:
            # Move low forward while elements are smaller than pivot
            while low <= high and nums[low] < pivot:
                low += 1
            # Move high backward while elements are larger than pivot
            while low <= high and nums[high] > pivot:
                high -= 1

            # Swap out-of-place elements
            if low <= high:
                nums[low], nums[high] = nums[high], nums[low]
                low += 1
                high -= 1

        # After partition:
        # - All elements on [left .. high] ≤ pivot
        # - All elements on [low .. right] ≥ pivot

        # Decide which side to recurse on:
        if k <= high:            # kth smallest lies in the left partition
            return quick_select(left, high)
        elif k >= low:           # kth smallest lies in the right partition
            return quick_select(low, right)
        else:                    # kth smallest is exactly in between
            return nums[k]

    # Perform quickselect on the entire range
    return quick_select(0, len(nums) - 1)


def lowest_common_ancestor(root, p, q):
    # https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/
    child_parent_dict, p_ancestors = dict(), set()

    def dfs(child, parent):
        if child:
            child_parent_dict[child] = parent
            dfs(child.left, child)
            dfs(child.right, child)

    dfs(root, None)

    while p:
        p_ancestors.add(p)
        p = child_parent_dict[p]

    while q not in p_ancestors:
        q = child_parent_dict[q]

    return q


def lowest_common_ancestor_of_binary_tree_iii(root, p, q):
    # https://zhenchaogan.gitbook.io/leetcode-solution/leetcode-1650-lowest-common-ancestor-of-a-binary-tree-iii
    # Base case: if root is None, or root is one of the targets
    if root is None or root == p or root == q:
        return root
    # We search both subtrees.
    # l will be either None (not found) or the node (p or q or an ancestor).
    # r behaves the same.
    l = lowest_common_ancestor_of_binary_tree_iii(root.left, p, q)
    r = lowest_common_ancestor_of_binary_tree_iii(root.right, p, q)
    # If both sides return something, it means:
    # p is in one branch
    # q is in the other branch
    # → so root is their lowest common ancestor.
    if l and r:
        return root
    return l or r


def binary_tree_vertical_order_traversal_1(root):
    # https://algo.monster/liteproblems/314
    if root is None:
        return
    result = []
    queue, column_dict = deque([(root, 0)]), defaultdict(list)
    while queue:
        current, column = queue.popleft()
        column_dict[column].append(current.val)
        if current.left:
            queue.append((current.left, column - 1))
        if current.right:
            queue.append((current.right, column + 1))
    for key in sorted(column_dict.keys()):
        result.append(column_dict[key])


def binary_tree_vertical_order_traversal_2(root):
    if root is None:
        return
    result = []
    column_dict = defaultdict(list)

    def dfs(node, depth, column):
        if not node:
            return
        column_dict[column].append((depth, node.val))
        dfs(node.left, depth + 1, column - 1)
        dfs(node.right, depth + 1, column + 1)

    dfs(root, 0, 0)

    for column in sorted(column_dict.keys()):
        depth_value_tuples = column_dict[column]
        depth_value_tuples.sort(key=lambda x: x[0])
        result.append([val for depth, val in depth_value_tuples])
    return result


def minimum_remove_to_make_valid_parentheses(s):
    s, stack = list(s), []
    for i, char in enumerate(s):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                stack.pop()
            else:
                s[i] = ''
    while stack:
        s[stack.pop()] = ''

    return ''.join(s)


def k_closest_points_to_origin_1(points, k):
    # Time Complexity: O(NlogN)
    # Space Complexity:
    return sorted(points, key=lambda x, y: x * x + y * y)[:k]


def k_closest_points_to_origin_2(points, k):
    # O(nlogk) --> n for loop and logk for push and pop
    # We want to maintain max heap, that's why we use -ve distance
    def euclidean(p, q):
        return p * p + q * q

    heap = []
    for i, (x, y) in enumerate(points):
        distance = euclidean(x, y)
        if len(heap) >= k:
            heapq.heappushpop(heap, (-distance, i))
        else:
            heapq.heappush(heap, (-distance, i))
    return [points[i] for (_, i) in heap]


def continuous_subarray_sum(nums, k):
    """
    If that subarray sum is divisible by k, then:
        (prefix_sum[j] - prefix_sum[i]) % k == 0 → prefix_sum[j] % k == prefix_sum[i] % k
    """
    seen = {0: -1}
    prefix_sum = 0

    for i, num in enumerate(nums):
        prefix_sum += num
        remainder = prefix_sum % k

        if remainder in seen:
            if i - seen[remainder] >= 2:
                return True
        else:
            seen[remainder] = i
    return False


def power_1(x, n):
    # https://leetcode.com/problems/powx-n/discuss/738830/Python-recursive-O(log-n)-solution-explained
    if abs(x) < 1e-40:
        return 0
    if n < 0:
        return power_1(1 / x, -n)
    elif n == 0:
        return 1
    else:
        a = power_1(x, n // 2)
        if n % 2 == 0:
            return a * a
        else:
            return a * a * x


def power_2(x: float, n: int) -> float:
    if n < 0:
        x = 1 / x
        n = -n

    result = 1
    while n > 0:
        if n % 2 == 1:  # if n is odd
            result *= x
        x *= x  # square the base
        n //= 2  # halve the exponent
    return result


def merge_intervals(intervals):
    intervals, result = sorted(intervals, key=lambda x: x[0]), []
    for start, end in intervals:
        if not result or start > result[-1][-1]:
            result.append([start, end])
        else:
            result[-1][-1] = max(result[-1][-1], end)
    return result


def first_and_last_position_of_element_in_sorted_array(nums, target):
    if not nums:
        return [-1, -1]

    n = len(nums)
    result = [-1, -1]

    left, right = 0, n - 1
    while left < right:
        mid = left + (right - left) // 2
        if target <= nums[mid]:
            right = mid
        else:
            left = mid + 1

    if nums[left] != target:
        return result

    result[0] = left

    left, right = 0, n - 1
    while left < right:
        mid = left + (right - left + 1) // 2
        if target < nums[mid]:
            right = mid + 1
        else:
            left = mid

    result[1] = right
    return result


def binary_tree_right_side_view_1(root):
    queue, result = deque([root]), []
    if not root:
        return result
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


def binary_tree_right_side_view_2(root):
    result, visited = [], set()
    if not root:
        return result

    def dfs(node, level):
        if node:
            if level not in visited:
                visited.add(level)
                result.append(node.val)
            dfs(node.right,level + 1)
            dfs(node.right,level + 1)

    dfs(root, 0)
    return result


def binary_tree_right_side_view_3(root):
    queue, result = deque([root]), []

    if not root:
        return result

    while queue:
        current_size = len(queue)
        while current_size > 0:
            current = queue.popleft()
            current_size -= 1
            if current_size == 0:
                result.append(current.val)

            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)

    return result


def shortest_path_in_binary_matrix(grid):
    # https://leetcode.com/problems/shortest-path-in-binary-matrix
    # This of shortest path in an unweighted graph -- classic BFS
    n = len(grid)
    if grid[0][0] == 1 or grid[n - 1][n - 1] == 1:
        return -1

    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    visited = set([(0, 0)])

    while queue:
        x, y, distance = queue.popleft()
        if (x, y) == (n - 1, n - 1):
            return distance

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, distance + 1))

    return -1


def basic_calculator_ii(s):
    # https://leetcode.com/problems/basic-calculator-ii/discuss/658480/Python-Basic-Calculator-I-II-III-easy-solution-detailed-explanation
    def update_stack(op, value):
        if op == '+': stack.append(value)
        if op == '-': stack.append(-value)
        if op == '*': stack.append(stack.pop() * value)
        if op == '/': stack.append(int(stack.pop() / value))

    i, num, stack, sign = 0, 0, [], '+'

    while i < len(s):
        if s[i].isdigit():
            num = num * 10 + int(s[i])
        elif s[i] in '+-*/':
            update_stack(sign, num)
            num, sign = 0, s[i]
        elif s[i] == '(':
            num, j = basic_calculator_ii(s[i + 1:])
            i = i + j
        elif s[i] == ')':
            update_stack(sign, num)
            return sum(stack), i + 1
        i += 1

    update_stack(sign, num)
    return sum(stack)  # Not really working for basic calculator iii


def subarray_sum_equals_k(nums, k):
    # In this all are positive numbers only
    # In comments of https://leetcode.com/problems/subarray-sum-equals-k/discuss/102111/Python-Simple-with-Explanation
    prefix_sum, prefix_sum_counts, result = 0, defaultdict(int), 0
    prefix_sum_counts[0] = 1

    for num in nums:
        prefix_sum += num
        diff = prefix_sum - k

        if diff in prefix_sum_counts:
            result += prefix_sum_counts[diff]

        prefix_sum_counts[prefix_sum] += 1

    return result


def nested_list_weighted_sum_1(nested_list):
    # https://leetcode.ca/all/339.html
    # https://jzleetcode.github.io/posts/leet-0339-lint-0551-nested-list-weight-sum/
    result, stack = 0, [(nested_list, 1)]
    while stack:
        elements, depth = stack.pop()
        for element in elements:
            if element.isInteger():
                result += element.getInteger() * depth
            else:
                stack.append((element.getList(), depth + 1))
    return result


def nested_list_weighted_sum_2(nested_list):

    def dfs(nl, depth):
        result = 0
        for element in nl:
            if element.isInteger():
                result += element.getInteger() * depth
            else:
                result += dfs(element.getList(), depth + 1)
        return result

    return dfs(nested_list, 1)


def top_k_frequent_elements_1(nums, k):
    """
    Time: O(n + m log k)
    Space: O(m + k)
    """
    counts = Counter(nums)                # O(n)
    heap = []
    for num, freq in counts.items():      # O(m log k) where m = unique elements
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    return [num for _, num in heap]


def top_k_frequent_elements_2(nums, k):
    """
    Time: O(n + m)
    Space: O(n + m)
    """
    counts, freq_dict, result = Counter(nums), defaultdict(list), []

    for num, freq in counts.items():
        freq_dict[freq].append(num)

    for freq in reversed(range(len(nums) + 1)):
        result.extend(freq_dict[freq])
        if len(result) >= k:
            return result[:k]
    return result[:k]


def top_k_frequent_elements_3(nums, k):
    """
    Time: O(n log m)
    Space: O(m)
    """
    counts = Counter(nums)                # O(n)
    return [num for num, _ in counts.most_common(k)]


def custom_sort_string(order, s):
    # https://leetcode.com/problems/custom-sort-string
    result, counter = [], Counter(s)
    for char in order:
        if char in counter:
            result.append(char * counter[char])
            del counter[char]
    for char, freq in counter.items():
        result.append(char * freq)

    return "".join(result)


def diagonal_traverse(mat):
    # https://leetcode.com/problems/diagonal-traverse/description/
    # Use dict and sum of i+j is same for elements on the diagonal
    m, n, result = len(mat), len(mat[0]), []
    diagonal_map = defaultdict(list)

    for i in range(m):
        for j in range(n):
            diagonal_map[i + j].append(mat[i][j])
            # use i - j for collecting elements on the other diagonal

    for key in sorted(diagonal_map):
        if key % 2 == 0:
            result.extend(diagonal_map[key][::-1])
        else:
            result.extend(diagonal_map[key])
    return result
    # itertools.chain(*[v if k % 2 else v[::-1] for k, v in d.items()])


def copy_list_with_random_pointer(head):
    # https://leetcode.com/problems/copy-list-with-random-pointer
    pass


class RandomPickWeight:

    def __init__(self, w):
        self.w = w
        self.total = sum(self.w)
        self.n = len(self.w)

        # 1. Normalize the weights. Now self.w contains probabilities that sum up to 1.
        # Example: w = [1,3,2] → self.w = [1/6, 3/6, 2/6] = [0.166, 0.5, 0.333].
        for i in range(self.n):
            self.w[i] = self.w[i] / self.total

        # 2. Convert to Cumulative Distribution Function (CDF) → self.w = [0.166, 0.666, 1.0].
        # This means:
        #   Index 0 covers [0, 0.166]
        #   Index 1 covers (0.166, 0.666]
        #   Index 2 covers (0.666, 1.0]
        for i in range(1, self.n):
            self.w[i] += self.w[i - 1]

    def pick_index(self):
        # k = random.random()
        # return bisect.bisect_left(self.w, k)

        # or
        # 3. Pick a random number from uniform distribution between 0, 1
        k = random.uniform(0, 1)  # returns a floating point number
        l, r = 0, len(self.w)
        while l < r:
            mid = l + (r - l) // 2
            if k <= self.w[mid]:
                r = mid
            else:
                l = mid + 1
        return l


def next_permutation(nums):
    # https://leetcode.com/problems/next-permutation/description/
    n = len(nums)
    pivot = n - 2

    # 1. Find first decreasing element from right
    while pivot >= 0 and nums[pivot] >= nums[pivot + 1]:
        pivot -= 1

    if pivot >= 0:  # found a pivot
        j = n - 1
        # 2. Find next greater element
        while nums[pivot] >= nums[j]:
            j -= 1
        # 3. Swap
        nums[pivot], nums[j] = nums[j], nums[pivot]

    # 4. Reverse suffix
    nums[pivot + 1:] = reversed(nums[pivot + 1:])


def buildings_with_an_ocean_view_1(heights):
    # https://goodtecher.com/leetcode-1762-buildings-with-an-ocean-view/
    result, n = [], len(heights)
    for i in range(n - 1, -1, -1):
        if not result:
            result.append(i)
        if heights[i] > heights[result[-1]]:
            result.append(i)
    return result[::-1]


def buildings_with_an_ocean_view_2(heights):
    # Using monotonic decreasing stack
    stack = []
    for i, height in enumerate(heights):
        while stack and heights[stack[-1]] <= height:
            stack.pop()
        stack.append(i)
    return stack


def buildings_with_an_ocean_view_3(heights):
    # Solution if ocean is to the left of the buildings.
    stack = []
    for i, height in enumerate(heights):
        if not stack or heights[stack[-1]] < height:
            stack.append(i)
    return stack


def group_shifted_strings(strings):
    # https://baihuqian.github.io/2018-07-26-group-shifted-strings/
    # https://techyield.blogspot.com/2020/10/group-shifted-strings-python-solution.html
    if len(strings) == 0:
        return []
    groups = defaultdict(list)
    for word in strings:
        key = []
        for i in range(len(word) - 1):
            key.append((26 + ord(word[i + 1]) - ord(word[i])) % 26)
        groups[tuple(key)].append(word)
    return groups.values()


def course_schedule(num_courses, prerequisites):
    # https://leetcode.com/problems/course-schedule/description/
    # Index of the graph and in_degree are also important
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses
    for curr, prev in prerequisites:
        graph[prev].append(curr)
        in_degree[curr] += 1

    queue = deque([v for v in range(num_courses) if in_degree[v] == 0])
    taken = 0

    while queue:
        curr_course = queue.popleft()
        taken += 1
        for next_course in graph[curr_course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return taken == num_courses


def insert_into_a_sorted_circular_linked_list(head, insert_val):
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
        if current > current.next.val:
            if insert_val >= current.val or insert_val <= current.next.val:
                break
        # Move forward
        current = current.next
        # Case 3: Completed full cycle, all values are same
        if current == head:
            break

    # Insert new_node between current and current.next
    new_node.next = current.next
    current.next = new_node
    return head


def sum_root_to_leaf_numbers_1(root):
    # https://leetcode.com/problems/sum-root-to-leaf-numbers/discuss/1556417/C%2B%2BPython-Recursive-and-Iterative-DFS-%2B-BFS-%2B-Morris-Traversal-O(1)-or-Beats-100

    def dfs(node, value):
        if not node:
            return 0
        value = value * 10 + node.val
        if not node.left and not node.right:
            return value
        return dfs(node.left, value) + dfs(node.right, value)

    return dfs(root, 0)


def sum_root_to_leaf_numbers_2(root):
    stack, result = [(root, 0)], 0
    while stack:
        current, value = stack.pop()
        value = value * 10 + current.val
        if not current.left and not current.right:
            result += value
        if current.left:
            stack.append((current.left, value))
        if current.right:
            stack.append((current.right, value))
    return result
