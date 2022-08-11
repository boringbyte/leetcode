import heapq
import random
import collections
from queue import PriorityQueue
from functools import lru_cache
from LC.LCMetaPractice import TreeNode, ListNode, DLLNode


class GraphNode:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def verify_alien_dictionary(words, order):
    order_map = {char: i for i, char in enumerate(order)}

    def check_order(word1, word2):
        for char1, char2 in zip(word1, word2):
            if char1 != char2:
                return order_map[char1] < order_map[char2]
        return len(word1) <= len(word2)

    return all(check_order(word1, word2) for word1, word2 in zip(words, words[1:]))


def minimum_remove_to_make_valid_parentheses1(s):
    in_valid, stack = set(), []
    for i, char in stack:
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                in_valid.add(i)
    while stack:
        in_valid.add(stack.pop())
    return ''.join(char for i, char in enumerate(s) if i not in in_valid)


def minimum_remove_to_make_valid_parentheses2(s):
    s, stack = list(s), []
    for i, char in enumerate(s):
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


def k_closest_points_to_origin1(points, k):
    # O(nlogn)
    return sorted(points, key=lambda x, y: x * x + y * y)[:k]


def k_closest_points_to_origin2(points, k):
    # O(nlogk) --> n for loop and logk for push and pop
    # We want to maintain max heap, that's why we use -ve distance
    heap, euclidean = [], lambda x, y: x * x + y * y
    for i, (x, y) in enumerate(points):
        distance = euclidean(x, y)
        if len(heap) > k:
            heapq.heappushpop(heap, (-distance, i))
        else:
            heapq.heappush(heap, (-distance, i))
    return [points[i] for (_, i) in heap]


def k_closest_points_to_origin3(points, k):
    # O(n) because of quick select algorithm. This is mainly used to find k min or max in an unsorted array
    # During quick select algorithm, our search space becomes half on average for each loop.
    """
    Quick select algorithm
        1. Set the left and right and pivot index to 0, n-1 and n as initial values
        2. Loop till pivot index not equals to k
        3. During the loop process, get new pivot index using partition function
        4. Now update left to pivot_index + 1 if pivot is less than k
        5. Now update right to pivot_index - 1 if pivot or else.

    Partition logic
        1. Find the pivot index between left and right mid point or some random point
        2. Set i and pivot_distance to left and euclidean in this instance
        3. Swap right and pivot index with each other
        4. Loop from left to right + 1
        5.      If euclidean distance of j is less than pivot_distance
        6.      Swap the i and j elements
        7.      Update i to i + 1
        8. Finally return i - 1
    """

    def euclidean(point):
        x, y = point
        return x * x + y * y

    def partition(left, right):
        p_index = left + (right - left) // 2
        i, pivot_distance = left, euclidean(points[p_index])
        points[right], points[p_index] = points[p_index], points[right]
        for j in range(left, right + 1):
            if euclidean(points[j]) < pivot_distance:
                points[i], points[j] = points[j], points[i]
                i += 1
        return i - 1

    L, R, pivot_index = 0, len(points) - 1, len(points)
    while pivot_index != k:
        pivot_index = partition(L, R)
        if pivot_index < k:
            L = pivot_index + 1
        else:
            R = pivot_index - 1
    return points[:k]


def k_closest_points_to_origin4(points, k):
    n = len(points)

    def euclidean(point):
        x, y = point
        return x * x + y * y

    def swap(i, j):
        points[i], points[j] = points[j], points[i]

    def partition(left, right, pivot_index):
        pivot = euclidean(points[pivot_index])
        swap(pivot_index, right)
        pivot_index = left
        for i in range(left, right):
            if euclidean(points[i]) <= pivot:
                swap(i, pivot_index)
                pivot_index += 1
        swap(pivot_index, right)
        return pivot_index

    def quick_select(left, right):
        if left < right:
            pivot_index = random.randint(left, right)
            pivot_index = partition(left, right, pivot_index)
            if k == pivot_index:
                return
            if k < pivot_index:
                quick_select(left, pivot_index - 1)
            else:
                quick_select(pivot_index + 1, right)

    quick_select(0, n - 1)
    return points[:k]


def product_of_array_except_self(nums):
    n = len(nums)
    prefix, suffix = [1] * n, [1] * n

    for i in range(1, n):
        prefix[i] = prefix[i - 1] * nums[i - 1]

    for i in range(n - 2, -1, -1):
        suffix[i] = suffix[i + 1] * nums[i + 1]

    return [p * f for p, f in zip(prefix, suffix)]


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
    # In this all are positive numbers only
    # In comments of https://leetcode.com/problems/subarray-sum-equals-k/discuss/102111/Python-Simple-with-Explanation
    total = result = 0
    cumulative_sum = [total := total + num for num in nums]
    count = collections.Counter({0: 1})
    for acc in cumulative_sum:
        result += count[acc - k]
        count[acc] += 1
    return result


class BinaryMatrix:
    def get(self, row, col):
        pass

    def dimensions(self):
        pass


def leftmost_column_with_at_least_a_one1(binary_matrix):
    m, n = binary_matrix.dimensions()
    result = float('inf')
    for i in range(m):
        l, r = 0, n
        while l < r:
            mid = l + (r - l) // 2
            if binary_matrix.get(i, mid) == 0:
                l = mid + 1
            else:
                r = mid
        if l < n and binary_matrix.get(i, l) == 1:
            result = min(result, l)
    return result if result < float('inf') else -1


def leftmost_column_with_at_least_a_one2(binary_matrix):
    # https://medium.com/@srihari.athiyarath/leftmost-column-with-at-least-a-one-24184d8f4052
    m, n = binary_matrix.dimensions()
    r, c = 0, n - 1
    while r < m and c >= 0:
        if binary_matrix.get(r, c) == 0:
            r += 1
        else:
            c -= 1
    return c + 1 if c + 1 != n else -1


def add_strings(num1, num2):
    def string_to_digit(d):
        return ord(d) - ord('0')

    i, j, carry, result = len(num1) - 1, len(num2) - 1, 0, []
    while i >= 0 or j >= 0 or carry:
        digit1 = string_to_digit(num1[i]) if i >= 0 else 0
        digit2 = string_to_digit(num2[j]) if j >= 0 else 0
        carry, digit = divmod(digit1 + digit2 + carry, 10)
        result.append(str(digit))
        i, j = i - 1, j - 1
    return ''.join(result[::-1])


def merge_intervals1(intervals):
    intervals, result = sorted(intervals, key=lambda x: x[0]), []
    for interval in intervals:
        if not result or result[-1][-1] < interval[0]:
            result.append(interval)
        else:
            result[-1][-1] = max(result[-1][-1], interval[-1])
    return result


def merge_intervals2(intervals):
    intervals, result = sorted(intervals, key=lambda x: x[0]), []
    for start, end in intervals:
        if not result or result[-1][-1] < start:
            result.append([start, end])
        else:
            result[-1][-1] = max(result[-1][-1], end)
    return result


def add_binary(a, b):
    digit_map = {'0': 0, '1': 1}
    i, j, carry, result = len(a) - 1, len(b) - 1, 0, []
    while i >= 0 or j >= 0 or carry:
        digit1 = digit_map[a[i]] if i >= 0 else 0
        digit2 = digit_map[b[j]] if j >= 0 else 0
        carry, digit = divmod(digit1 + digit2 + carry, 2)
        result.append(digit)
        i, j = i - 1, j - 1
    return ''.join(result[::-1])


def binary_tree_maximum_path_sum(root):
    result = [float('-inf')]

    def dfs(node):
        if not node:
            return 0
        l, r = dfs(node.left), dfs(node.right)
        result[0] = max(result[0], l + r + node.val)
        return max(l + node.val, r + node.val, 0)

    dfs(root)
    return result[0]


def valid_palindrome(s):
    l, r, = 0, len(s) - 1
    while l < r:
        if not s[l].isalnum():
            l += 1
        elif not s[r].isalnum():
            r -= 1
        else:
            if s[l] != s[r]:
                return False
            else:
                l, r = l + 1, r - 1
    return True


def k_th_smallest_element_in_an_array(nums, k):
    # https://www.techiedelight.com/quickselect-algorithm/
    n = len(nums)
    k = k - 1

    def swap(i, j):
        nums[i], nums[j] = nums[j], nums[i]

    def partition(left, right, p_index):
        pivot = nums[p_index]
        swap(p_index, right)
        p_index = left
        for i in range(left, right):
            if nums[i] <= pivot:
                swap(i, p_index)
                p_index += 1
        swap(p_index, right)
        return p_index

    def quick_select(left, right):
        if left == right:
            return nums[left]
        p_index = random.randint(left, right)
        p_index = partition(left, right, p_index)
        if k == p_index:
            return nums[k]
        elif k < p_index:
            return quick_select(left, p_index - 1)
        else:
            return quick_select(p_index + 1, right)
    return quick_select(0, n - 1)


def k_th_largest_element_in_an_array(nums, k):
    n = len(nums)
    k = n - k

    def swap(i, j):
        nums[i], nums[j] = nums[j], nums[i]

    def partition(left, right, p_index):
        pivot = nums[p_index]
        swap(p_index, right)
        p_index = left
        for i in range(left, right):
            if nums[i] <= pivot:
                swap(i, p_index)
        swap(p_index, right)
        return p_index

    def quick_select(left, right):
        if left == right:
            return nums[left]
        p_index = random.randint(left, right)
        p_index = partition(left, right, p_index)
        if k == p_index:
            return nums[k]
        elif k < p_index:
            return quick_select(left, p_index - 1)
        else:
            return quick_select(p_index + 1, right)

    return quick_select(0, n-1)


class SparseVector:
    def __init__(self, nums):
        self.hashmap = {i: val for i, val in enumerate(nums) if val}

    def dot_product(self, vec):
        if len(self.hashmap) > len(vec.hashmap):
            self, vec = vec, self
        return sum(val * vec.hashmap[key] for key, val in self.hashmap.items() for key in vec.hashmap)


def range_sum_of_binary_search_tree(root, low, high):
    result = [0]

    def dfs(node):
        if node:
            if low <= node.val <= high:
                result[0] += node.val
            if low < node.val:
                dfs(node.left)
            if high > node.val:
                dfs(node.right)

    dfs(root)
    return result[0]


def binary_tree_right_side_view1(root):
    result, queue = [], collections.deque([root])
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


def binary_tree_right_side_view2(root):
    result, visited = [], set()
    if not root:
        return result

    def dfs(node, level):
        if node:
            if level not in visited:
                visited.add(level)
                result.append(node.val)
            dfs(node.right, level + 1)
            dfs(node.left, level + 1)

    dfs(root, 0)
    return result


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
                    if dfs(child, i + 1):
                        return True
            if word[i] in node.children:
                child = node.children[word[i]]
                return dfs(child, i + 1)
            return False

        return dfs(self.root, 0)


def trapping_rain_water1(heights):
    n, result = len(heights), 0
    max_left, max_right = [0] * n, [0] * n

    for i in range(1, n):
        max_left[i] = max(max_left[i - 1], heights[i - 1])

    for i in range(n - 2, -1, -1):
        max_right[i] = max(max_right[i + 1], heights[i + 1])

    for i in range(n):
        water_level = min(max_left[i], max_right[i])
        if water_level >= heights[i]:
            result += (water_level - heights[i])
    return result


def trapping_rain_water2(heights):
    max_left, max_right, result = heights[0], heights[-1], 0
    l, r = 1, len(heights) - 2
    while l <= r:
        max_left = max(max_left, heights[l])
        max_right = max(max_right, heights[r])
        if max_left < max_right:
            result += (max_left - heights[l])
            l += 1
        else:
            result += (max_right - heights[r])
            r += 1
    return result


def merge_sorted_array(nums1, nums2, m, n):
    while m > 0 and n > 0:
        if nums1[m - 1] >= nums2[n - 1]:
            nums1[m + n - 1] = nums1[m - 1]
            m -= 1
        else:
            nums1[m + n - 1] = nums1[n - 1]
            n -= 1
    if n > 0:
        nums1[:n] = nums2[:n]


def first_bad_version(n):
    def is_bad_version(x):
        return x == 0

    l, r = 1, n
    while l < r:
        mid = l + (r - l) // 2
        if is_bad_version(mid):
            r = mid
        else:
            l = mid + 1
    return l


def diameter_of_binary_tree(root):
    result = [0]

    def dfs(node):
        if not node:
            return 0
        l, r = dfs(node.left), dfs(node.right)
        result[0] = max(result[0], l + r)
        return max(l, r) + 1

    dfs(root)
    return result[0]


class RandomPickWithWeight:
    def __init__(self, w):
        self.w, total, total_weight = w, 0, sum(w)
        cumulative_sum = [total := total + weight for weight in self.w]
        self.w = [weight / total_weight for weight in cumulative_sum]

    def pick_index(self):
        r, index = random.uniform(0, 1), 0
        while index < len(self.w):
            if r <= self.w[index]:
                return index
            index += 1
        # return bisect.bisect_left(self.w, r)


def lowest_common_ancestor_of_binary_tree1(root, p, q):
    if root is None or root == p or root == q:
        return root
    l = lowest_common_ancestor_of_binary_tree1(root.left, p, q)
    r = lowest_common_ancestor_of_binary_tree1(root.right, p, q)
    if l and r:
        return root
    return l or r


def lowest_common_ancestor_of_binary_tree2(root, p, q):
    parent_dict, ancestors = dict(), set()

    def dfs(child, parent):
        if child:
            parent_dict[child] = parent
            dfs(child.left, child)
            dfs(child.right, child)

    dfs(root, None)

    while p:
        ancestors.add(p)
        p = parent_dict[p]

    while q not in ancestors:
        q = parent_dict[q]
    return q


def lowest_common_ancestor_of_binary_tree3(root, p, q):
    parent_dict, ancestors, stack = {root: None}, set(), [root]

    while stack:
        child = stack.pop()
        if child.left:
            stack.append(child.left)
            parent_dict[child.left] = child
        if child.right:
            stack.append(child.right)
            parent_dict[child.right] = child

    while p:
        ancestors.add(p)
        p = parent_dict[p]

    while q not in ancestors:
        q = parent_dict[q]
    return q


class Codec:
    i = 0

    @staticmethod
    def serialize(root):
        values = []

        def dfs(node):
            if node:
                values.append(str(node.val))
                dfs(node.left)
                dfs(node.right)
            else:
                values.append('#')
        dfs(root)
        return ' '.join(values)

    def deserialize(self, data):
        values = data.split()

        def dfs():
            if self.i == len(values) or values[self.i] == '#':
                return None
            node = TreeNode(int(values[self.i]))
            self.i += 1
            node.left = dfs()
            node.right = dfs()
            return node
        return dfs()


class BinarySearchTreeIterator:
    def __init__(self, root):
        self.stack = []
        self.push_left(root)

    def push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self):
        node = self.stack.pop()
        self.push_left(node.right)
        return node.val

    def has_next(self):
        return len(self.stack) > 0


def alien_dictionary(words):
    # https://medium.com/@timothyhuang514/graph-alien-dictionary-d2b104c36d8e
    in_degree, result = {char: 0 for word in words for char in word}, ''

    graph = collections.defaultdict(set)
    for word1, word2 in zip(words, words[1:]):
        for char1, char2 in zip(word1, word2):
            if char1 != char2:
                if char2 not in graph[char1]:
                    graph[char1].add(char2)
                    in_degree[char2] += 1
                break
        else:
            if len(word2) < len(word1):
                return ''

    no_incoming_edges_queue = collections.deque([ch for ch in in_degree if in_degree[ch] == 0])

    while no_incoming_edges_queue:
        vertex = no_incoming_edges_queue.popleft()
        result += vertex
        for char in graph[vertex]:
            in_degree[char] -= 1
            if in_degree[char] == 0:
                no_incoming_edges_queue.append(char)

    if len(result) < len(in_degree):
        return ''

    return result


def word_break1(s, word_dict):
    word_dict, n = set(word_dict), len(s)

    @lru_cache
    def dfs(start):
        if start == n:
            return True
        for end in range(start + 1, n + 1):
            sub_string = s[start: end]
            if sub_string in word_dict and dfs(end):
                return True
        return False
    return dfs(0)


def word_break2(s, word_dict):
    queue, visited, n, word_dict = collections.deque([0]), set(), len(s), set(word_dict)
    visited.add(0)
    while len(queue) > 0:
        current = queue.popleft()
        for i in range(current + 1, n + 1):
            if i in visited:
                continue
            if s[current: i] in word_dict:
                if i == n:
                    return True
                queue.append(i)
                visited.add(i)
    return False


def word_break3(s, word_dict):
    word_dict, n = set(word_dict), len(s)
    dp = [False] * (n + 1)
    dp[n] = True

    for i in range(n-1, -1, -1):
        for j in range(i + 1, n + 1):
            if dp[j] and s[i: j] in word_dict:
                dp[i] = True
                break
    return dp[0]


def word_break4(s, word_dict):
    queue, visited = collections.deque([s]), set()
    while queue:
        s = queue.popleft()
        for word in word_dict:
            if s.startswith(word):
                new_s = s[len(word):]
                if new_s == "":
                    return True
                if new_s not in visited:
                    queue.append(new_s)
                    visited.add(new_s)
    return False


def bisect_left(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = l + (r - l) // 2
        if target > nums[mid]:
            l = mid + 1
        else:
            r = mid
    return -1 if nums[l] != target else l


def bisect_right(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = l + (r - l) // 2
        if target >= nums[mid]:
            l = mid + 1
        else:
            r = mid
    return -1 if nums[l] != target else l


def find_first_and_last_position_sorted_array1(nums, target):
    return [bisect_left(nums, target), bisect_right(nums, target)]


def find_first_and_last_position_sorted_array2(nums, target):
    def search(l, r):
        if nums[l] == target == nums[r]:
            return [l, r]
        if nums[l] <= target <= nums[r]:
            mid = (l + r) // 2
            l, r = search(l, mid), search(mid + 1, r)
            return max(l, r) if -1 in l + r else [l[0], r[1]]
        return [-1, -1]
    return search(0, len(nums) - 1)


def divide_two_integers(dividend, divisor):
    # dividend = (quotient) * divisor + remainder
    pass


def continuous_sub_array_sum(nums, k):
    prefix_sum, prefix_sum_indices = 0, {0: -1}
    for i, num in enumerate(nums):
        prefix_sum = (prefix_sum + num) % k
        if prefix_sum in prefix_sum_indices and i - prefix_sum_indices[prefix_sum] > 1:
            return True
        if prefix_sum not in prefix_sum_indices:
            prefix_sum_indices[prefix_sum] = i
    return False


def power(x, n):
    if abs(x) < 1e-40:
        return 0
    if n == 0:
        return 1
    if n < 0:
        return power(1/x, -n)
    a = power(x, n // 2)
    if n % 2 == 0:
        return a * a
    if n % 2 == 1:
        return a * a * x


def next_permutation(nums):
    pass


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


def integer_to_english_words1(num):
    one_to_19 = 'One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen' \
                'Seventeen Eighteen Nineteen'.split()
    tens = 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()
    thousands = 'Thousand Million Billion'.split()

    def convert(num, idx=0):
        if num == 0:
            return []
        if num < 20:
            return [one_to_19[num-1]]
        if num < 100:
            return [tens[num // 10 - 2]] + convert(num % 10)
        if num < 1000:
            return [one_to_19[num // 100 - 1]] + ['Hundred'] + convert(num % 100)
        q, r = divmod(num, 1000)
        space = [thousands[idx]] if q % 1000 != 0 else []
        return convert(q, idx + 1) + space + convert(r)
    return ' '.join(convert(num)) or 'Zero'


def integer_to_english_words2(num):
    one_to_19 = 'One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen' \
                'Seventeen Eighteen Nineteen'.split()
    tens = 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()
    thousands = {1000_000_000: 'Billion', 1000_000: 'Million', 1000: 'Thousand'}

    def convert(num):
        if num == 0:
            return []
        if num < 20:
            return [one_to_19[num - 1]]
        if num < 100:
            return [tens[num // 10 - 2]] + convert(num % 10)
        if num < 1000:
            return [one_to_19[num // 100 - 1]] + ['Hundred'] + convert(num % 100)
        else:
            for i in [100_000_0000, 1000_000, 1000]:
                if num // i > 0:
                    return convert(num // i) + [thousands[i]] + convert(num % i)
    return ' '.join(convert(num)) or 'Zero'


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
    # https://leetcode.com/problems/remove-invalid-parentheses/discuss/1639879/python-backtracking-easily-derived-from-combination-backtracking-template
    l, r = 0, 0
    for char in s:
        l += (char == '(')
        if l == 0:
            r += (char == ')')
        else:
            l -= (char == ')')

    def is_valid(s):
        count = 0
        for char in s:
            if char == '(':
                count += 1
            if char == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0

    def dfs(sofar, i, l, r):
        current = ''.join(sofar)
        if l == 0 and r == 0 and is_valid(current):
            result.append(current)
            return
        for j in range(i, n):
            if s[j] not in {'(', ')'}:
                continue
            if j != i and s[j] == s[j - 1]:
                continue
            if r > 0 and s[j] == ')':
                sofar[j] = ''
                dfs(sofar, j + 1, l, r - 1)
                sofar[j] = s[i]
            elif l > 0 and s[j] == '(':
                sofar[j] = ''
                dfs(sofar, j + 1, l - 1, r)
                sofar[j] = s[i]
    n, result = len(s), []
    dfs(list(s), 0, l, r)
    return result


def merge_k_sorted_lists1(lists):
    # We are just putting the first value of each list in the pq
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


def merge_k_sorted_lists2(lists):
    if not lists:
        return None
    if len(lists) == 1:
        return lists[0]
    mid = len(lists) // 2
    list1, list2 = merge_k_sorted_lists2(lists[:mid]), merge_k_sorted_lists2(lists[mid:])
    return merge(list1, list2)


def task_scheduler(tasks, n):
    # https://leetcode.com/problems/task-scheduler/discuss/104507/Python-Straightforward-with-Explanation
    task_counts = collections.Counter(tasks)
    most_frequent_task, most_frequent_task_count = task_counts.most_common(1)[0]
    tasks_with_max_frequency = sum(task_counts[key] == most_frequent_task_count for key in task_counts.keys())
    return max(len(tasks), (most_frequent_task_count - 1) * (n + 1) + tasks_with_max_frequency)


def interval_list_intersections(first_list, second_list):
    i, j, n1, n2, result = 0, 0, len(first_list), len(second_list), []
    while i < n1 and j < n2:
        lo = max(first_list[0], second_list[0])
        hi = min(first_list[1], second_list[1])
        if lo <= hi:
            result.append([lo, hi])
        if first_list[i][1] < second_list[j][1]:
            i += 1
        else:
            j += 1
    return result


def squares_of_a_sorted_array(nums):
    l, r, n = 0, len(nums) - 1, len(nums)
    result, k = [None] * n, n-1
    while l <= r:
        a, b = nums[l] ** 2, nums[r] ** 2
        if a >= b:
            result[k], l = a, l + 1
        else:
            result[k], r = b, r - 1
        k -= 1
    return result


def clone_graph(node):
    if not node:
        return node
    queue = collections.deque([node])
    visited = {node: GraphNode(val=node.val, neighbors=[])}
    while queue:
        current_node = queue.popleft()
        for neighbor in current_node.neighbors:
            if neighbor not in visited:
                copy_node = GraphNode(neighbor.val, [])
                visited[neighbor] = copy_node
                visited[current_node].neighbors.append(copy_node)
                queue.append(neighbor)
            else:
                visited[current_node].neighbors.append(visited[neighbor])
    return visited[node]


def k_th_missing_positive_number(arr, k):
    # https://leetcode.com/problems/kth-missing-positive-number/discuss/1004535/Python-Two-solutions-O(n)-and-O(log-n)-explained
    l, r = 0, len(arr)
    while l < r:
        mid = l + (r - l) // 2
        if arr[mid] - mid + 1 < k:
            l = mid + 1
        else:
            r = mid
    return r + k


def exclusive_time_of_functions(n, logs):
    result, stack, prev_time = [0] * n, [], 0
    for log in logs:
        func, status, ti = log.split(':')
        func, ti = int(func), int(ti)
        if status == 'start':
            if stack:
                result[stack[-1][0]] += (ti - stack[-1][1])
            stack.append([func, ti])
        else:
            result[func] += (ti - stack.pop()[1] + 1)
            if stack:
                stack[-1][1] = ti + 1
    return result


def minimum_window_substring(s, t):
    # https://leetcode.com/problems/minimum-window-substring/discuss/226911/Python-two-pointer-sliding-window-with-explanation
    # https://leetcode.com/problems/minimum-window-substring/discuss/968611/Simple-Python-sliding-window-solution-with-detailed-explanation
    # needed is a default dictionary with 0 default value
    # needed[char] -= 1, even if D is not there in t but in s, then we will add D to needed and reduce it's value
    # from 0 to -1
    pass


def vertical_order_traversal_binary_tree(root):
    """
    More information about sorted(dict.items()). Converts dict to (keys, values) tuple and sorts by keys
    x = {1: [2, 1], 3: 4, 4: 3, 2: 1, 0: 0}
    sorted(x.items()) --> [(0, 0), (1, [2, 1]), (2, 1), (3, 4), (4, 3)]
    x = {0: [(0, 1)], 1: [(-1, 2), (1, 3)], 2: [(-2, 4), (0, 5), (0, 6), (2, 7)]}
    """
    queue = collections.deque([(root, 0, 0)])
    distance_map, result = collections.defaultdict(list), []
    while queue:
        for _ in range(len(queue)):
            current, distance, level = queue.popleft()
            distance_map[distance].append((level, current.val))
            if current.left:
                queue.append((current.left, distance - 1, level + 1))
            if current.right:
                queue.append((current.right, distance + 1, level + 1))
    for key in sorted(distance_map.keys()):  # First order by distance
        col = [i[1] for i in sorted(distance_map[key])]  # Second order by level
        result.append(col)
    return result


def word_break_2(s, word_dict):
    # https://leetcode.com/problems/word-break-ii/discuss/44311/Python-easy-to-understand-solution
    # This solution is in the comments
    result, n, word_dict = [], len(s), set(word_dict)

    def backtrack(i, sofar):
        if i == n:
            result.append(' '.join(sofar))
        for j in range(i, n):
            current_word = s[i: j + 1]
            if current_word in word_dict:
                backtrack(j + 1, sofar + [current_word])
    backtrack(0, [])
    return result


def lowest_common_ancestor_of_binary_tree_3(root, p, q):
    # https://zhenchaogan.gitbook.io/leetcode-solution/leetcode-1650-lowest-common-ancestor-of-a-binary-tree-iii
    if root is None or root == p or root == q:
        return root
    l = lowest_common_ancestor_of_binary_tree_3(root.left, p, q)
    r = lowest_common_ancestor_of_binary_tree_3(root.right, p, q)
    if l and r:
        return root
    return l or r


def move_zeros(nums):
    snow_ball, n = 0, len(nums)
    for i in range(n):
        if nums[i] == 0:
            snow_ball += 1
        elif snow_ball > 0:
            nums[i], nums[i - snow_ball] = nums[i - snow_ball], nums[i]


def buildings_with_an_ocean_view1(heights):
    # https://goodtecher.com/leetcode-1762-buildings-with-an-ocean-view/
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
