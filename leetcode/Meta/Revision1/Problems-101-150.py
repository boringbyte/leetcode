import heapq
import collections
from functools import lru_cache
from leetcode.LCMetaPractice import ListNode, TreeNode


def longest_common_prefix1(strs):
    if not strs:
        return ''
    prefix, n = [], len(strs)
    for chars in zip(*strs):
        if len(set(chars)) == 1:
            prefix.append(chars[0])
        else:
            break
    return ''.join(prefix)


def longest_common_prefix2(strs: list[str]):
    if not strs:
        return ''
    shortest = min(strs, key=len)
    for i, char in enumerate(shortest):
        for other in strs:
            if other[i] != char:
                return shortest[:i]
    return shortest


def meeting_rooms_2(intervals):
    # https://zhenyu0519.github.io/2020/07/13/lc253/#sample-io
    n, heap = len(intervals), []
    if n <= 1:
        return n
    for start, end in sorted(intervals):
        if heap and start >= heap[0]:
            heapq.heappushpop(heap, end)
        else:
            heapq.heappush(heap, end)
    return len(heap)


def validate_binary_search_tree(root):
    # In comments of https://leetcode.com/problems/validate-binary-search-tree/discuss/32178/Clean-Python-Solution
    def dfs(node, low=float('-inf'), high=float('inf')):
        if not root:
            return True
        if not low < node.val < high:
            return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)

    return dfs(root)


def diagonal_traversal(matrix):
    # Use dict and sum of i+j is same for elements on the diagonal
    m, n, result = len(matrix), len(matrix[0]), []
    diagonal_map = collections.defaultdict(list)

    for i in range(m):
        for j in range(n):
            diagonal_map[i + j].append(matrix[i][j])

    for key in sorted(diagonal_map):
        if key % 2 == 0:
            result.extend(diagonal_map[key][::-1])
        else:
            result.extend(diagonal_map[key])
    return result
    # itertools.chain(*[v if k % 2 else v[::-1] for k, v in d.items()])


def check_completeness_of_a_binary_tree1(root):
    # https://leetcode.com/problems/check-completeness-of-a-binary-tree/discuss/533798/Python-O(n)-by-level-order-traversal.-90%2B-w-Diagram
    if not root:
        return True
    queue = collections.deque([root])
    prev_node = root
    while queue:
        current = queue.popleft()
        if current:
            if not prev_node:
                return False
            queue.append(current.left)
            queue.append(current.right)
        prev_node = current
    return True


def check_completeness_of_a_binary_tree2(root):
    # https://leetcode.com/problems/check-completeness-of-a-binary-tree/discuss/242287/Python-solution
    if not root:
        return True
    queue, result = collections.deque([(root, 1)]), []
    while queue:
        current, coord = queue.popleft()
        result.append(coord)
        if current.left:
            queue.append((current.left, 2 * coord))
        if current.right:
            queue.append((current.right, 2 * coord + 1))
    return len(result) == result[-1]


def nested_list_weight_sum(nested_list):
    # https://zhenyu0519.github.io/2020/03/16/lc339/
    def dfs(current_list, depth):
        total = 0
        for value in current_list:
            if isinstance(value, int):
                total += (value * depth)
            else:
                total += dfs(value, depth + 1)
        return total

    return dfs(nested_list, 1)


def permutations(nums):
    # pvr logic
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


def course_schedule(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses
    for curr, prev in prerequisites:
        graph[prev].append(curr)
        in_degree[curr] += 1

    queue = collections.deque(v for v in range(num_courses) if in_degree[v] == 0)
    n = len(queue)
    while queue and n != num_courses:
        current_course = queue.popleft()
        for next_course in graph[current_course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                n += 1
                queue.append(next_course)
    return n == num_courses


def reverse_linked_list(head):
    prev = None
    while head:
        current = head
        head = head.next
        current.next = prev
        prev = current
    return prev


def palindrome_linked_list(head):
    # https://leetcode.com/problems/palindrome-linked-list/discuss/1137696/Short-and-Easy-w-Explanation-or-T-%3A-O(N)-S-%3A-O(1)-Solution-using-Fast-and-Slow
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    slow = reverse_linked_list(slow)
    while slow:
        if slow.val != head.val:
            return False
        slow = slow.next
        head = head.next
    return True


def strobogrammatic_number(num):
    # https://www.tutorialcup.com/leetcode-solutions/strobogrammatic-number-leetcode-solution.htm
    number_map = {('0', '0'), ('1', '1'), ('6', '9'), ('8', '8'), ('9', '6')}
    i, j = 0, len(num) - 1
    while i <= j:
        if (num[i], num[j]) not in number_map:
            return False
        i, j = i + 1, j - 1
    return True


def first_missing_positive1(nums):
    # https://leetcode.com/problems/first-missing-positive/discuss/17080/Python-O(1)-space-O(n)-time-solution-with-explanation
    nums = [0] + nums
    n = len(nums)
    for i in range(n):
        if nums[i] < 0 or nums[i] >= n:
            nums[i] = 0
    for i in range(n):
        nums[nums[i] % n] += n
    for i in range(1, n):
        if nums[i] // n == 0:
            return i
    return n


def first_missing_positive2(nums):
    # https://leetcode.com/problems/first-missing-positive/discuss/17161/Python-O(n)-and-O(nlgn)-solutions.
    n = len(nums)
    for i in range(n):
        while 0 <= nums[i] - 1 < n and nums[nums[i] - 1] != nums[i]:
            ni = nums[i] - 1
            nums[i], nums[ni] = nums[ni], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


def construct_binary_tree_from_string(s):
    # https://linlaw0229.github.io/2021/09/16/536-Construct-Binary-Tree-from-String/
    if not s or len(s) == 0:
        return None

    def dfs(s, i):
        start = i
        if s[start] == '-':
            i += 1
        while i < len(s) and s[i].isdigit():
            i += 1
        node = TreeNode(int(s[start: i]))

        if i < len(s) and s[i] == '(':
            i += 1
            node.left, i = dfs(s, i)
            i += 1

        if i < len(s) and s[i] == '(':
            i += 1
            node.right, i = dfs(s, i)
            i += 1

        return node, i

    root, idx = dfs(s, 0)
    return root


def generate_parentheses(n):
    result = []

    def backtrack(sofar, left, right):
        if len(sofar) == 2 * n:
            result.append(sofar)
        else:
            if left < n:
                backtrack(sofar + '(', left + 1, right)
            if right < left:
                backtrack(sofar + ')', left, right + 1)

    backtrack('', 0, 0)
    return result


def median_of_two_sorted_arrays(nums1, nums2):
    # https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2511/Intuitive-Python-O(log-(m%2Bn))-solution-by-kth-smallest-in-the-two-sorted-arrays-252ms
    pass


def missing_ranges(nums, lower, upper):
    # https://goodtecher.com/leetcode-163-missing-ranges/
    result, previous = [], lower - 1

    def get_range(left, right):
        if left == right:
            return f'{left}'
        return f'{left}->{right}'

    if not nums:
        gap = get_range(lower, upper)
        result.append(gap)
        return result

    for num in nums:
        if previous + 1 != num:
            gap = get_range(previous + 1, num - 1)
            result.append(gap)
        previous = num

    if nums[-1] < upper:
        gap = get_range(nums[-1] + 1, upper)
        result.append(gap)

    return result


class NumMatrix:
    # https://leetcode.com/problems/range-sum-query-2d-immutable/discuss/572648/C%2B%2BJavaPython-Prefix-sum-with-Picture-explain-Clean-and-Concise
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        self.total = [[0] * (n + 1) for _ in range(m + 1)]
        for r in range(1, m + 1):
            for c in range(1, n + 1):
                self.total[r][c] = self.total[r - 1][c] + self.total[r][c - 1] - self.total[r - 1][c - 1] + matrix[r - 1][c - 1]

    def sum_region(self, r1, c1, r2, c2):
        r1, c1, r2, c2 = r1 + 1, c1 + 1, r2 + 1, c2 + 1
        return self.total[r2][c2] - self.total[r1 - 1][c1] - self.total[r1][c1 - 1] + self.total[r1 - 1][c1 - 1]


def populating_next_right_pointer_in_each_node1(root):
    # https://leetcode.com/problems/populating-next-right-pointers-in-each-node/discuss/1654181/C%2B%2BPythonJava-Simple-Solution-w-Images-and-Explanation-or-BFS-%2B-DFS-%2B-O(1)-Optimized-BFS
    if root is None:
        return root

    queue = collections.deque([root])
    while queue:
        right_node = None
        for _ in range(len(queue)):
            current = queue.popleft()
            current.next = right_node
            right_node = current
            if current.right:
                queue.extend([current.right, current.left])
    return root


def populating_next_right_pointer_in_each_node2(root):
    if not root:
        return None
    left, right, nxt = root.left, root.right, root.next
    if left:
        left.next = right
        if nxt:
            right.next = nxt.left
            populating_next_right_pointer_in_each_node2(left)
            populating_next_right_pointer_in_each_node2(right)
    return root


def reverse_integer(x):
    # https://leetcode.com/problems/reverse-integer/discuss/4527/A-Python-solution-O(n)-58ms
    result, sign = 0, 1
    if x < 0:
        sign, x = -sign, -x

    while x:
        q, r = divmod(x, 10)
        result = result * 10 + r  # x % 10
        x = q

    return 0 if result > pow(2, 31) else result * sign


def minimum_cost_for_tickets(days, costs):
    days_map, last_day = set(days), days[-1]
    dp = [0] * (last_day + 1)

    for day in range(1, last_day + 1):
        if day not in days_map:
            dp[day] = dp[day - 1]
        else:
            dp[day] = min(
                dp[max(0, day - 1)] + costs[0],  # per days value
                dp[max(0, day - 7)] + costs[1],  # per week value
                dp[max(0, day - 30)] + costs[2]  # per year value
            )
    return dp[-1]


def minimum_knight_moves():
    pass


def sum_root_to_leaf_numbers1(root):
    # https://leetcode.com/problems/sum-root-to-leaf-numbers/discuss/1556417/C%2B%2BPython-Recursive-and-Iterative-DFS-%2B-BFS-%2B-Morris-Traversal-O(1)-or-Beats-100
    def dfs(node, value):
        if not node:
            return 0
        value = value * 10 + node.val
        if not node.left and not node.right:
            return value
        return dfs(node.left, value) + dfs(node.right, value)

    return dfs(root, 0)


def sum_root_to_leaf_numbers2(root):
    stack, result = [(root, 0)], 0
    while stack:
        current, value = stack.pop()
        value = value * 10 + current.val
        if not current.left and not current.right:
            result += value
        if current.right:
            stack.append((current.right, value))
        if current.left:
            stack.append((current.left, value))
    return result


def palindromic_substrings1(s):
    # https://leetcode.com/problems/palindromic-substrings/discuss/105687/Python-Straightforward-with-Explanation-(Bonus-O(N)-solution)
    n, result = len(s), 0
    for i in range(2 * n - 1):
        left, right = i // 2, (i + 1) // 2
        while left >= 0 and right < n and s[left] == s[right]:
            result, left, right = result + 1, left - 1, right + 1
    return result


def palindromic_substrings2(s):
    n, result = len(s), [0]

    def count_palindrome(s, l, r):
        while l >= 0 and r < n and s[l] == s[r]:
            result[0], l, r = result[0] + 1, l - 1, r + 1

    for i in range(n):
        count_palindrome(s, i, i)
        count_palindrome(s, i, i + 1)

    return result[0]


def binary_tree_level_order_traversal(root):
    if not root:
        return []
    queue, result = collections.deque([root]), []

    while queue:
        cur_level, size = [], len(queue)
        for _ in range(size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            cur_level.append(node.val)
        result.append(cur_level)
    return result


def read_n_characters_given_read4():
    pass


def path_sum(root, total):
    # https://leetcode.com/problems/path-sum/discuss/36486/Python-solutions-(DFS-recursively-DFS%2Bstack-BFS%2Bqueue
    if not root:
        return False
    stack = [(root, root.val)]
    while stack:
        current, value = stack.pop()
        if not current.left and not current.right and value == total:
            return True
        if current.right:
            stack.append((current.right, value + current.right.val))
        if current.left:
            stack.append((current.left, value + current.left.val))
    return False


def find_pivot_index(nums):
    total, left_sum = sum(nums), 0
    for i, num in enumerate(nums):
        if left_sum == (total - left_sum - num):
            return i
        left_sum += num
    return -1


def intersection_of_two_arrays(nums1, nums2):
    # https://leetcode.com/problems/intersection-of-two-arrays/discuss/82006/Four-Python-solutions-with-simple-explanation
    result, hashmap = [], collections.Counter(nums1)
    for num2 in nums2:
        if num2 in hashmap and hashmap[num2] > 0:
            result.append(num2)
            hashmap[num2] = 0
    return result


def can_place_flowers(flowerbed, n):
    count = 0
    flowerbed = [0] + flowerbed + [0]
    for i in range(1, len(flowerbed) - 1):
        if flowerbed[i - 1] == flowerbed[i] == flowerbed[i + 1] == 0:
            flowerbed[i] = 1
            count += 1
    return count >= n


def kth_smallest_element_in_a_sorted_matrix(matrix, k):
    # https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/1322101/C%2B%2BJavaPython-MaxHeap-MinHeap-Binary-Search-Picture-Explain-Clean-and-Concise
    m, n = len(matrix), len(matrix[0])  # For general, matrix doesn't need to be a square
    max_heap = []
    for r in range(m):
        for c in range(n):
            heapq.heappush(max_heap, -matrix[r][c])
            if len(max_heap) > k:
                heapq.heappop(max_heap)
    return -heapq.heappop(max_heap)


def first_unique_character_in_a_string(s):
    count = collections.Counter(s)
    for i, ch in enumerate(s):
        if count[ch] == 1:
            return i
    return -1


def n_queens(n):
    # https://leetcode.com/problems/n-queens/discuss/826728/python-easy_to_read-wexplanation-or-backtracking
    col, diagonal, anti_diagonal = set(), set(), set()
    result = []

    def backtrack(k, sofar):
        if k == n:
            result.append(sofar[:])

        for c in range(n):
            if c in col or (k + c) in diagonal or (k - c) in anti_diagonal:
                continue
            col.add(c)
            diagonal.add(k + c)
            anti_diagonal.add(k - c)
            sofar.append('.' * c + 'Q' + '.' * (n - c - 1))
            backtrack(c + 1, sofar)
            col.remove(c)
            diagonal.remove(k + c)
            anti_diagonal.remove(k - c)
            sofar.pop()
    backtrack(0, [])
    return result


def contiguous_array(nums):
    # https://leetcode.com/problems/contiguous-array/discuss/1743720/Python-Javascript-Easy-solution-with-very-clear-Explanation
    result, hashmap, count = 0, {}, 0
    for i, num in range(len(nums)):
        if num == 0:
            count -= 1
        else:
            count += 1
        if count == 0:
            result = i + 1
        if count in hashmap:
            result = max(result, i - hashmap[count])
        else:
            hashmap[count] = i
    return result


def binary_tree_paths(root):
    # https://leetcode.com/problems/binary-tree-paths/discuss/68272/Python-solutions-(dfs%2Bstack-bfs%2Bqueue-dfs-recursively).
    if not root:
        return []
    result = []

    def dfs(node, path):
        if not node.left and not node.right:
            result.append(path + str(node.val))
        if node.left:
            dfs(node.left, path + str(node.val) + '->')
        if node.right:
            dfs(node.right, path + str(node.val) + '->')

    dfs(root, '')
    return result


def find_all_anagrams_in_a_string(s, p):
    # https://leetcode.com/problems/find-all-anagrams-in-a-string/discuss/639309/JavaPython-Sliding-Window-Detail-explanation-Clean-and-Concise
    count = collections.Counter(p)
    result, l = [], 0
    for r, char in enumerate(s):
        count[char] -= 1
        while count[char] < 0:
            count[s[l]] += 1
            l += 1
        if r - l + 1 == len(p):
            result.append(l)
    return result


def kth_smallest_element_in_a_bst(root, k):
    stack, current = [], root
    while True:
        if current:
            stack.append(current)
            current = current.left
        elif stack:
            current = stack.pop()
            k -= 1
            if k == 0:
                return current.val
            current = current.right


def robot_room_cleaner(robot):
    # Correct
    # directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    # visited = set()
    #
    # def dfs(robot, x, y):
    #     if (x, y) in visited:
    #         return
    #     visited.add((x, y))
    #     robot.clean()
    #     for dx, dy in directions:
    #         nx, ny = x + dx, y + dy
    #         dfs(robot, nx, ny)
    pass


def decode_ways(s):
    # https://leetcode.com/problems/decode-ways/discuss/1410794/C%2B%2BPython-From-Top-down-DP-to-Bottom-up-DP-O(1)-Space-Clean-and-Concise
    n = len(s)

    @lru_cache
    def dfs(i):
        if i == n:
            return 1
        result = 0
        if s[i] != 0:
            result += dfs(i + 1)
        if i + 1 < n and (s[i] == '1' or s[i] == '2' and s[i + 1] <= '6'):
            result += dfs(i + 2)
        return result
    return dfs(0)


def backspace_string_compare1(s, t):
    # https://leetcode.com/problems/backspace-string-compare/discuss/145786/Python-tm
    stack1, stack2 = [], []

    def process(word, stack):
        for char in word:
            if char != '#':
                stack.append(char)
            else:
                if not stack:
                    continue
                stack.pop()
        return stack

    l1 = process(s, stack1)
    l2 = process(t, stack2)
    return l1 == l2


def get_char(word, i):
    char, count = '', 0
    while i >= 0 and not char:
        if word[i] == '#':
            count += 1
        elif count == 0:
            char = word[i]
        else:
            count -= 1
        i -= 1
    return char, i


def backspace_string_compare2(s, t):
    r1, r2 = len(s) - 1, len(t) -1

    while r1 >= 0 or r2 >= 0:
        char1, char2 = '', ''
        if r1 >= 0:
            char1, r1 = get_char(s, r1)
            char2, r2 = get_char(t, r2)
        if char1 != char2:
            return False
        return True


def find_largest_value_in_each_tree_row(root):
    if not root:
        return []
    result, queue = [], collections.deque([root])
    while queue:
        level_result, new_queue = float('-inf'), collections.deque()
        for current in queue:
            level_result = max(level_result, current.val)
            if current.left:
                new_queue.append(current.left)
            if current.right:
                new_queue.append(current.right)
        queue = new_queue
        result.append(level_result)
    return result


def letter_combinations_of_a_phone_number(digits):
    hashmap = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
    result, n = [], len(digits)

    def dfs(sofar, k):
        if k == n:
            result.append(sofar)
        else:
            letters = hashmap[digits[k]]
            for letter in letters:
                dfs(sofar + letter, k + 1)

    dfs('', 0)
    return result if digits else []


def merge(l, r):
    if not l or not r:
        return l or r
    dummy = prev = ListNode(0)
    while l and r:
        if l.val < r.val:
            prev.next = l
            l = l.next
        else:
            prev.next = r
            r = r.next
        prev = prev.next
    prev.next = l or r
    return dummy.next


def sort_list(head):
    # https://leetcode.com/problems/sort-list/discuss/46711/Python-easy-to-understand-merge-sort-solution
    if not head or not head.next:
        return head
    slow, fast = head, head.next
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    list2 = slow.next
    slow.next = None
    list1 = head
    l, r = sort_list(list1), sort_list(list2)
    return merge(l, r)


def island_perimeter(grid):
    # https://leetcode.com/problems/island-perimeter/discuss/723842/Python-O(mn)-simple-loop-solution-explained
    m, n, result = len(grid), len(grid[0]), 0
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for x in range(m):
        for y in range(n):
            if grid[x][y] == 0:
                continue
            result += 4
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                    result -= 1
    return result


def path_sum1(root, target_sum):
    # https://leetcode.com/problems/path-sum-ii/discuss/1382332/C%2B%2BPython-DFS-Clean-and-Concise-Time-complexity-explained
    result = []

    def dfs(node, target_sum, path):
        if not node:
            return None
        target_sum -= node.val
        path.append(node.val)
        if node.left is None and node.right is None:
            if target_sum == 0:
                result.append(path[:])
        else:
            dfs(node.left, target_sum, path)
            dfs(node.right, target_sum, path)
        path.pop()

    dfs(root, target_sum, [])
    return result


def path_sum2(root, target_sum):
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


def monotonic_array(nums):
    increasing = decreasing = True
    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]:
            increasing = False
        if nums[i] < nums[i + 1]:
            decreasing = False
    return increasing or decreasing


def container_with_most_water(height):
    result, l, r = 0, 0, len(height) - 1
    while l < r:
        result = max(result, (r - l) * min(height[l], height[r]))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return result


def spiral_matrix(matrix):
    # comments of https://leetcode.com/problems/spiral-matrix/discuss/1466413/Python-simulate-process-explained
    m, n = len(matrix), len(matrix[0])
    x, y, dx, dy = 0, 0, 1, 0
    result = []
    for _ in range(m * n):
        nx, ny = x + dx, y + dy
        if not 0 <= nx < n or not 0 <= ny < m or matrix[ny][nx] == '*':
            dx, dy = -dy, dx
        result.append(matrix[y][x])
        matrix[y][x] = '*'
        x, y = x + dx, y + dy
    return result
