import collections
from functools import lru_cache
from LC.LCMetaPractice import TreeNode, ListNode


def two_sum(nums, target):
    hashmap = {num: i for i, num in enumerate(nums)}
    for i, num in enumerate(nums):
        diff = num - target
        if diff in hashmap and i != hashmap[diff]:
            return [i, hashmap[diff]]


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
        carry, digit = divmod(carry, 10)
        current.next = ListNode(digit)
        current = current.next
    return dummy.next


def longest_substring_without_repeating_characters(s):
    # https://leetcode.com/problems/longest-substring-without-repeating-characters/discuss/742926/Simple-Explanation-or-Concise-or-Thinking-Process-and-Example
    n, result, seen, left = len(s), 1, {}, 0
    if n <= 1:
        return n
    for right, char in enumerate(s):
        if char in seen:
            left = max(left, seen[char] + 1)
        result = max(result, right - left + 1)
        seen[char] = right
    return result


def median_of_two_sorted_arrays(nums1, nums2):
    # https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2511/Intuitive-Python-O(log-(m%2Bn))-solution-by-kth-smallest-in-the-two-sorted-arrays-252ms
    n1, n2 = len(nums1), len(nums2)

    def get_kth(nums1, nums2, start1, end1, start2, end2, k):
        if start1 > end1:
            return nums2[k - start1]
        if start2 > end2:
            return nums1[k - start2]
        mid1, mid2 = (start1 + end1) // 2, (start2 + end2) // 2
        mid1_val, mid2_val = nums1[mid1], nums2[mid2]
        if mid1 + mid2 < k:
            if mid1_val > mid2_val:
                return get_kth(nums1, nums2, start1, end1, mid2 + 1, end2, k)
            else:
                return get_kth(nums1, nums2, mid1 + 1, end1, start2, end2, k)
        else:
            if mid1_val > mid2_val:
                return get_kth(nums1, nums2, start1, mid1 - 1, start2, end2, k)
            else:
                return get_kth(nums1, nums2, start1, end1, start2, mid2 - 1, k)

    if (n1 + n2) % 2 == 1:
        return get_kth(nums1, nums2, 0, n1 - 1, 0, n2 - 1, (n1 + n2) // 2)
    else:
        mid1 = get_kth(nums1, nums2, 0, n1 - 1, 0, n2 - 1, (n1 + n2) // 2)
        mid2 = get_kth(nums1, nums2, 0, n1 - 1, 0, n2 - 1, (n1 + n2) // 2 - 1)
        return (mid1 + mid2) / 2


def longest_palindromic_substring(s):
    # https://leetcode.com/problems/longest-palindromic-substring/discuss/2954/Python-easy-to-understand-solution-with-comments-(from-middle-to-two-ends).
    # in comments
    n, result = len(s), ''

    @lru_cache
    def helper(l, r):
        while l >= 0 and r < n and s[l] == s[r]:
            l, r = l - 1, r + 1
        return s[l + 1: r]  # Because after the loop -> s[l] != s[r], we need to take the previous slice -> s[l+1:r]

    for i in range(n):
        result = max(helper(i, i), helper(i, i + 1), result, key=len)
    return result


def regular_expression_matching_without_star(s, p):
    if not p:
        return not s
    first_match = bool(s) and p[0] in {s[0], '.'}
    return first_match and regular_expression_matching_without_star(s[1:], p[1:])


def regular_expression_matching(s, p):
    ns, np, memo = len(s), len(p), {}

    def recursive(i, j):
        if (i, j) not in memo:
            if j == np:
                result = i == ns
            else:
                first_match = i < ns and p[j] in {s[i], '.'}
                if j + 1 < np and p[j + 1] == '*':
                    result = recursive(i, j + 2) or first_match and recursive(i + 1, j)
                else:
                    result = first_match and recursive(i + 1, j + 1)

            memo[i, j] = result
        return memo[i, j]

    return recursive(0, 0)


def container_with_most_water(heights):
    result, left, right = 0, 0, len(heights) - 1
    while left < right:
        result = max(result, min(heights[left], heights[right]) * (right - left))
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    return result


def three_sum1(nums):
    result, negatives, positives, zeros, neg_set, pos_set = set(), [], [], [], set(), set()
    for num in nums:
        if num > 0:
            positives.append(num)
            pos_set.add(num)
        elif num < 0:
            negatives.append(num)
            neg_set.add(num)
        else:
            zeros.append(num)

    if len(zeros) >= 3:
        result.add((0, 0, 0))

    if zeros:
        for num in pos_set:
            if -num in neg_set:
                result.add((-num, 0, num))

    for i in range(len(negatives)):
        for j in range(i + 1, len(negatives)):
            target = -1 * (negatives[i] + negatives[j])
            if target in pos_set:
                result.add((negatives[i], negatives[j], target))

    for i in range(len(positives)):
        for j in range(i + 1, len(positives)):
            target = -1 * (positives[i] + positives[j])
            if target in neg_set:
                result.add((positives[i], positives[j], target))
    return list(result)


def three_sum2(nums):
    # comments of https://leetcode.com/problems/3sum/discuss/7392/Python-easy-to-understand-solution-(O(n*n)-time).
    n, nums, result = len(nums), sorted(nums), []
    for l in range(n - 2):
        if l > 0 and nums[l] == nums[l - 1]:
            continue
        m, r = l + 1, n - 1
        while m < r:
            current_sum = nums[l] + nums[m] + nums[r]
            if current_sum < 0:
                m += 1
            elif current_sum > 0:
                r -= 1
            else:
                result.append([nums[l], nums[m], nums[r]])
                while m < r and nums[m] == nums[m + 1]:
                    m += 1
                while m < r and nums[r] == nums[r - 1]:
                    r -= 1
                m, r = m + 1, r - 1
    return result


def letter_combinations_of_a_phone_number1(digits):
    mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
    result, n = [], len(digits)

    def backtrack(sofar, k):
        if k == n:
            result.append(sofar)
        else:
            letters = mapping[digits[k]]
            for chosen in letters:
                backtrack(sofar + chosen, k + 1)

    backtrack(sofar='', k=0)
    return result if digits else []


def letter_combinations_of_a_phone_number2(digits):
    mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
    result, n = [], len(digits)

    def backtrack(sofar, k):
        if len(sofar) == n:
            result.append(sofar)
        else:
            for i in range(k, n):
                letters = mapping[digits[i]]
                for chosen in letters:
                    backtrack(sofar + chosen, i + 1)

    backtrack(sofar='', k=0)
    return result if digits else []


def remove_nth_node_from_end_of_list1(head, n):
    size, current = 0, head
    while current:
        current = current.next
        size += 1

    current = head
    for _ in range(1, size - n):
        current = current.next

    current.next = current.next.next
    return head


def remove_nth_node_from_end_of_list2(head, n):
    slow = fast = head
    for _ in range(n):
        fast = fast.next
    if not fast:
        return head.next
    while fast.next:
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next
    return slow


def valid_parentheses(s):
    if len(s) % 2 == 1:
        return False
    mapping, stack = {'(': ')', '[': ']', '{': '}'}, []
    for char in s:
        if char in mapping:
            stack.append(char)
        else:
            if stack and mapping[stack[-1]] == char:
                stack.pop()
            else:
                return False
    return len(stack) == 0


def merge_two_sorted_lists1(list1, list2):
    dummy = current = ListNode(-1)
    while list1 and list2:
        if list1.val < list2.val:
            current.next = list1
            list1.next = list1
        else:
            current.next = list2
            list2.next = list2
        current.next = current
    current.next = list1 or list2
    return dummy.next


def merge_two_sorted_lists2(list1, list2):
    def recursive(l1, l2):
        if not l1 or not l2:
            return l1 or l2
        if l1.val < l2.val:
            l1.next = recursive(l1.next, l2)
            return l1
        else:
            l2.next = recursive(l1, l2.next)
            return l2

    return recursive(list1, list2)


def generate_parenthesis(n):
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


def merge_k_sorted_lists(lists):
    if not lists:
        return None
    if len(lists) == 1:
        return lists[0]
    mid = len(lists) // 2
    l1, l2 = merge_k_sorted_lists(lists[:mid]), merge_k_sorted_lists(lists[mid:])
    return merge(l1, l2)


def next_permutation(nums):
    # https://leetcode.com/problems/next-permutation/discuss/14054/Python-solution-with-comments.
    # https://www.nayuki.io/page/next-lexicographical-permutation-algorithm
    i = j = len(nums) - 1
    while i > 0 and nums[i - 1] >= nums[i]:
        i -= 1
    if i == 0:
        nums = nums[::-1]

    k = i - 1
    while nums[k] >= nums[j]:
        j -= 1
    nums[k], nums[j] = nums[j], nums[k]
    l, r = k + 1, len(nums) - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l, r = l + 1, r - 1


def search_in_rotated_sorted_array(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if target == nums[mid]:
            return mid
        if nums[l] <= nums[mid]:
            if nums[l] <= target <= nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        else:
            if nums[mid] <= target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
    return -1


def combination_sum1(candidates, target):
    result, n, candidates = [], len(candidates), sorted(candidates)

    def backtrack(sofar, total, k):
        if total < 0:
            return
        if total == 0:
            result.append(sofar[:])
            return
        for i in range(k, n):
            chosen = candidates[i]
            backtrack(sofar + [chosen], total - chosen, i)

    backtrack([], target, 0)
    return result


def combination_sum2(candidates, target):
    result, n, candidates = [], len(candidates), sorted(candidates)

    def backtrack(sofar, total, k):
        if total < 0:
            return
        if total == 0:
            result.append(sofar[:])
        else:
            for i in range(k, n):
                chosen = candidates[i]
                backtrack(sofar + [chosen], total - chosen, i)

    backtrack([], target, 0)
    return result


def first_missing_positive(nums):
    # https://leetcode.com/problems/first-missing-positive/discuss/17080/Python-O(1)-space-O(n)-time-solution-with-explanation
    # Read dennisch comment under angelsun comment
    nums.append(0)
    n = len(nums)
    for i in range(n):
        if not 0 <= nums[i] < n:
            nums[i] = 0
    for i in range(n):
        nums[nums[i] % n] += n
    for i in range(1, n):
        if nums[i] / n == 0:
            return i
    return n


def trapping_rain_water(heights):
    n, result = len(heights), 0
    max_left_heights, max_right_heights = [0] * n, [0] * n
    for i in range(1, n):
        max_left_heights[i] = max(max_right_heights[i - 1], heights[i - 1])
    for i in range(n - 2, -1, -1):
        max_right_heights[i] = max(max_right_heights[i + 1], heights[i + 1])
    for i in range(n):
        water_level = min(max_left_heights[i], max_right_heights[i])
        if water_level > heights[i]:
            result += (water_level - heights[i])
    # for i in range(1, n):
    #     # l, r
    #     print(i, i - 1, n - i - 1, n - i)
    return result


def permutations(nums):
    # https://leetcode.com/problems/permutations/discuss/993970/Python-4-Approaches-%3A-Visuals-%2B-Time-Complexity-Analysis
    # https://leetcode.com/problems/subsets/discuss/1598122/Python-subsets-vs.-combinations-vs.-permutations-or-Visualized
    # pv remembering keyword
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


def rotate_image(matrix):
    # https://leetcode.com/problems/rotate-image/discuss/18884/Seven-Short-Solutions-(1-to-7-lines)
    n = len(matrix)
    for i in range(n):  # This is Transpose
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:  # This is Reverse
        for j in range(n // 2):
            row[j], row[~j] = row[~j], row[j]


def print_matrix(matrix):
    for row in matrix:
        print(row)


def group_anagrams1(strs):
    hashmap = collections.defaultdict(list)
    for word in strs:
        key = tuple(sorted(word))
        hashmap[key].append(word)
    return hashmap.values()


def group_anagrams2(strs):
    mapping = collections.defaultdict(list)
    for word in strs:
        letter_count_map = [0] * 26
        for char in word:
            letter_count_map[ord(char) - ord('a')] += 1
        mapping[tuple(letter_count_map)].append(word)
    return mapping.values()


def maximum_subarray1(nums):
    # https://leetcode.com/problems/maximum-subarray/discuss/1595195/C%2B%2BPython-7-Simple-Solutions-w-Explanation-or-Brute-Force-%2B-DP-%2B-Kadane-%2B-Divide-and-Conquer
    n = len(nums)

    @lru_cache
    def recursive(i, pick):
        if i >= n:
            return 0 if pick else float('-inf')
        return max(nums[i] + recursive(i + 1, True), 0 if pick else recursive(i + 1, False))

    return recursive(0, False)


def maximum_subarray2(nums):
    result, local_result = float('-inf'), 0
    for num in nums:
        local_result = max(num, num + local_result)
        result = max(result, local_result)
    return result


def jump_game1(nums):
    # https://leetcode.com/problems/jump-game/discuss/1443541/Python-3-approaches%3A-Top-down-DP-Bottom-up-DP-Max-Pos-So-Far-Clean-and-Concise
    n = len(nums)

    @lru_cache
    def recursive(i):
        if i >= n - 1:
            return True
        for j in range(i + 1, min(i + nums[i], n - 1) + 1):
            if recursive(j):
                return True
        return False
    return recursive(0)


def jump_game2(nums):
    i, max_pos, n = 0, 0, len(nums)
    while i <= max_pos:
        max_pos = max(max_pos, i + nums[i])
        if max_pos >= n - 1:
            return True
        i += 1
    return False


def jump_game3(nums):
    # https://leetcode.com/problems/jump-game/discuss/596454/Python-Simple-solution-with-thinking-process-Runtime-O(n)
    last_pos, n = len(nums) - 1, len(nums)
    for i in range(n - 2, -1, -1):
        if i + nums[i] >= last_pos:
            last_pos = i
    return last_pos == 0


def merge_intervals(intervals):
    result, intervals = [], sorted(intervals, key=lambda x: x[0])
    for start, end in intervals:
        if not result or result[-1][-1] < start:
            result.append([start, end])
        else:
            result[-1][-1] = max(result[-1][-1], end)
    return result


def unique_paths(m, n):
    # for more solutions
    # https://leetcode.com/problems/unique-paths/discuss/1581998/C%2B%2BPython-5-Simple-Solutions-w-Explanation-or-Optimization-from-Brute-Force-to-DP-to-Math
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]


def minimum_path_sum(grid):
    # https://leetcode.com/problems/minimum-path-sum/discuss/1467216/Python-Bottom-up-DP-In-place-Clean-and-Concise
    # https://leetcode.com/problems/minimum-path-sum/discuss/1271002/Python-Recursive-and-Dynamic-Programming-Solutions
    m, n = len(grid), len(grid[0])
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                pass
            elif i == 0:
                grid[i][j] += grid[i][j - 1]
            elif j == 0:
                grid[i][j] += grid[i - 1][j]
            else:
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
    return grid[-1][-1]


def climbing_stairs1(n):
    # https://leetcode.com/problems/climbing-stairs/discuss/1531764/Python-%3ADetail-explanation-(3-solutions-easy-to-difficult)-%3A-Recursion-dictionary-and-DP
    def recursive(i):
        if i in [1, 2]:
            return i
        return recursive(i - 1) + recursive(i - 2)
    return recursive(n)


def climbing_stairs2(n):
    if n <= 3:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[-1]


def edit_distance(word1, word2):
    # https://leetcode.com/problems/edit-distance/discuss/159295/Python-solutions-and-intuition
    # https://leetcode.com/problems/edit-distance/discuss/1475220/Python-3-solutions-Top-down-DP-Bottom-up-DP-O(N)-in-Space-Clean-and-Concise
    cache = {}

    def recursive(word1, word2):
        if not word1 and not word2:
            return 0
        if not word1 or not word2:
            return len(word1) or len(word2)
        if word1[0] == word2[0]:
            return recursive(word1[1:], word2[1:])
        if (word1, word2) not in cache:
            inserted = 1 + recursive(word1, word2[1:])
            deleted = 1 + recursive(word1[1:], word2)
            replaced = 1 + recursive(word1[1:], word2[1:])
            cache[(word1, word2)] = min(inserted, deleted, replaced)
        return cache[(word1, word2)]
    return recursive(word1, word2)


def sort_colors(nums):
    # https://leetcode.com/problems/sort-colors/discuss/26481/Python-O(n)-1-pass-in-place-solution-with-explanation
    red, white, blue = 0, 0, len(nums) - 1

    while white <= blue:
        if nums[white] == 0:
            nums[white], nums[red] = nums[red], nums[white]
            white, red = white + 1, red + 1
        elif nums[white] == 1:
            white += 1
        else:
            nums[white], nums[blue] = nums[blue], nums[white]
            blue -= 1


def minimum_window_substring(s, t):
    pass


def subsets1(nums):
    n, result = len(nums), []

    def backtrack(sofar, k):
        result.append(sofar[:])
        for i in range(k, n):
            chosen = nums[i]
            backtrack(sofar + [chosen], i + 1)
    backtrack(sofar=[], k=0)
    return result


def subsets2(nums):
    # https://leetcode.com/problems/subsets/discuss/1598122/Python-subsets-vs.-combinations-vs.-permutations-or-Visualized
    result, stack = [], [(0, [])]
    while stack:
        k, sofar = stack.pop()
        result.append(sofar[:])
        for i in range(k, len(nums)):
            chosen = nums[i]
            stack.append((i + 1, sofar + [chosen]))
    return result


def word_search(board, word):
    directions, found = [(1, 0), (0, 1), (-1, 0), (0, -1)], [False]
    m, n, k = len(board), len(board[0]), len(word)

    def backtrack(idx, x, y):
        if found[0]:
            return
        if idx == k:
            found[0] = True
            return
        if x < 0 or x >= m or y < 0 or y >= n or board[x][y] != word[idx]:
            return
        board[x][y], temp = '#', board[x][y]
        for dx, dy in directions:
            backtrack(idx + 1, x + dx, y + dy)
        board[x][y] = temp

    for i in range(m):
        for j in range(n):
            if found[0]:
                return True
            backtrack(0, i, j)
    return found[0]


def largest_rectangle_in_histogram(heights):
    # https://leetcode.com/problems/largest-rectangle-in-histogram/discuss/995249/Python-increasing-stack-explained
    stack, result = [], 0
    for i, height in enumerate(heights + [0]):
        while stack and height[stack[-1]] >= height:
            H = heights[stack.pop()]
            W = i if not stack else i - stack[-1] - 1
            result = max(result, H * W)
        stack.append(i)
    return result


def maximal_rectangle(matrix):
    pass


def binary_tree_inorder_traversal1(root):
    result = []

    def dfs(node):
        if node:
            dfs(node.left)
            result.append(node.val)
            dfs(node.right)
    dfs(root)
    return result


def binary_tree_inorder_traversal2(root):
    result, stack, current = [], [], root
    while True:
        if current:
            stack.append(current)
            current = current.left
        elif stack:
            current = stack.pop()
            result.append(current.val)
            current = current.right
        else:
            break
    return result


def unique_binary_search_trees1(n):
    # https://leetcode.com/problems/unique-binary-search-trees/discuss/1565543/C%2B%2BPython-5-Easy-Solutions-w-Explanation-or-Optimization-from-Brute-Force-to-DP-to-Catalan-O(N)

    @lru_cache
    def recursive(n):
        if n <= 1:
            return 1
        return sum(recursive(i - 1) * recursive(n - i) for i in range(1, n + 1))


def unique_binary_search_trees2(n):
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        for j in range(1, i + 1):
            dp[i] += dp[j - 1] * dp[i - j]
    return dp[n]


def validate_binary_search_tree(root):
    def dfs(node, low=float('-inf'), high=float('inf')):
        if not root:
            return True
        if not low < node.val < high:
            return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)


def symmetric_tree1(root):
    if not root:
        return True

    def dfs(left, right):
        if left and right:
            return left.val == right.val and dfs(left.left, right.right) and dfs(left.right, right.left)
        return left == right

    return dfs(root.left, root.right)


def symmetric_tree2(root):
    # https://leetcode.com/problems/symmetric-tree/discuss/33325/Python-short-recursive-and-iterative-solutions
    if not root:
        return True
    stack = [(root.left, root.right)]
    while stack:
        l, r = stack.pop()
        if not l and not r:
            continue
        if not l or not l or l.val != r.val:
            return False
        stack.append((l.left, r.right))
        stack.append((l.right, r.left))
    return True


def symmetric_tree3(root):
    # https://leetcode.com/problems/symmetric-tree/discuss/33057/Python-iterative-way-using-a-queue
    if not root:
        return True
    queue = collections.deque([(root.left, root.right)])
    while queue:
        l, r = queue.popleft()
        if not l and not r:
            continue
        if not l or not l or l.val != r.val:
            return False
        queue.append((l.left, r.right))
        queue.append((l.right, r.left))
    return True


def maximum_depth_of_binary_tree1(root):
    if not root:
        return 0
    l, r = maximum_depth_of_binary_tree1(root.left), maximum_depth_of_binary_tree1(root.right)
    return max(l, r) + 1


def maximum_depth_of_binary_tree2(root):
    if not root:
        return 0
    level, queue = 0, collections.deque([root])
    if queue:
        level, size = level + 1, len(queue)
        for _ in range(size):
            current = queue.popleft()
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
    return level


def construct_binary_tree_from_preorder_and_inorder_traversal1(preorder, inorder):
    # https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/discuss/34579/Python-short-recursive-solution.
    if inorder:
        index = inorder.index(preorder.pop(0))
        node = TreeNode(inorder[index])
        node.left = construct_binary_tree_from_preorder_and_inorder_traversal1(preorder, inorder[:index])
        node.right = construct_binary_tree_from_preorder_and_inorder_traversal1(preorder, inorder[index + 1:])
        return node


def construct_binary_tree_from_preorder_and_inorder_traversal(preorder, inorder):
    inorder_hashmap = {num: i for i, num in enumerate(inorder)}
    preorder_iter, n = iter(preorder), len(preorder)

    def recursive(start, end):
        if start > end:
            return None
        node_val = next(preorder_iter)
        node = TreeNode(node_val)
        index = inorder_hashmap[node_val]
        node.left = recursive(start, index - 1)
        node.right = recursive(index + 1, end)
        return node

    return recursive(0, n - 1)


def flatten_binary_tree_to_linked_list(root):
    # https://leetcode.com/problems/flatten-binary-tree-to-linked-list/discuss/1208004/Extremely-Intuitive-O(1)-Space-solution-with-Simple-explanation-Python
    current = root
    while current:
        if current.left:
            p = current.left
            while p.right:
                p = p.right
            p.right = current
            current.right = current.left
            current.left = None
        current = current.right


def best_time_to_buy_and_sell_stock(prices):
    current_max, result, n = 0, 0, len(prices)
    for i in range(1, n):
        change_in_price = prices[i] - prices[i - 1]
        current_max += change_in_price
        current_max = max(current_max, 0)
        result = max(current_max, result)
    return result


def longest_consecutive_sequence1(nums):
    # https://leetcode.com/problems/longest-consecutive-sequence/discuss/1254638/Short-and-Easy-Solution-w-Explanation-or-O(N)-Solution-with-comments-using-hashset
    nums, n = sorted(nums), len(nums)
    result, current_longest = 0, min(1, n)
    for i in range(1, n):
        if nums[i] == nums[i - 1]:
            continue
        if nums[i] != nums[i - 1] + 1:
            current_longest += 1
        else:
            result, current_longest = max(result, current_longest), 1
    return max(result, current_longest)


def longest_consecutive_sequence2(nums):
    result, s = 0, set(nums)
    for num in s:
        if num - 1 in s:
            continue
        j = 1
        while num + j in s:
            j += 1
        result = max(result, j)
    return result


def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result


def word_break(s, word_dict):
    # https://leetcode.com/problems/word-break/discuss/1455100/Python-3-solutions%3A-Top-down-DP-Bottom-up-DP-then-Optimised-with-Trie-Clean-and-Concise
    word_dict, n = set(word_dict), len(s)

    @lru_cache
    def dfs(k):
        if k == n:
            return True
        for i in range(k + 1, n + 1):
            chosen = s[k: i]
            if chosen in word_dict and dfs(i):
                return True
        return True
    return dfs(0)


def linked_list_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def linked_list_cycle_2(head):
    # https://leetcode.com/problems/linked-list-cycle-ii/discuss/912276/Python-2-pointers-approach-explained
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return

    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow


class LRUCache:
    pass


def sort_list(head):
    def merge(list1, list2):
        if not list1 or not list2:
            return list1 or list2
        current = dummy = ListNode(0)

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

    if not head or not head.next:
        return head
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    start = slow.next
    slow.next = None
    left, right = sort_list(head), sort_list(start)
    return merge(left, right)
