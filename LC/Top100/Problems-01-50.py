import collections
from functools import lru_cache
from LC.LCMetaPractice import TreeNode, ListNode, DLLNode


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
    n, result, seen, left = len(s), 1, {}, 0
    if n == 0:
        return 0
    for right, char in enumerate(s):
        if char in seen:
            left = max(result, seen[char] + 1)
        result = max(result, right - left + 1)
        seen[char] = right
    return result


def median_of_two_sorted_arrays(nums1, nums2):
    # https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2511/Intuitive-Python-O(log-(m%2Bn))-solution-by-kth-smallest-in-the-two-sorted-arrays-252ms
    pass


def longest_palindromic_substring(s):
    # https://leetcode.com/problems/longest-palindromic-substring/discuss/2954/Python-easy-to-understand-solution-with-comments-(from-middle-to-two-ends).
    # in comments
    n, result = len(s), ''

    def helper(l, r):
        while l >= 0 and r < n and s[l] == s[r]:
            l, r = l - 1, r + 1
        return s[l + 1: r]

    for i in range(n):
        result = max(helper(i, i), helper(i, i + 1), result, key=len)
    return result


def regular_expression_matching(s, p):
    pass


def container_with_most_water(heights):
    result, left, right = 0, 0, len(heights) - 1
    while left < right:
        result = max(result, min(heights[left], heights[right]) * (right - left))
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    return result


def three_sum(nums):
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


def letter_combinations_of_a_phone_number(digits):
    hashmap = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs',
               '8': 'tuv', '9': 'wxyz'}
    result, n = [], len(digits)

    def dfs(sofar, start):
        if start == n:
            result.append(sofar)
        else:
            letters = hashmap[digits[start]]
            for letter in letters:
                dfs(sofar + letter, start + 1)

    dfs(sofar='', start=0)
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


def next_permutation():
    pass


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


def combination_sum(candidates, target):
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
    hashmap = collections.defaultdict(list)
    for word in strs:
        letter_count_map = [0] * 26
        for char in word:
            letter_count_map[ord(char) - ord('a')] += 1
        hashmap[tuple(letter_count_map)].append(word)
    return hashmap.values()


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
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[-1]


def edit_distance(word1, word2):
    # https://leetcode.com/problems/edit-distance/discuss/159295/Python-solutions-and-intuition
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


