from leetcode.utils import ListNode
from collections import defaultdict


def valid_parentheses(s):
    # https://leetcode.com/problems/valid-parentheses
    if len(s) % 2:
        return False

    mapping = {"(": ")", "[": "]", "{": "}"}
    stack = []

    for char in s:
        if char in mapping:
            stack.append(char)
        else:
            if stack and mapping[stack[-1]] == char:
                stack.pop()
            else:
                return False
    return len(stack) == 0


def merge_two_sorted_lists(list1, list2):
    if list1 and list2:
        current = head = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                current.next = ListNode(list1.val)
                list1 = list1.next
            else:
                current.next = ListNode(list2.val)
                list2 = list2.next
            current = current.next

        current.next = list1 or list2
        return head.next
    else:
        return list1 or list2


def merge_k_sorted_lists(lists):
    # https://leetcode.com/problems/merge-k-sorted-lists
    if lists:
        if len(lists) == 1:
            return lists[0]
        mid = len(lists) // 2
        left = merge_k_sorted_lists(lists[:mid])
        right = merge_k_sorted_lists(lists[mid:])
        return merge_two_sorted_lists(left, right)
    else:
        return None   # Is it really None. For [] output should be [].


def remove_duplicates_from_sorted_array(nums):
    # https://leetcode.com/problems/remove-duplicates-from-sorted-array
    left, n = 0, len(nums)
    for right in range(1, n):
        if nums[left] != nums[right]:
            left += 1
            nums[left] = nums[right]
    return left + 1


def max_consecutive_ones_iii(nums, k):
    # https://leetcode.com/problems/max-consecutive-ones-iii
    n, result, left, zeros = len(nums), 0, 0, 0
    for right in range(n):
        if nums[right] == 0:
            zeros += 1
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        result = max(result, right - left + 1)
    return result


def next_permutation(nums):
    # https://leetcode.com/problems/next-permutation
    n = len(nums)
    i = n - 2

    # 1. Find first decreasing element from the right
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i >= 0:  # Found a pivot
        j = n - 1
        # 2. Find next greater element
        while nums[i] >= nums[j]:
            j -= 1

        # 3. Swap numbers at i and j
        nums[i], nums[j] = nums[j], nums[i]

    # 4. Reverse suffix
    nums[i + 1:] = reversed(nums[i + 1:])


def search_in_rotated_sorted_array(nums, target):
    # https://leetcode.com/problems/search-in-rotated-sorted-array
    """
    Key idea:
    - Even though the array is rotated, at least ONE HALF is always sorted.
    - At every step, decide:
        1) Which half is sorted?
        2) Is the target inside that sorted half?
        3) If yes → shrink toward it, else → go to the other half.

    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


def find_first_and_last_position_of_element_in_sorted_array(nums, target):
    # https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/
    result = [-1, -1]
    if not nums:
        return result

    n = len(nums)
    left, right = 0, n - 1
    while left < right:
        mid = left + (right - left) // 2
        if target > nums[mid]:
            left = mid + 1
        else:
            right = mid

    if nums[left] != target:
        return result
    result[0] = left

    left, right = 0, n - 1
    while left < right:
        mid = left + (right - left + 1) // 2
        if target >= nums[mid]:
            left = mid
        else:
            right = mid - 1
    result[1] = left

    return result


def trapping_rain_water(height):
    # https://leetcode.com/problems/trapping-rain-water
    # Dynamic Programming Version
    # This condition might be unnecessary as per the provided constraints in the problem.
    if height is None or len(height) == 0:
        return 0

    n = len(height)
    left, right = [0] * n, [0] * n
    left[0], right[-1] = max(0, height[0]), max(0, height[-1])
    for i in range(1, n):
        left[i] = max(height[i], left[i - 1])

    for i in range(n - 2, -1, -1):
        right[i] = max(height[i], right[i + 1])

    result = 0
    for i in range(n):
        result += min(left[i], right[i]) - height[i]

    return result


def multiply_strings(num1, num2):
    # https://leetcode.com/problems/multiply-strings
    if num1 == "0" or num2 == "0":
        return 0

    m, n = len(num1), len(num2)
    result = [0] * (m + n)
    num1, num2 = num1[::-1], num2[::-1]

    for i in range(m):
        for j in range(n):
            mul = int(num1[i]) * int(num2[j])
            result[i + j] += mul
            carry, digit = divmod(result[i + j], 10)
            result[i + j + 1] += carry
            result[i + j] = digit

    while result[-1] == 0:
        result.pop()

    return "".join(map(str, result[::-1]))


def group_anagrams(strs):
    # https://leetcode.com/problems/group-anagrams
    anagram_dict = defaultdict(list)

    def convert_word_to_binary_string(word):
        counts = [0] * 26
        for char in word:
            counts[ord(char) - ord("a")] += 1
        return "".join(map(str, counts)) # or tuple(counts)

    for s in strs:
        key = convert_word_to_binary_string(s)
        anagram_dict[key].append(s)

    return list(anagram_dict.values())
