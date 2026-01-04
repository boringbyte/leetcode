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
    """
    Imagine you're quality control in a sorted factory assembly line.
    Your job is to compact identical items and report how many unique items remain.
        Conveyor belt: nums (sorted array of items)
        Quality Inspector (left pointer): Marks where the next unique item should go
        Scanner (right pointer): Moves down the belt checking each item
    """
    left, n = 0, len(nums)              # Inspector at position 0
    for right in range(1, n):           # Scanner starts at position 1
        if nums[left] != nums[right]:   # Found a NEW unique item!
            left += 1                   # Inspector moves to next spot
            nums[left] = nums[right]    # Place unique item here
        # If items are the same, scanner just keeps moving. Inspector stays put, waiting for something unique
    return left + 1                     # Number of unique items = position + 1


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
    """
    Mountain Peak Analogy:
    Imagine each number in the list as a mountain peak's height. To find the next arrangement (permutation), we:

    1. Find the PIVOT - the first peak from the right that's lower than the peak to its right. This is where we can "climb higher."
       Example: [1, 3, 2] → Peaks: 1─3─2
                From right: 2→3 (climbing), 3→1 (descending!)
                Pivot is 1 (it wants to be higher)

    2. Find the SUCCESSOR - the smallest peak to the right of the pivot that's taller than the pivot. This is who the pivot swaps with.
       Right of pivot 1: [3, 2] → 2 is the smallest taller than 1

    3. SWAP the pivot with the successor.
       [1, 3, 2] → swap 1 and 2 → [2, 3, 1]

    4. REVERSE the "downhill" (suffix) - everything after the pivot's original position is reversed to make it the smallest possible arrangement.
       After swap: [2, 3, 1] → suffix [3, 1] reversed to [1, 3]
       Final: [2, 1, 3] (the next mountain arrangement)

    Special Case: If the entire range is descending (like [3, 2, 1]), there's no pivot.
    We reverse the whole array to get the smallest permutation.

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    n = len(nums)
    i = n - 2  # Start from second last digit

    # 1. Find the "pivot" - first digit that's smaller than its right neighbor
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i >= 0:  # If we found a pivot (not already the largest permutation)
        j = n - 1
        # 2. Find the "successor" - smallest digit to the right that's larger than pivot
        while nums[i] >= nums[j]:
            j -= 1

        # 3. Swap pivot and successor
        nums[i], nums[j] = nums[j], nums[i]

    # 4. Reverse the suffix (everything after the pivot)
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
    num1, num2 = num1[::-1], num2[::-1]                 # Reverse to multiply from RIGHT to LEFT

    for i in range(m):
        for j in range(n):
            mul = int(num1[i]) * int(num2[j])           # Multiply single digits (like 3 × 4 = 12)
            result[i + j] += mul                        # Add to the correct "place value" position

            carry, digit = divmod(result[i + j], 10)    # Handle carrying over (like carrying the 1 from 12)
            result[i + j] = digit                       # Store the ones digit in current position
            result[i + j + 1] += carry                  # Carry over to next column

    # Remove leading zeros (extra blank spaces we didn't use)
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
