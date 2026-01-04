from leetcode.utils import ListNode


def two_sum(nums, target):
    # https://leetcode.com/problems/two-sum
    diff_dict = {num: i for i, num in enumerate(nums)}
    for i, num in enumerate(nums):
        diff = target - num
        j = diff_dict[diff]
        if diff in diff_dict and i != j:
            return [i, j]


def add_two_numbers(l1, l2):
    # https://leetcode.com/problems/add-two-numbers
    if l1 and l2:
        carry = 0
        head = current = ListNode()
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
        return head.next
    else:
        return l1 or l2


def longest_substring_without_repeating_characters(s):
    # https://leetcode.com/problems/longest-substring-without-repeating-characters
    if len(s) <= 1:
        return len(s)

    result = left = 0
    seen_dict = {}

    for right, char in enumerate(s):
        if char in seen_dict:
            left = max(left, seen_dict[char] + 1)
        result = max(result, left - right + 1)
        seen_dict[char] = right
    return result


def median_of_two_sorted_arrays_1(nums1, nums2):
    # https://leetcode.com/problems/median-of-two-sorted-arrays/solutions/4070500/99journey-from-brute-force-to-most-optim-z3k8/
    # This is a two pointer solution which is less efficient
    n, m = len(nums1), len(nums2)
    total_len = n + m
    i, j = 0, 0
    prev, curr = 0, 0

    for _ in range(total_len // 2 + 1):
        prev = curr
        if i < n and (j >= m or nums1[i] <= nums2[j]):
            curr = nums1[i]
            i += 1
        else:
            curr = nums2[j]
            j += 1

    if total_len % 2 == 1:
        return float(curr)
    else:
        return (prev + curr) / 2.0


def median_of_two_sorted_arrays_2(nums1, nums2):
    # https://leetcode.com/problems/median-of-two-sorted-arrays
    if len(nums1) > len(nums2):
        median_of_two_sorted_arrays_2(nums2, nums1)

    m, n = len(nums1), len(nums2)
    low, high = 0, m

    while low <= high:
        i = (low + high) // 2
        j = (m + n + 1) // 2 - i

        # Edge values (treat out-of-bounds as +- infinity)
        left1  = nums1[i - 1] if i > 0 else float('-inf')
        right1 = nums1[i]     if i < m else float('inf')

        left2  = nums2[j - 1] if j > 0 else float('-inf')
        right2 = nums2[j]     if j < n else float('inf')

        # Found correct partition
        if left1 <= right2 and left2 <= right1:
            # Odd total length → median = max of left
            if (m + n) % 2 == 1:
                return max(left1, left2)
            # Even length → avg of max(left) + min(right)
            return (max(left1, left2) + min(right1, right2)) / 2

        # Need to move partition left
        elif left1 > right2:
            high = i - 1

        # Need to move partition right
        else:
            low = i + 1


def longest_palindromic_substring(s):
    # https://leetcode.com/problems/longest-palindromic-substring
    n = len(s)
    result = s[0]   # 's' at least contains 1 character
    max_length = 1  # As 's' at least contains 1 character, maximum length is 1

    def expand_around(left, right):
        nonlocal result, max_length
        while left >= 0 and right < n and s[left] == s[right]:
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
                result = s[left: right + 1]
            left -= 1
            right += 1

    for i in range(n):
        expand_around(left=i, right=i)     # odd-length palindromes
        expand_around(left=i, right=i + 1) # even-length palindromes

    return result


def string_to_integer(s):
    # https://leetcode.com/problems/string-to-integer-atoi
    INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1
    num, sign, started = 0, 1, False

    for char in s:
        if char == " " and not started:
            continue
        elif char in "+-" and not started:
            started = True
            if char == "-":
                sign = -1
        elif char.isdigit():
            started = True
            digit = int(char)
            if num > INT_MAX // 10 or (num == INT_MAX // 10 and digit > INT_MAX % 10):
                if sign == 1:
                    return INT_MAX
                else:
                    return INT_MIN
            num = num * 10 + digit
        else:
            break

    return num * sign


def palindrome_number(x):
    # https://leetcode.com/problems/palindrome-number
    if x < 0:
        return False

    reverse = 0
    original = x
    while x > 0:
        last_digit = x % 10     # Access the last digit
        reverse = reverse * 10 + last_digit
        x = x // 10             # Delete the last digit from x

    return original == reverse


def longest_common_prefix(strs):
    # https://leetcode.com/problems/longest-common-prefix
    if not strs:  # This condition might not be necessary as there is at least 1 string in strs list
        return ''
    shortest = min(strs, key=len)
    for i, char in enumerate(shortest):
        for word in strs:
            if word[i] != char:
                return shortest[:i]  # [:i] handles when there is no match like strs = ["dog", "racecar", "car"]
    return shortest


def three_sum(nums):
    # https://leetcode.com/problems/3sum
    """
    Need to find numbers that satisfy following 4 conditions:
        1. All the 3 numbers are zero
        2. 1 is zero, 1 is positive and 1 is negative
        3. 2 are negative and 1 is positive
        4. 2 are positive and 1 is negative
    """
    result = set()

    n_list = [n for n in nums if n < 0]
    z_list = [n for n in nums if n == 0]
    p_list = [n for n in nums if n > 0]
    n_set, p_set = set(n_list), set(p_list)

    # Condition 1:
    if len(z_list) >= 3:
        result.add((0, 0, 0))

    # Condition 2:
    if z_list:
        for num in p_set:
            if -num in n_set:
                result.add((-num, 0, num))

    k = len(n_list)

    # Condition 3:
    for i in range(k):
        for j in range(i + 1, k):
            target = -1 * (n_list[i] + n_list[j])
            if target in p_set:
                result.add((n_list[i], n_list[j], target))

    k = len(p_list)

    # Condition 4:
    for i in range(k):
        for j in range(i + 1, k):
            target = -1 * (p_list[i] + p_list[j])
            if target in n_set:
                result.add((p_list[i], p_list[j], target))

    return [list(t) for t in result]


def letter_combinations_of_a_phone_number(digits):
    # https://leetcode.com/problems/letter-combinations-of-a-phone-number
    mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
    result, n = [], len(digits)

    def backtrack(sofar, start):
        if len(sofar) == n:
            result.append(sofar)
        else:
            for i in range(start, n):
                letters = mapping[digits[i]]
                for chosen in letters:
                    # As sofar is a string, a new copy is sent to recursive function,
                    # and we don't have to explicitly remove chosen from the sofar string.
                    backtrack(sofar + chosen, i + 1)

    backtrack(sofar='', start=0)
    return result if digits else []
