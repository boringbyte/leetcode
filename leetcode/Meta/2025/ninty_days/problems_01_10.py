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
            carry, digit =divmod(carry, 10)
            current.next = ListNode(digit)
            current = current.next
        return head.next
    else:
        return l1 or l2


def longest_substring_without_repeating_characters(s):
    # https://leetcode.com/problems/longest-substring-without-repeating-characters
    n = len(s)
    result = right = 0
    seen_dict = {}
    if n <= 1:
        return n

    for left, char in enumerate(s):
        if char in seen_dict:
            left = max(left, seen_dict[char] + 1)
        result = max(result, left - right + 1)
        seen_dict[char] = right
    return result


def median_of_two_sorted_arrays(nums1, nums2):
    # https://leetcode.com/problems/median-of-two-sorted-arrays
    pass


def longest_palindromic_substring(s):
    # https://leetcode.com/problems/longest-palindromic-substring
    n = len(s)
    result = s[0]
    max_length = 1

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
        expand_around(i, i)     # odd-length palindromes
        expand_around(i, i + 1) # even-length palindromes

    return result


def string_to_integer(s):
    # https://leetcode.com/problems/string-to-integer-atoi
    INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1
    num, sign, started = 0, 1, False

    for char in s:
        if char == " " and not started:
            continue
        elif char in "+-" and not started:
            if char == "-":
                sign = -1
            started = True
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
        digit = x % 10
        reverse = reverse * 10 + digit
        x = x // 10

    return original == reverse


def longest_common_prefix(strs):
    # https://leetcode.com/problems/longest-common-prefix
    if not strs:
        return ''
    shortest = min(strs, key=len)
    for i, char in enumerate(shortest):
        for word in strs:
            if word[i] != char:
                return shortest[:i]
    return shortest


def three_sum(nums):
    # https://leetcode.com/problems/3sum
    result = set()

    n_list = [n for n in nums if n < 0]
    p_list = [n for n in nums if n > 0]
    z_list = [n for n in nums if n == 0]
    n_set, p_set = set(n_list), set(p_list)

    if len(z_list) >= 3:
        result.add((0, 0, 0))

    if z_list:
        for num in p_set:
            if -num in n_set:
                result.add((-num, 0, num))

    k = len(n_list)
    for i in range(k):
        for j in range(i + 1, k):
            target = -1 * (n_list[i] + n_list[j])
            if target in p_set:
                result.add((n_list[i], n_list[j], target))

    k = len(p_list)
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
