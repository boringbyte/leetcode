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