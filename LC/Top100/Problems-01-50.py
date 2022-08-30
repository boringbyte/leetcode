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
        carry, rem = divmod(carry, 10)
        current.next = ListNode(rem)
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
        return s[l+1: r]

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
    pass