from collections import deque
from unittest import result

from leetcode.utils import ListNode


def odd_even_linked_list(head):
    # https://leetcode.com/problems/odd-even-linked-list
    if not head or not head.next:
        return head

    odd = head
    even = head.next
    even_dummy = head.next

    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next

    odd.next = even_dummy.next
    return head


def generate_parentheses(n):
    # https://leetcode.com/problems/generate-parentheses
    result = []

    def backtrack(sofar, left, right):
        if len(sofar) == 2 * n:
            result.append(sofar)
        else:
            if left < n:
                backtrack(sofar + "(", left + 1, right)
            elif right < left:
                backtrack(sofar + ")", left, right + 1)


    backtrack(sofar="", left=0, right=0)
    return result


def populating_next_right_pointers_in_each_node(root):
    # https://leetcode.com/problems/populating-next-right-pointers-in-each-node
    if not root:
        return root

    queue = deque([root])

    while queue:
        right_node = None

        for _ in range(len(queue)):
            current = queue.popleft()
            current.next = right_node
            right_node = current

            if current.right:
                queue.append(current.right)
            if current.left:
                queue.append(current.left)

    return root


def binary_tree_zigzag_level_order_traversal(root):
    # https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal
    if not root:
        return []

    result = []
    queue = deque([root])
    direction = -1

    while queue:
        direction *= -1
        level_values = []

        for _ in range(len(queue)):
            current = queue.popleft()
            level_values.append(current.val)

            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)

        result.append(level_values[::direction])

    return result


def reverse_word_in_a_string(s):
    # https://leetcode.com/problems/reverse-words-in-a-string
    # https://leetcode.com/problems/reverse-words-in-a-string/solutions/1531693/c-2-solutions-o1-space-picture-explain-c-ky57/
    # words = s.split()
    # words = words[::-1]
    # return " ".join(words)
    def reverse(s, left, r):
        while left < r:
            s[left], s[r] = s[r], s[left]
            left += 1
            r -= 1

    s = list(s)                 # each character is an element in the list
    s.reverse()
    i, k, n = 0, 0, len(s)

    while i < n:
        # Find the starting position of the next word
        while i < n and s[i] == " ":
            i += 1

        if i != n and k > 0:
            s[k] = " "
            k += 1

        start_index = k
        while i < n and s[i] != " ":
            s[k] = s[i]
            i += 1
            k += 1

        reverse(s, start_index, k - 1)

    s = s[:k]
    return "".join(s)
