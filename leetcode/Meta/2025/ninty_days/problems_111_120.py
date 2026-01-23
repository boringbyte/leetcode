import heapq
from collections import deque
from unittest import result

from leetcode.utils import ListNode, TreeNode


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


def find_kth_smallest_element_in_a_sorted_matrix(matrix, k):
    # https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix
    # https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/solutions/1322101/cjavapython-maxheap-minheap-binary-searc-3hgs/
    m, n = len(matrix), len(matrix[0])
    max_heap = []

    for row in range(m):
        for col in range(n):
            heapq.heappush(max_heap, -matrix[row][col])
            if len(max_heap) > k:
                heapq.heappop(max_heap)

    return -heapq.heappop(max_heap)


class MedianFinder:
    # https://leetcode.com/problems/find-median-from-data-stream

    def __init__(self):
        pass

    def add_num(self, num):
        pass

    def find_median(self):
        pass


def construct_binary_tree_from_string(s):
    # https://leetcode.com/problems/construct-binary-tree-from-string
    # https://algo.monster/liteproblems/536
    # https://github.com/doocs/leetcode/blob/main/solution/0500-0599/0536.Construct%20Binary%20Tree%20from%20String/README_EN.md
    def dfs(s):
        if not s:
            return None

        first_open_parenthesis_index = s.find("(")
        if first_open_parenthesis_index == -1:
            return TreeNode(int(s))                                     # No parenthesis found, so it is a leaf node

        node_value = int(s[:first_open_parenthesis_index])              # Everything before the parentheses
        node = TreeNode(node_value)

        start_index = first_open_parenthesis_index
        parentheses_count = 0

        for current_index in range(first_open_parenthesis_index, len(s)):
            current_char = s[current_index]

            if current_char == "(":
                parentheses_count += 1
            elif current_char == ")":
                parentheses_count -= 1

            if parentheses_count == 0:
                if start_index == first_open_parenthesis_index:
                    node.left = TreeNode(int(s[start_index + 1 : current_index]))
                    start_index = current_index + 1
                else:
                    node.right = TreeNode(int(s[start_index + 1 : current_index]))

        return node

    return dfs(s)



def merge_two_sorted_lists(list1, list2):
    # https://leetcode.com/problems/merge-two-sorted-lists
    if list1 and list2:
        head = current = ListNode()

        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next

            current = current.next

            if list1 or list2:                          # This check is not mandatory
                current.next = list1 or list2

        return head.next
    else:
        return list1 or list2


def basic_calculator(s):
    # https://leetcode.com/problems/basic-calculator
    pass