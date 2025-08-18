from functools import lru_cache
from collections import defaultdict


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.count = 0


def text_justification(words, max_width):
    result, line, width = [], [], 0

    for word in words:
        if width + len(word) + len(line) > max_width:
            for i in range(max_width - width):
                line[i % (len(line) - 1 or 1)] += ' '
                result, line, width = result + [''.join(line)], [], 0
            line += [word]
            width += len(word)

    return result + [' '.join(line).ljust(max_width)]


def reverse_nodes_in_k_group(head, k):
    # https://labuladong.gitbook.io/algo-en/iv.-high-frequency-interview-problem/reverse-nodes-in-k-group
    # https://leetcode.com/problems/reverse-nodes-in-k-group/solutions/4335870/easy-solution

    count = 0
    current = head
    while current and count < k:
        current = current.next
        count += 1

    if count < k:
        return head

    prev, curr = None, head
    for _ in range(k):
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt

    # Recursively reverse the remaining groups
    head.next = reverse_nodes_in_k_group(curr, k)
    return prev


def largest_rectangle_in_histogram(heights):
    # https://leetcode.com/problems/largest-rectangle-in-histogram/solutions/1727641/python3-monotonic-stack-t-t-explained

    stack, result = [], 0
    for height in heights + [-1]:
        width = 0
        while stack and stack[-1][1] >= height:
            w, h = stack.pop()
            width += w
            result = max(result, width * h)

        stack.append((width + 1, height))

    return result


def split_message_based_on_limit(message, limit):
    # https://leetcode.com/problems/split-message-based-on-limit/solutions/2807759/python-binary-search-is-redundant-just-brute-force-it-explained
    # Helper: number of digits in an integer
    def digits(n: int) -> int:
        return len(str(n))

    # Step 1: Find the minimal number of parts `p`
    p = 1  # number of parts
    extra_digits = 1  # total digits needed for indices (1,2,3,...p)

    while p * (digits(p) + 3) + extra_digits + len(message) > p * limit:
        # If suffix itself can't fit, return []
        if 3 + digits(p) * 2 >= limit:
            return []

        p += 1
        extra_digits += digits(p)

    # Step 2: Split the message
    parts = []
    for i in range(1, p + 1):
        # Characters available for actual message in this part
        avail_len = limit - (digits(p) + digits(i) + 3)

        part, msg = msg[:avail_len], msg[avail_len:]
        parts.append(f"{part}<{i}/{p}>")

    return parts


def count_prefix_and_suffix_pairs_2(words):
    root = TrieNode()
    result = 0

    for word in words:
        node = root
        reversed_word = word[::-1]

        for i in range(len(word)):
            key = (word[i], reversed_word[i])
            node = node.children[key]
            result += node.count

        node.count += 1
    return result


def remove_boxes(boxes):
    n = len(boxes)

    @lru_cache(None)
    def dp(l, r, k):
        if l > r:
            return 0

        # optimization: extend "k" if consecutive boxes are the same
        while l + 1 <= r and boxes[l] == boxes[l + 1]:
            l += 1
            k += 1

        # option 1: remove [l] directly with k extras
        res = (k + 1) * (k + 1) + dp(l + 1, r, 0)

        # option 2: merge with a future same-colored box
        for i in range(l + 1, r + 1):
            if boxes[i] == boxes[l]:
                res = max(res, dp(l + 1, i - 1, 0) + dp(i, r, k + 1))

        return res

    return dp(0, n - 1, 0)