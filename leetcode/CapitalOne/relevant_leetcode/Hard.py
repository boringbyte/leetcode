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
    result, line, num_of_chars = [], [], 0

    for word in words:
        # Check if adding this word + minimal spaces would overflow
        # num_of_chars measures the length of words already present in the line list
        # len(word) measures the length of word if it is added to the current list
        # len(line) is for number of spaces
        if num_of_chars + len(word) + len(line) > max_width:
            # Finalize current line by distributing spaces
            # max_width - number of actual characters without any spaces
            spaces_to_add = max_width - num_of_chars
            # If there are multiple words, gaps = len(line) - 1.
            # If there’s only one word in the line, len(line) - 1 is 0, so we use or 1 to avoid division by zero; this puts all spaces after the only word (i.e., left-justified).
            gaps = len(line) - 1 or 1
            for i in range(spaces_to_add):
                # This adds spaces to the end of the end of each word in the list expect the last word
                line[i % gaps] += ' '
            result.append(''.join(line))
            # Reset for the next line
            line, num_of_chars = [], 0

        # Add the word to the (new or continuing) line
        line.append(word)
        num_of_chars += len(word)

    # Last line: left-justify
    result.append(' '.join(line).ljust(max_width))
    return result


def reverse_nodes_in_k_group(head, k):
    # https://labuladong.gitbook.io/algo-en/iv.-high-frequency-interview-problem/reverse-nodes-in-k-group
    # https://leetcode.com/problems/reverse-nodes-in-k-group/solutions/4335870/easy-solution

    # Check if we have k nodes available
    count = 0
    current = head
    while current and count < k:
        current = current.next
        count += 1

    if count < k:
        return head

    # Reverse the first k nodes
    prev_node, current_node = None, head
    for _ in range(k):
        next_node = current_node.next  # save the next node
        current_node.next = prev_node  # reverse pointer
        prev_node = current_node  # move prev forward
        current_node = next_node  # move curr forward

    # Recursively reverse the remaining groups
    head.next = reverse_nodes_in_k_group(current_node, k)
    # Return new head of this group
    return prev_node


def largest_rectangle_in_histogram(heights):
    # https://leetcode.com/problems/largest-rectangle-in-histogram/solutions/1727641/python3-monotonic-stack-t-t-explained

    stack, result = [], 0  # Stack contains both the width and height and The stack maintains increasing heights.
    for height in heights + [-1]:  # The -1 ensures the stack gets completely emptied at the end (all rectangles get processed).
        width = 0
        while stack and stack[-1][1] >= height: # If the current height is smaller than or equal to the height on top of the stack
            w, h = stack.pop()
            width += w
            result = max(result, width * h)

        stack.append((width + 1, height)) # This bar itself contributes width 1.

    return result


class TrieNode2:
    def __init__(self):
        self.children = defaultdict(TrieNode2)
        self.word = None   # store the complete word at the end node


def word_search_2(board, words):
    pass


def block_placement_queries(queries):
    pass


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
        reversed_word = word[::-1]

        node = root
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