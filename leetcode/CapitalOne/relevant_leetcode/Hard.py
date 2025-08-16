class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


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

