
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def best_time_to_buy_and_sell_stock(prices):
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock/solutions/5501275/video-keep-minimum-price-solution/
    buy_price, profit = prices[0], 0

    for price in prices[1:]:
        if price < buy_price:
            buy_price = price
        profit = max(profit, price - buy_price)
    return profit


def valid_parenthesis(s):
    if len(s) % 2 == 1:
        return False

    stack, hashmap = [], {'{': '}', '(': ')', '[': ']'}

    for char in s:
        if char in hashmap:
            stack.append(char)
        else:
            if stack and hashmap[stack[-1]] == char:
                stack.pop()
            else:
                return False
    return len(stack) == 0


def merge_two_sorted_lists(list1, list2):
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

        if list1 or list2:
            current.next = list1 or list2

        return head.next

    else:
        return list1 or list2


def roman_to_integer(s):
    result, symbols, n = 0, {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}, len(s)

    for i in range(n):
        if i + 1 < n and symbols[s[i]] < symbols[s[i + 1]]:
            result -= symbols[s[i]]
        else:
            result += symbols[s[i]]
    return result


def palindrome_number(x):
    if x < 0:
        return False

    div = 1
    while x // div >= 10:
        div *= 10   # Gets a number such that for 1221 we get div as 1000

    while x:
        left, rem = divmod(x, div)
        right = x % 10

        if left != right:
            return False

        x = rem % 10  # Remove right digit as left was already removed
        div //= 100   # As two digits are getting removed, we are using 100 here

    return True


def add_strings(num1, num2):

    def string_to_digit(val):
        return ord(val) - ord('0')

    i, j, carry, result = len(num1) - 1, len(num2) - 1, 0,[]

    while i >= 0 or j >= 0 or carry:
        digit1 = digit2 = 0
        if i >= 0:
            digit1 = string_to_digit(num1[i])
        if j >= 0:
            digit2 = string_to_digit(num2[j])
        carry, digit = divmod(digit1 + digit2 + carry, 10)
        result.append(str(digit))
        i, j = i - 1, j - 1
    return ''.join(result[::-1])


def binary_tree_paths(root):
    result = []

    def dfs(node, path):
        if node:
            path += str(node.val)
            if not node.left and not node.right:
                result.append(path)
            else:
                path += '->'
                dfs(node.left, path)
                dfs(node.right, path)

    dfs(root, '')
    return result


def count_operations_to_obtain_zero(num1, num2):
    result = 0

    while num1 and num2:
        if num1 >= num2:
            num1 = num1 - num2
            result += 1
        else:
            num2 = num2 - num1
            result += 1

    return result


def count_prefix_and_suffix_pairs_1(words):
    def is_prefix_and_suffix(str1, str2):
        n1, n2 = len(str1), len(str2)
        if n1 > n2:
            return False
        return str2[:n1] == str1 and str2[-n1:] == str1

    n, result = len(words), 0
    for i in range(n):
        for j in range(i + 1, n):
            if is_prefix_and_suffix(words[i], words[j]):
                result += 1
    return result
