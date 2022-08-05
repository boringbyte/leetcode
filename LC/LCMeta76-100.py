import collections
import random
from functools import lru_cache


def remove_adjacent_two_duplicates(s):
    stack = []
    for char in s:
        if stack and char == stack[-1]:
            stack.pop()
        else:
            stack.append(char)
    return ''.join(stack)


def word_ladder_length(begin_word, end_word, word_list):
    if end_word not in word_list or not begin_word \
            or not end_word or not word_list \
            or len(begin_word) != len(end_word):
        return 0
    n, hashmap = len(begin_word), collections.defaultdict(list)
    for word in word_list:
        for i in range(n):
            hashmap[word[:i] + '*' + word[i+1:]].append(word)

    queue, visited = collections.deque([(begin_word, 1)]), {[begin_word]}
    while queue:
        current_word, level = queue.popleft()
        for i in range(n):
            intermediate_word = current_word[:i] + '*' + current_word[i+1:]
            for word in hashmap[intermediate_word]:
                if word == end_word:
                    return level+1
                if word not in visited:
                    visited.add(word)
                    queue.append((word, level+1))
    return 0


def two_sum(nums, target):
    hashmap = {num: i for i, num in enumerate(nums)}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in hashmap and i != hashmap[diff]:
            return [i, hashmap[diff]]


def max_area(grid):
    m, n, result = len(grid), len(grid[0]), 0
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(x, y, area):
        if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
            area, grid[x][y] = area + 1, 0
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                area = dfs(nx, ny, area)
        return area

    for x in range(m):
        for y in range(n):
            result = max(result, dfs(x, y, 0))
    return result


def find_peak_linear_scan(nums):
    n = len(nums)
    for i in range(n - 1):
        if nums[i] > nums[i + 1]:
            return i
    return n - 1


def find_peak_binary_search(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < nums[mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return lo


def max_profit(prices):
    current_max, result = 0, 0
    for i in range(1, len(prices)):
        current_max += prices[i] - prices[i - 1]
        if current_max < 0:
            current_max = 0
        result = max(current_max, result)
    return result


def daily_temperatures(temperatures):
    result, stack = [0] * len(temperatures), []
    for idx, temp in enumerate(temperatures):
        while stack and temp > temperatures[stack[-1]]:
            result[stack[-1]] = idx - stack[-1]
            stack.pop()
        stack.append(idx)
    return result


def valid_parentheses(s):
    if len(s) % 2 == 1:
        return False

    hashmap, stack = {'(': ')', '[': ']', '{': '}'}, []
    for char in s:
        if char in hashmap:
            stack.append(char)
        else:
            if stack and hashmap[stack[-1]] == char:
                stack.pop()
            else:
                return False
    return len(stack) == 0


def minimum_add_to_make_parentheses_valid1(s):
    left, right = 0, 0
    for char in s:
        if char == '(':
            right += 1
        elif right > 0:
            right -= 1
        else:
            left += 1
    return left + right


def minimum_add_to_make_parentheses_valid2(s):
    right, stack = 0, []
    for char in s:
        if char == '(':
            stack.append(char)
        elif stack and char == ')':
            stack.pop()
        else:
            right += 1
    return right + len(stack)


def remove_nth_from_end1(head, n):
    length, current = 0, head
    while current:
        current = current.next
        length += 1

    current = head
    for _ in range(1, length - n):
        current = current.next

    current.next = current.next.next
    return head


def remove_nth_from_end2(head, n):
    slow = fast = head
    for _ in range(n):
        fast = fast.next
    if not fast:
        return head.next
    while fast.next:
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next
    return slow


class RandomizedSet:
    def __init__(self):
        self.nums, self.pos = [], {}

    def insert(self, val):
        if val not in self.pos:
            self.pos[val] = len(self.nums)
            self.nums.append(val)
            return True
        return False

    def remove(self, val):
        if val in self.pos:
            idx, last = self.pos[val], self.nums[-1]
            self.nums[idx], self.pos[last] = last, idx
            self.nums.pop()
            self.pos.pop(val, 0)
            return True
        return False

    def get_random(self):
        return random.choice(self.nums)


def multiply(num1, num2):
    m, n = len(num1), len(num2)
    result = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            value = (ord(num1[i]) - ord('0')) * (ord(num2[j]) - ord('0'))
            p1, p2 = i + j, i + j + 1
            total = value + result[p2]
            result[p1] += total // 10
            result[p2] = total % 10
    sb = []
    for value in result:
        if len(sb) != 0 or value != 0:
            sb.append(str(value))
    return '0' if len(sb) == 0 else ''.join(sb)


def add_operators(num, target):
    pass


def longest_increasing_path_matrix_dfs(matrix):
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    m, n, result = len(matrix), len(matrix[0]), 0

    @lru_cache
    def dfs(x, y):
        local_result = 1
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[x][y]:
                local_result = max(local_result, dfs(nx, ny) + local_result)
        return local_result

    for x in range(m):
        for y in range(n):
            result = max(result, dfs(x, y))
    return result
