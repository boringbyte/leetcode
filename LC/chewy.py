# Simple linked list related problem with a small twist
import collections


def insert_in_sorted_linked_list(head, node):
    # https://www.geeksforgeeks.org/given-a-linked-list-which-is-sorted-how-will-you-insert-in-sorted-way/
    current = node
    if not head:
        node.next = None
    elif current.val >= node.val:
        node.next = current
    else:
        while current and current.val < node.val:
            current = current.next
        node.next = current.next
        current.next = node
    head = node
    return head


def shifting_letters(s, shifts):
    # https://leetcode.com/problems/shifting-letters/discuss/?currentPage=1&orderBy=most_votes&query=
    n, result = len(s), ''
    for i in range(n - 2, -1, -1):
        shifts[i] = (shifts[i] + shifts[i + 1]) % 26

    for i, char in enumerate(s):
        idx = (ord(char) - ord('a') + shifts[i]) % 26
        result += chr(idx + ord('a'))
    return result


def distance_nearest_cell_1_binary_matrix(matrix):
    # https://www.geeksforgeeks.org/distance-nearest-cell-1-binary-matrix/
    # https://leetcode.com/problems/01-matrix/solution/
    m, n, directions = len(matrix), len(matrix[0]), [(1, 0), (0, 1), (-1, 0), (0, -1)]
    queue = collections.deque()

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                queue.append((i, j))
            else:
                matrix[i][j] = -1

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx == m or ny < 0 or ny == n or matrix[nx][ny] != -1:
                continue
            matrix[nx][ny] = matrix[x][y] + 1
            queue.append((nx, ny))
    return matrix


def is_subset_sum(nums, target):
    # https://www.techiedelight.com/subset-sum-problem/
    # https://www.geeksforgeeks.org/subset-sum-problem-dp-25/
    last_idx, cache = len(nums) - 1, {}

    def recursive(i, total):
        if total == 0:
            return True
        if i < 0 or total < 0:
            return False
        key = (i, total)
        if key not in cache:
            include = recursive(i - 1, total - nums[i])
            exclude = recursive(i - 1, total)
            cache[key] = include or exclude
        return cache[key]
    return recursive(last_idx, target)


def flatten_array(nums):
    # Implement array flat method.
    result = []

    def recursive(num):
        for el in num:
            if not isinstance(el, list):
                result.append(el)
            else:
                recursive(el)
    recursive(nums)
    return result


def check_anagrams1(word1, word2):
    # check if words are anagrams
    return sorted(word1) == sorted(word2)


def check_anagrams2(word1, word2):

    def char_map(word):
        order_map = [0] * 26
        mapping = {char: ord(char) - ord('a') for char in word}
        for char in word:
            order_map[mapping[char]] += 1
        return ''.join([str(char) for char in order_map])

    return char_map(word1) == char_map(word2)


def robot_exit_maze(maze: list[list], entrance: list):
    # Write a program to help a robot find the exit to a maze.
    # https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/
    m, n = len(maze), len(maze[0])
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    queue = collections.deque([(*entrance, 0)])
    maze[entrance[0]][entrance[1]] = '+'

    while queue:
        x, y, steps = queue.popleft()
        if (x == 0 or x == m - 1 or y == 0 or y == n - 1) and [x, y] != entrance:
            return steps
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and maze[nx][ny] == '.':
                maze[nx][ny] = '+'
                queue.append((nx, ny, steps+1))
    return -1


def group_count_anagrams(words, sentences):
    # https://leetcode.com/discuss/interview-question/1541093/How-many-sentences-Can-someone-provide-a-python-solution-for-this-question
    word_map = collections.defaultdict(int)
    for word in words:
        word = tuple(sorted(word))
        word_map[word] += 1

    result = [1] * len(sentences)

    for i, s in enumerate(sentences):
        for word in s.split():
            key = tuple(sorted(word))
            if key in word_map:
                result[i] *= word_map[key]
    return result


def minimum_deletions_to_make_character_frequencies_unique(s):
    # https://leetcode.com/problems/minimum-deletions-to-make-character-frequencies-unique/
    # https://leetcode.com/problems/minimum-deletions-to-make-character-frequencies-unique/discuss/928137/Python-Best-Simple-Solution
    counter_dict = collections.Counter(s)
    counter_set, result = set(), 0

    for key, val in counter_dict.items():
        while key > 0 and key not in counter_set:
            key -= 1
            result += 1
        counter_set.add(key)
    return result
