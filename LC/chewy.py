# Simple linked list related problem with a small twist
import collections


def shifting_letters(s, shifts):
    n, result = len(s), ''
    for i in range(n - 2, -1, -1):
        shifts[i] = (shifts[i] + shifts[i + 1]) % 26

    for i, char in enumerate(s):
        idx = (ord(char) - ord('a') + shifts[i]) % 26
        result += chr(idx + ord('a'))
    return result


# https://www.geeksforgeeks.org/distance-nearest-cell-1-binary-matrix/
# https://www.geeksforgeeks.org/subset-sum-problem-dp-25/
def is_subset_sum(nums, n, target):
    if target == 0:
        return True
    if n == 0:
        return False
    if nums[n - 1] > target:
        pass


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
        map = [0] * 26
        mapping = {char: ord(char) - ord('a') for char in word}
        for char in word:
            map[mapping[char]] += 1
        return ''.join([str(char) for char in map])

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
