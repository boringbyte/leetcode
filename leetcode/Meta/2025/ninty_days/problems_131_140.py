from collections import deque, defaultdict


def add_two_integers(num1, num2):
    # https://leetcode.com/problems/add-two-integers
    """
    Example:
    ---------
    Input:
        num1 = -2
        num2 = 3

    Binary (32-bit two's complement):
        -2 = 11111111 11111111 11111111 11111110
         3 = 00000000 00000000 00000000 00000011

    Step 1:
        temp_sum = num1 ^ num2
                 = 11111111 11111111 11111111 11111101
        carry    = (num1 & num2) << 1
                 = 00000000 00000000 00000000 00000100

    Step 2:
        temp_sum = 11111111 11111111 11111111 11111001
        carry    = 00000000 00000000 00000000 00001000

    Carry keeps moving left until it becomes 0.

    Final:
        num1 = 00000000 00000000 00000000 00000001
        => result = 1

    Output:
        1
    """
    MAX = 0x7FFFFFFF        # Maximum positive value for 32 bit integer
    MASK  = 0xFFFFFFFF      # Mask to handle overflow in negative numbers

    while num2 != 0:
        temp_sum = num1 ^ num2          # Calculate sum with carry
        carry = (num1 & num2) << 1      # Calculate the carry
        num1 = temp_sum & MASK          # Update num1 for next iteration and ensure the result is within 32-bit range
        num2 = carry & MASK             # Update num2 for next iteration

    return num1 if num1 <= MAX else ~(num1 ^ MASK)  # If num1 is negative, we need to handle the sign for 32 bit integer


def happy_number(n):
    # https://leetcode.com/problems/happy-number
    # https://leetcode.com/problems/happy-number/solutions/6750358/video-2-solutions-using-remainder-and-tw-bwks/
    visited = set()

    def get_next_number(num):
        squared_number = 0

        while num:
            last_digit = num % 10                               # Get the last digit from the number
            squared_number = squared_number + last_digit ** 2
            num = num // 10                                     # Get the remaining number after removing last digit

        return squared_number

    while n not in visited:
        visited.add(n)
        n = get_next_number(n)
        if n == 1:
            return True

    return False


def mine_sweeper(board, click):
    # https://leetcode.com/problems/minesweeper
    if board[click[0]][click[1]] == "M":
        board[click[0]][click[1]] = "X"
        return board

    m, n = len(board), len(board[0])
    queue = deque([click])
    visited = set(tuple(click))
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]

    while queue:
        for _ in range(len(queue)):
            x, y = queue.popleft()

            count_mine = 0
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if 0 <= nx < m and 0 <= ny < n:
                    if board[nx][ny] == "M":
                        count_mine += 1

            if count_mine > 0:
                board[x][y] = str(count_mine)
            else:
                board[x][y] = "B"
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited:
                        queue.append((nx, ny))
                        visited.add((nx, ny))
    return board


def majority_element(nums):
    # https://leetcode.com/problems/majority-element
    major_element, count = None, 0

    for num in nums:
        if count == 0:
            major_element = num
        if major_element == num:
            count += 1
        else:
            count -= 1

    return major_element


def word_ladder_ii(begin_word, end_word, word_list):
    # https://leetcode.com/problems/word-ladder-ii
    if begin_word not in word_list or end_word not in word_list:
        return []

    n = len(begin_word)
    word_list = set(word_list)
    mapping = defaultdict(list)     # {"h*t": ["hot", "hit"], "ho*": ["hot"]}

    for word in word_list:
        for i in range(n):
            intermediate_word = word[:i] + "*" + word[i + 1:]
            mapping[intermediate_word].append(word)

    queue = deque([(begin_word, 1)])
    parents = defaultdict(list)
    visited = {begin_word}
    found = False

    while queue and not found:
        level_visited = set()

        for _ in range(len(queue)):
            current = queue.popleft()

            for i in range(n):
                intermediate_word = current[:i] + "*" + current[i + 1:]

                for word in mapping[intermediate_word]:
                    if word not in visited:
                        if word == end_word:
                            found = True

                        if word not in visited:
                            level_visited.add(word)
                            queue.append(word)

                        parents[word].append(current)

        visited |= level_visited        # Mark after level finishes

    result = []

    def dfs(word, path):
        if word == begin_word:
            result.append(path[::-1])
            return
        for p in parents[word]:
            dfs(p, path + [p])

    if found:
        dfs(end_word, [end_word])

    return result
