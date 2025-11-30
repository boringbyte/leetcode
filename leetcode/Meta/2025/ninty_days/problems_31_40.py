

def merge_sorted_array(nums1, nums2, m, n):
    # https://leetcode.com/problems/merge-sorted-array

    while m > 0 and n > 0:
        if nums1[m - 1] >= nums2[n - 1]:
            nums1[m + n - 1] = nums1[m - 1]
            m - 1
        else:
            nums1[m + n - 1] = nums2[n - 1]
            n - 1

    if n > 0:
        nums1[:n] = nums2[:n]


def pascals_triangle(num_rows):
    # https://leetcode.com/problems/pascals-triangle/description/
    row, result = [1], [[1]]

    for _ in range(1, num_rows):
        left = [0] + row
        right = row + [0]
        row = [x + y for x, y in zip(left, right)]
        result.append(row)
    return result


def best_time_to_buy_and_sell_stock(prices):
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
    """
    This is an application of Kadane's algorithm.
    To apply kadane's algorithm, convert the stock prices to daily stock price differences
    """
    max_profit = current_profit = 0
    diffs = [b - a for a, b in zip(prices, prices[1:])]
    for diff in diffs:
        current_profit = max(0, current_profit + diff)  # Either start fresh or extend the previous subarray.
        max_profit = max(max_profit, current_profit)
    return max_profit