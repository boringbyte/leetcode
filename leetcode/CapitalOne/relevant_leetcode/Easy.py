

def best_time_to_buy_and_sell_stock(prices):
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock/solutions/5501275/video-keep-minimum-price-solution/
    buy_price = prices[0]
    profit = 0

    for price in prices[1:]:
        buy_price = min(buy_price, price)
        profit = max(profit, price - buy_price)
    return profit

