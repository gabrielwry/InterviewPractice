"""
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.
"""
"""
solution: find local min, subtract from price, update if bigger than max
"""
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        local_min = float('inf')
        max_profit = 0
        for each in prices:
            if each-local_min > max_profit:
                max_profit = each-local_min
            if each < local_min:
                local_min = each
        return max_profit