# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/10/12 21:48
# @FileName : 29_两数相除.py
# @SoftWare : PyCharm


class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        is_neg = (dividend > 0) ^ (divisor > 0)
        dividend, divisor = abs(dividend), abs(divisor)
        cur, mul = divisor, 1
        ans = 0
        while dividend >= cur:
            if cur + cur <= dividend:
                cur += cur
                mul += mul
            else:
                ans += mul
                dividend -= cur
                cur, mul = divisor, 1

        return max(-ans, -2147483648) if is_neg else min(ans, 2147483647)


if __name__ == '__main__':
    dividend = 20
    divisor = 3
    print(Solution().divide(dividend, divisor))