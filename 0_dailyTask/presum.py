# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/9/10 22:59
# @FileName : presum.py
# @SoftWare : PyCharm

from itertools import accumulate
import itertools
import bisect

class Solution:
    def chalkReplacer(self, chalk, k):
        presum_chalk = list(itertools.accumulate(chalk))
        print(presum_chalk)
        return bisect.bisect_right(presum_chalk, k % sum(chalk))


print(Solution().chalkReplacer([5, 2, 4, 7, 11], 35))

import heapq
