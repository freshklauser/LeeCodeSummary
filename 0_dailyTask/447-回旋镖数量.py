# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/9/13 22:12
# @FileName : 447-回旋镖数量.py
# @SoftWare : PyCharm

from collections import Counter
import math



class Solution:
    def numberOfBoomerangs(self, points):
        p_nums = dict()
        num = 0
        for i, p in enumerate(points):
            cur_nums = [
                self.distCalc(p, p_ext) for p_ext in points if p_ext != p
            ]
            cur_p_cnt = Counter(cur_nums)
            # print(cur_p_cnt)
            valid_ps = [v for k, v in cur_p_cnt.items() if v > 1]
            if not valid_ps:
                continue
            # 计算当前i为中心的所有距离中的回旋镖数量，排列，cnt 中选2个
            cnts = [self.arrange_from_num(m, 2) for m in valid_ps]
            num += sum(cnts)
        return num


    @staticmethod
    def distCalc(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def arrange_from_num(m, n=2):
        # A(m, n) = m! / (m-n)!
        if m == 2:
            return 2
        pre_m = 1
        pre_m_n = 1
        for i in range(1, m + 1):
            pre_m *= i
        for i in range(1, m - n + 1):
            pre_m_n *= i
        return pre_m / pre_m_n


if __name__ == '__main__':
    points = [[1, 1], [2, 2], [3, 3]]
    inst = Solution()
    print(inst.numberOfBoomerangs(points))
    # print(inst.arrange_from_num(4, 2))