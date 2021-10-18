# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/10/12 22:09
# @FileName : 84_柱状图最大矩形.py
# @SoftWare : PyCharm



class Solution:
    def largestRectangleArea(self, heights):
        max_area = 0
        n = len(heights)
        for i in range(n):
            left = i - 1
            right = i + 1
            if n * heights[i] > max_area:
                while left >= 0 and heights[left] >= heights[i]:
                    left -= 1
                while right <= n - 1 and heights[right] >= heights[i]:
                    right += 1
                max_area = max(max_area, (right - left - 1) * heights[i])
        return max_area



class Solution1:
    def largestRectangleArea(self, heights):
        max_area = 0
        n = len(heights)
        if n == 1:
            return heights[0]
        for i in range(n):
            # 选定起点，然后从起点开始往后遍历更新max_area
            min_val = heights[i]
            max_area = max(heights[i], max_area)
            if min_val == 0:
                continue
            if i == n - 1:
                max_area = max(max_area, heights[-1])
            for j in range(i + 1, n):
                min_val = min(min_val, heights[j])
                broad = j - i + 1
                height = min_val
                max_area = max(max_area, broad * height)
        return max_area


if __name__ == '__main__':
    heights = [2, 1, 5, 6, 4, 2, 3]
    # heights = [0, 9]
    print(Solution().largestRectangleArea(heights))