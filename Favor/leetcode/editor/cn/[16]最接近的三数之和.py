# 给你一个长度为 n 的整数数组 nums 和 一个目标值 target。请你从 nums 中选出三个整数，使它们的和与 target 最接近。 
# 
#  返回这三个数的和。 
# 
#  假定每组输入只存在恰好一个解。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [-1,2,1,-4], target = 1
# 输出：2
# 解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = [0,0,0], target = 1
# 输出：0
#  
# 
#  
# 
#  提示： 
# 
#  
#  3 <= nums.length <= 1000 
#  -1000 <= nums[i] <= 1000 
#  -104 <= target <= 104 
#  
#  Related Topics 数组 双指针 排序 
#  👍 988 👎 0


# leetcode submit region begin(Prohibit modification and deletion)

"""
双指针法需要排序
"""
from typing import List


class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        res = sum(nums[:3])
        for i, num in enumerate(nums):
            left = i + 1
            right = len(nums) - 1
            while left < right:
                vsum = num + nums[left] + nums[right]
                if vsum == target:
                    return target
                elif vsum > target:
                    right -= 1
                    # todo: 剪枝，如果下一个元素相同，直接继续移动指针
                    while nums[right] == nums[right + 1] and right > left:
                        right -= 1
                else:
                    left += 1
                    # todo: 剪枝，如果下一个元素相同，直接继续移动指针
                    while nums[left] == nums[left - 1] and left < right:
                        left += 1
                if abs(res - target) > abs(vsum - target):
                    res = vsum
        return res



# leetcode submit region end(Prohibit modification and deletion)
