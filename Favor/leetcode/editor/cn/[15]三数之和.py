# 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重
# 复的三元组。 
# 
#  注意：答案中不可以包含重复的三元组。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [-1,0,1,2,-1,-4]
# 输出：[[-1,-1,2],[-1,0,1]]
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = []
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：nums = [0]
# 输出：[]
#  
# 
#  
# 
#  提示： 
# 
#  
#  0 <= nums.length <= 3000 
#  -105 <= nums[i] <= 105 
#  
#  Related Topics 数组 双指针 排序 
#  👍 4136 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        for i, num in enumerate(nums):

            # 剪枝
            if num > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            l = i + 1
            r = len(nums) - 1
            while l < r:
                vsum = num + nums[l] + nums[r]
                if vsum == 0:
                    res.append([num, nums[l], nums[r]])

                    # 剪枝
                    while r > l and nums[r] == nums[r - 1]:
                        r -= 1
                    while r > l and nums[l + 1] == nums[l]:
                        l += 1

                    l += 1
                    r -= 1
                elif vsum > 0:
                    r -= 1
                else:
                    l += 1

        return res

# leetcode submit region end(Prohibit modification and deletion)
