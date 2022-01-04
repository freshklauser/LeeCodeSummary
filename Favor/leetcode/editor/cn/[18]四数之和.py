# 给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b
# ], nums[c], nums[d]] （若两个四元组元素一一对应，则认为两个四元组重复）： 
# 
#  
#  0 <= a, b, c, d < n 
#  a、b、c 和 d 互不相同 
#  nums[a] + nums[b] + nums[c] + nums[d] == target 
#  
# 
#  你可以按 任意顺序 返回答案 。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [1,0,-1,0,-2,2], target = 0
# 输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = [2,2,2,2,2], target = 8
# 输出：[[2,2,2,2]]
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= nums.length <= 200 
#  -109 <= nums[i] <= 109 
#  -109 <= target <= 109 
#  
#  Related Topics 数组 双指针 排序 
#  👍 1056 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result = []
        for i, fnum in enumerate(nums):

            # 剪枝去重, 每一层遍历的时候
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            for j in range(i + 1, len(nums), 1):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue

                left = j + 1
                right = len(nums) - 1
                while left < right:
                    vsum = fnum + nums[j] + nums[left] +nums[right]
                    if vsum == target:
                        result.append([fnum, nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        left += 1
                        right -= 1
                    elif vsum > target:
                        right -= 1
                    else:
                        left += 1
        return result

# leetcode submit region end(Prohibit modification and deletion)
