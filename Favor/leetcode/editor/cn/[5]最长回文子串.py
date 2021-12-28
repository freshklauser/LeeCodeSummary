# 给你一个字符串 s，找到 s 中最长的回文子串。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：s = "babad"
# 输出："bab"
# 解释："aba" 同样是符合题意的答案。
#  
# 
#  示例 2： 
# 
#  
# 输入：s = "cbbd"
# 输出："bb"
#  
# 
#  示例 3： 
# 
#  
# 输入：s = "a"
# 输出："a"
#  
# 
#  示例 4： 
# 
#  
# 输入：s = "ac"
# 输出："a"
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= s.length <= 1000 
#  s 仅由数字和英文字母（大写和/或小写）组成 
#  
#  Related Topics 字符串 动态规划 
#  👍 4507 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = ''
        maxLen = 0
        for i in range(len(s)):
            scenter_sub = self.findPalindrome(i, i, s)
            dcenter_sub = self.findPalindrome(i, i + 1, s)
            if max(len(scenter_sub), len(dcenter_sub)) > maxLen:
                res = scenter_sub if len(scenter_sub) > len(dcenter_sub) else dcenter_sub
                maxLen = len(res)
        return res

    def findPalindrome(self, i, j, s):
        while i >= 0 and j < len(s) and s[i] == s[j]:
            i -= 1
            j += 1
        return s[i + 1:j]

# leetcode submit region end(Prohibit modification and deletion)
