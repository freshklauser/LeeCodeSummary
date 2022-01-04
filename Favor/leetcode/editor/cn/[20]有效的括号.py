# 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。 
# 
#  有效字符串需满足： 
# 
#  
#  左括号必须用相同类型的右括号闭合。 
#  左括号必须以正确的顺序闭合。 
#  
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：s = "()"
# 输出：true
#  
# 
#  示例 2： 
# 
#  
# 输入：s = "()[]{}"
# 输出：true
#  
# 
#  示例 3： 
# 
#  
# 输入：s = "(]"
# 输出：false
#  
# 
#  示例 4： 
# 
#  
# 输入：s = "([)]"
# 输出：false
#  
# 
#  示例 5： 
# 
#  
# 输入：s = "{[]}"
# 输出：true 
# 
#  
# 
#  提示： 
# 
#  
#  1 <= s.length <= 104 
#  s 仅由括号 '()[]{}' 组成 
#  
#  Related Topics 栈 字符串 
#  👍 2868 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        ks = {')': '(', ']': '[', '}': '{'}
        for char in s:
            # char 是左的情况
            if char in ks.values():
                stack.append(char)
                continue

            # char是右，但之前字符没有左的情况,返回False
            if len(stack) == 0:
                return False
            # char是右，但stack栈顶与char不成对的情况，返回False
            top = stack.pop()
            if ks[char] != top:
                return False
        # 最终 stack 不为空返回False, 为空返回 True
        return True if len(stack) == 0 else False

# leetcode submit region end(Prohibit modification and deletion)
