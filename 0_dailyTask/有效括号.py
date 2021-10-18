# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/9/12 17:46
# @FileName : 有效括号.py
# @SoftWare : PyCharm


class Solution:
    def checkValidString(self, s: str) -> bool:
        stack = []
        asst_stack = []
        for i, sign in enumerate(s):
            if sign in ("(", "*"):
                stack.append(sign)
                continue
            if len(stack) == 0:
                return False
            # 弹出栈顶元素，如果是 ( 则继续循环，如果是*则将*暂存至另stack中并继续弹出栈顶元素
            # 直至弹出了( 或 原栈为空后，将辅助栈的*再依次压入原stack中
            while stack:
                if '(' in stack:
                    top_sign = stack.pop()
                    if top_sign == "*":
                        asst_stack.append(top_sign)
                        continue
                    if top_sign == '(':
                        stack.extend(asst_stack)
                        asst_stack = []
                        break
                stack.pop()
                break

        if '(' not in stack:
            return True
        cnt = 0
        stack_str = ''.join(stack).lstrip('*')
        for i in stack_str:
            cnt = cnt + 1 if i == '(' else cnt - 1
        if cnt <= 0:
            return True
        return False



if __name__ == '__main__':
    nums = "(((((()*)(*)*))())())(()())())))((**)))))(()())()"
    nums = "(((((()*)(*)*))())())(()())())))((**)))))(()())()"
    inst = Solution()
    print(inst.checkValidString(nums))