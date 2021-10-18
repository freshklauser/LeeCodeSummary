# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/10/15 22:41
# @FileName : 38_外观数列.py
# @SoftWare : PyCharm



class Solution:
    def countAndSay(self, n: int) -> str:
        disp = str(n)
        while n:
            disp = self.transfer(disp)
            n -= 1
        return disp

    def transfer(self, str_num):
        cnt_stack = []
        str_stack = []
        cnt = 1
        str_num = str_num + '#'
        for i, s in enumerate(str_num):
            if len(str_stack) == 0:
                str_stack.append(s)
                continue
            top_s = str_stack[-1]
            if s == top_s:
                cnt += 1
            else:
                cnt_stack.append(cnt)
                cnt = 1
                str_stack.append(s)
        cnt_str = [str(si) + str(sj) for si, sj in zip(cnt_stack, str_stack)]
        return ''.join(cnt_str)


if __name__ == '__main__':
    ws = 5
    inst = Solution()
    res = inst.countAndSay(ws)
    print(res)
