# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/10/16 20:39
# @FileName : 282_给表达式添加运算符.py
# @SoftWare : PyCharm


class Solution:
    def addOperators(self, num, target):

        def dfs(expr, cur_pos, pre_expr_res, cur_expr_res, results):
            """
            示例如：num='1236', dfs( 1+2*3-6, 4, 3, 7 )
            :param expr: 当前表达式  1+2*3-6
            :param cur_pos: 当前位置 第4个数字
            :param cur_expr_res: 当前表达式结果：1+2*3-6=1
            :param pre_expr_res: 前一个表达式结果: 1+2*3=7
            :return:
            """

            # 如果当前位置已经到达num的最后一个数字，判断当前表达式结果是不是target
            if cur_pos == len(num):
                if cur_expr_res == target:
                    results.append(expr)
                return

            # 从当前位置遍历到num最后一个数字
            for i in range(1, len(num) - cur_pos + 1):
                sub_num = num[cur_pos:cur_pos + i]
                sub_val = int(sub_num)
                if sub_num[0] == '0' and len(sub_num) > 1:
                    # 前导0的直接跳过
                    break
                if cur_pos == 0:
                    # 表达式开头不能添加符号，dfs
                    dfs(sub_num, cur_pos + i, sub_val, sub_val, results)
                    continue
                dfs(expr + '+' + sub_num, cur_pos + i, +sub_val, cur_expr_res + sub_val, results)
                dfs(expr + '-' + sub_num, cur_pos + i, -sub_val, cur_expr_res - sub_val, results)
                dfs(expr + '*' + sub_num, cur_pos + i, pre_expr_res * sub_val,
                    cur_expr_res - pre_expr_res + pre_expr_res * sub_val, results)

        results = []
        dfs("", 0, 0, 0, results)
        return results






