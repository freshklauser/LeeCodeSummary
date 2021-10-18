# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/9/6 21:49
# @FileName : 704-二分查找.py
# @SoftWare : PyCharm

from pprint import pprint

# 68 文本左右对齐


def fullJustify(words, maxWidth):
    measuers = [len(word) for word in words]
    start = 0
    end = 0
    lines = []
    cur_len = measuers[0]
    word_cnt = 0
    for i in range(1, len(measuers)):
        if cur_len + measuers[i] + 1 <= maxWidth:
            end += 1
            cur_len += measuers[i] + 1
        else:
            lines.append((words[start: end + 1], maxWidth - cur_len))
            word_cnt += (end - start + 1)
            start = i
            end = i
            cur_len = measuers[i]
    if word_cnt < len(words):
        end_line = ' '.join(words[word_cnt:])
        lines.append((words[word_cnt:], maxWidth - len(end_line)))
    pprint(lines)
    res = []
    for i in range(len(lines)):
        word_num = len(lines[i][0])
        space_num = lines[i][1]
        if i == len(lines) - 1:
            res.append(' '.join(lines[i][0]) + ' ' * space_num)
            break
        if word_num == 1:
            lines[i][0][0] = lines[i][0][0] + ' ' * space_num
        else:
            unit_space = space_num // (word_num - 1)
            unit_space_remain = space_num % (word_num - 1)
            if unit_space >= 1:
                for j in range(0, word_num - 1):
                    lines[i][0][j] = lines[i][0][j] + ' ' * unit_space
            for j in range(0, word_num - 1):
                if unit_space_remain > 0:
                    lines[i][0][j] = lines[i][0][j] + ' '
                    unit_space_remain -= 1
        res.append(' '.join(lines[i][0]))
    return res


if __name__ == '__main__':
    words = ["This", "is", "an", "example", "of", "text", "justification."]
    wordss = ["Science","is","what","we","understand","well","enough","to","explain",
             "to","a","computer.","Art","is","everything","else","we","do"]
    max_width = 20
    res = fullJustify(wordss, max_width)
    pprint(res)
    for i in res:
        print(len(i))

    from collections import deque
