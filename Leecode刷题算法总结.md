​	

[TOC]

# 二分查找

## 模板汇总

| 模板             | 1                                                  | 2                                                            | 3                                                            |
| ---------------- | -------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 初始条件         | left = 0<br>right = length -1                      | left = 0<br>right = length                                   | left = 0<br>right = length - 1                               |
| 终止             | left > right<br>(while left <= right)              | left == right<br>(while left < right)                        | left + 1 == right<br>(while left + 1 < right)                |
| 向左查找         | right = mid - 1                                    | right = mid                                                  | right = mid                                                  |
| 向右查找         | left = mid + 1                                     | left = mid + 1                                               | left = mid                                                   |
| 适用范围         | 查找可以通过访问数组中的单个索引来确定的元素或条件 | 访问数组中当前索引及其直接右邻居索引的元素或条件             | 搜索需要*访问当前索引及其在数组中的直接左右邻居索引*的元素或条件 |
| 后处理           | 无                                                 | 剩下 1 个元素时，结束 需**评估剩余的1个元素是否符合条件**    | 剩下2 个元素时结束需**评估剩余的1个元素是否符合条件**        |
| 边界比较         | 不需要与元素的两侧比较                             | 使用元素的右邻居来确定<br>是否满足条件，并决定是向左还是向右 | 使用直接左右邻居来确定它是向右还是向左                       |
| 查找空间元素个数 |                                                    | 保证查找空间在每一步中至少有 2 个元素                        | 保证查找空间在每一步中至少有 3个元素                         |

**模板分析：**

举例而言

1. 在严格递增有序数组中寻找某个数

2. **在有序数组中寻找某个数第一次出现的位置（或者在有序数组中寻找第一个大于等于某个数的位置）**

3. 已知有一个先严格递增后严格递减的数组，找数组最大值的位置

​    实际上在解题时，一般前两种模板就足够了（大部分应用第三种模板的情况都可以转化为前两种），第二种的模板和 Python 中 bisect.bisect() 类似。

### 模板1

**适用性**

- 用于查找可以通过*访问数组中的单个索引*来确定的元素或条件

**关键属性**

- 二分查找的最基础和最基本的形式。
- **查找条件可以在不与元素的两侧进行比较的情况下确定**（或使用它周围的特定元素） ---  <font color=red>区间左开右开</font>。
- 不需要后处理，因为每一步中，你都在检查是否找到了元素。如果到达末尾，则知道未找到该元素。

**区分语法**

- 初始条件：left = 0, right = length-1
- 终止：left > right
- 向左查找：right = mid-1
- 向右查找：left = mid+1

**标准示例**

```python
def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # End Condition: left > right
    return -1
```



### 模板2

**适用性**

- 用于查找需要*访问数组中当前索引及其直接右邻居索引*的元素或条件（<font color=red>右边界查找</font>）

**关键属性**

- 一种实现二分查找的高级方法。
- **查找条件需要访问元素的直接右邻居** （<font color=red>区间左开右闭</font>）。
- 使用元素的右邻居来确定是否满足条件，并决定是向左还是向右。
- **保证查找空间在每一步中至少有 2 个元素**。
- 需要进行**后处理**。 当你剩下 1 个元素时，循环 / 递归结束。 需要**评估剩余元素是否符合条件**。

**区分语法**

- 初始条件：left = 0, right = length
- 终止：left == right
- 向左查找：right = mid  
- 向右查找：left = mid+1

**标准示例**

```python
def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums)		# right = len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    # Post-processing:
    # End Condition: left == right
    if left != len(nums) and nums[left] == target:  # 由于right初始为len(nums)超出索引右边界，当right一直没改变的情况下，mid = len(nums) - 1时，left=right 也超出右边界，因此需要判断left是否超出右边界
        return left
    return -1

```



### 模板3

**适用性**

- 用于搜索需要*访问当前索引及其在数组中的直接左右邻居索引*的元素或条件 （<font color=red>左右边界</font>）

**关键属性**

- 实现二分查找的另一种方法。
- 搜索条件需要访问元素的直接左右邻居。
- 使用元素的邻居来确定它是向右还是向左。
- 保证查找空间在每个步骤中至少有 **3 个元素**。
- 需要进行后处理。 当剩下 2 个元素时，循环 / 递归结束。 需要评估其余元素是否符合条件。

**区分语法**

- 初始条件：left = 0, right = length-1
- 终止：left + 1 == right
- 向左查找：right = mid
- 向右查找：left = mid

### 参考链接

- [二分查找分析](https://leetcode-cn.com/problems/search-insert-position/solution/te-bie-hao-yong-de-er-fen-cha-fa-fa-mo-ban-python-/)

## 实际应用

### [搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)

#### 题目

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

-  每行中的整数从左到右按升序排列。

- 每行的第一个整数大于前一行的最后一个整数。

示例：

![img](https://assets.leetcode.com/uploads/2020/10/05/mat.jpg)

```reStructuredText
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
输出：true
```

#### 解题思路 (二分)

二分（竖向二分找边界，横向二分找target）

**1）代码**

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        竖向二分找边界，横向二分找target
        i = bisect.right(nums, target)
        all e in a[:i] have e <= x (不包含索引i), and all e in a[i:] have e > x
        nums[:idx] 即0 ~~ idx-1 索引的元素 e <= x, nums[idx - 1] == target 则True
        """
        m = len(matrix)
        # 竖向二分找边界
        frow_nums = [matrix[i][0] for i in range(m)]
        row_idx = bisect.bisect_right(frow_nums, target)
        if frow_nums[row_idx - 1] == target:
            return True
        # row_idx - 1 索引行内横向二分查找目标值
        nums = matrix[row_idx - 1]
        idx = bisect.bisect_right(nums, target)
        if nums[idx - 1] == target:
            return True
        return False
```



### 搜索二维矩阵 II

#### 题目

编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。

示例：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/25/searchgrid2.jpg)

```reStructuredText
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true
```

#### 解题思路（二分）

竖向二分确定下边界行，横向二分确定右边界列，根据下边界所在行二分确定左边界，右边界所在列确定上边界;

根据四个边界缩小范围, 递归调用; 递归后缩小的矩形范围内逐行或逐列二分查找target

**1）技巧**

bisect模块直接调用二分

```reStructuredText
bisect_right: a[:i] 中的 e <= x; a[i:] 中的e>x, a[i] > x;
bisect_left : a[:i] 中的 e < x ; a[i:] 中的e>=x, a[i] >= x
```

**2）代码**

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        竖向二分确定下边界行，横向二分确定右边界列，
        根据下边界所在行二分确定左边界，根据右边界所在列确定上边界
        根据四个边界缩小范围, 递归调用
        矩形搜索时从下往上，从右往左
        bisect_right: a[:i] 中的 e <= x; a[i:] 中的e>x, a[i] > x
        """
        def iteration_matrix(nums, target):

            rows = len(nums)
            cols = len(nums[0])
            frow_nums = [nums[i][0] for i in range(rows)]
            fcol_nums = [nums[0][i] for i in range(cols)]

            b_idx = bisect.bisect_right(frow_nums, target)
            r_idx = bisect.bisect_right(fcol_nums, target)
            if b_idx <= 0 or r_idx <= 0:
                return nums

            brow_nums = nums[b_idx - 1]
            l_idx = bisect.bisect_right(brow_nums, target)
            rcol_nums = [nums[i][r_idx - 1] for i in range(len(nums))]
            u_idx = bisect.bisect_right(rcol_nums, target)
            if l_idx <= 0 or u_idx <= 0:
                return nums

            # 矩形子区间
            tmp = nums[u_idx - 1: b_idx]
            sub_matrix = [tmp[i][l_idx - 1:r_idx] for i in range(len(tmp))]
            if sub_matrix == nums:
                return sub_matrix
            return iteration_matrix(sub_matrix, target)
        
        nums = iteration_matrix(matrix, target)
        m = len(nums)
        n = len(nums[0])
        if m < n:
            for i in range(m):
                unit = nums[i]
                idx = bisect.bisect_right(unit, target)
                if unit[idx - 1] == target:
                    return True
        else:
            for j in range(n):
                unit = [nums[i][j] for i in range(m)]
                idx = bisect.bisect_right(unit, target)
                if unit[idx - 1] == target:
                    return True
        return False
```



#### 解题思路（单调性）

从左下角（或右上角）开始走，遇到比目标值大的，就往上走（因为往上是递减的），遇到比目标值小的，就往右走（因为往右是递增的）

**1）代码**

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        从左下角（或右上角）开始走，遇到比目标值大的，就往上走（因为往上是递减的），
        遇到比目标值小的，就往右走（因为往右是递增的
        """
        row = len(matrix) - 1
        col = len(matrix[0]) - 1
        i = row
        j = 0
        while i >= 0 and j <= col:
            print(matrix[i][j])
            if matrix[i][j] == target:
                return True
            if matrix[i][j] > target:
                i -= 1
            else:
                j += 1
        return False
```

# DFS/BFS 回溯

## 回溯模板

![image.png](https://pic.leetcode-cn.com/1600377894-BuZLhZ-image.png)



## 使用场景

- 返回所有可能结果



## 实际应用

### 全排列

#### 题目

给定一个不含重复数字的数组 `nums` ，返回其 **所有可能的全排列** 。你可以 **按任意顺序** 返回答案。

示例：

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

#### 解题思路（DFS+回溯 标准模板）

1）代码

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        # return list(itertools.permutations(nums))
        res = []
        visited = set()
        self.dfs(nums, [], visited, res)
        return res

    def dfs(self, nums, path, visited, res):
        if len(path) == len(nums):
            res.append(path)
            return
        
        for i, num in enumerate(nums):
            if num in visited:
                continue
            path.append(num)
            visited.add(num)
            self.dfs(nums, path[:], visited, res)
            # 回溯
            path.pop()
            visited.remove(num)
```

### [重新排序得到 2 的幂](https://leetcode-cn.com/problems/reordered-power-of-2/)

#### 题目

给定正整数 N ，我们按任何顺序（包括原始顺序）将数字重新排序，注意其前导数字不能为零。

如果我们可以通过上述方式得到 2 的幂，返回 true；否则，返回 false。

示例：

```
输入：24
输出：false
```

#### 解题思路（全排列/DFS/回溯/剪枝）

1）剪枝

以下三种情况的任一种情况出现都可以剪枝

- len(path) == 0 and num == '0' ： 第一个数字为 0
- visited[i] == 1：已经出现过的

- i < len(nums) - 1 and num == nums[i + 1] and visited[i + 1] == 0：连续相同数的排列重复情况剪枝

2）代码

```python
class Solution:
    def reorderedPowerOf2(self, n: int) -> bool:

        def dfs(nums, path, visited):
            if len(path) == len(nums):
                return self.check_power(int(path))
            for i, num in enumerate(nums):
                if len(path) == 0 and num == '0' or visited[i] == 1 or i < len(nums) - 1 and num == nums[i + 1] and visited[i + 1] == 0:
                    continue
                visited[i] = 1
                status = dfs(nums, path + num, visited)
                if status:
                    return True
                visited[i] = 0
            return False

        nums = sorted(list(str(n)))
        visited = [0] * len(nums)
        status = dfs(nums, '', visited)
        return status

    @staticmethod
    def check_power(num):
        return num & (num - 1) == 0
```

#### 解题思路（itertools库全排列）

1）代码

```python
class Solution:
    def reorderedPowerOf2(self, n: int) -> bool:
        """
        找出所有组合，利用itertools库实现全排列
        """
        permus = itertools.permutations(str(n))
        queue = set([int(''.join(s)) for s in permus if s[0] != '0'])
        result = list(filter(self.check_power, queue))
        if result:
            return True
        return False

    @staticmethod
    def check_power(num):
        return num & (num - 1) == 0
```



### [ 删除无效的括号](https://leetcode-cn.com/problems/remove-invalid-parentheses/)

#### 题目

给你一个由若干括号和字母组成的字符串 `s` ，删除最小数量的无效括号，使得输入的字符串有效。

返回所有可能的结果。答案可以按 **任意顺序** 返回。

#### 解题思路（BFS）

分层遍历，每一层 while loop 只减少一个半括号 并判断该层的所有子串中是否存在合理的子字符串，有则返回结果，无则在该轮的不合理子字符串基础上，再减少一个半括号来轮循下一层校验

**1）技巧**

**filter函数**过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。

该函数接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。

```python
# 语法：filter(function, iterable)
# 示例：
import math
def is_sqr(x):
    return math.sqrt(x) % 1 == 0
 
newlist = list(filter(is_sqr, range(1, 101)))
print(newlist)	# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

**2）代码**

```class Solution:

    def removeInvalidParentheses(self, s: str) -> List[str]:

        """

        bfs

        每一轮while loop 只减少一个半括号 并判断该轮是否右合理的字符串，有则返回结果，

        无则在该轮的不合理字符串基础上，再减少一个半括号来轮循校验

        """

        # 每一层减少一个半括号有可能存在重复的子字符串，直接用set代替列表来去重

        queue = set()

        queue.add(s)


        while queue:

            # 判断当前队列集合中是否存在合理的字符串，有则直接返回结果，无则继续

            result = list(filter(self.is_valid, queue))

            if result:

                return result

            

            cur_layer_queue = set()

            for chars in queue:

                for i, char in enumerate(chars):

                    if char == '(' or char == ')':

                        cur_layer_queue.add(chars[:i] + chars[i + 1:])

            # 当前层遍历完后，将所有的去掉一个字符的子字符串赋值给 queue

            queue = cur_layer_queue

    

    def is_valid(self, s):

        cnt = 0

        for char in s:

            if char == '(':

                cnt += 1

            if char == ')':

                cnt -= 1

                # 右括号多则直接无效

                if cnt < 0:

                    return False

        # 合理的字符串，左右括号相等，cnt==0

        return cnt == 0

#### 解题思路（回溯+剪枝）

利用括号匹配的规则求出该字符串 s 中最少需要去掉的左括号的数目 lremove 和右括号的数目 rremove，然后我们尝试在原字符串 s 中去掉 lremove 个左括号和 rremove 个右括号，然后检测剩余的字符串是否合法匹配，如果合法匹配则我们则认为该字符串为可能的结果，我们利用回溯算法来尝试搜索所有可能的去除括号的方案。

在进行回溯时可以利用以下的剪枝技巧来增加搜索的效率：

- 充分利用括号左右匹配的特点（性质），因此我们设置变量 lcount 和 rcount，分别表示在遍历的过程中已经用到的左括号的个数和右括号的个数，当出现 lcount < rcount 时，则我们认为当前的字符串已经非法，停止本次搜索。
- 我们从字符串中每去掉一个括号，则更新 lremove 或者 rremove，当我们发现剩余未尝试的字符串的长度小于 lremove + rremove 时，则停止本次搜索。
- 当 lremove 和}rremove 同时为 0 时，则我们检测当前的字符串是否合法匹配，如果合法匹配则我们将其记录下来。

由于记录的字符串可能存在重复，因此需要对重复的结果进行去重，去重的办法有如下两种：

- 利用哈希表对最终生成的字符串去重。

- 我们在每次进行搜索时，如果遇到连续相同的括号我们只需要搜索一次即可，比如当前遇到的字符串为 "(((())"，去掉前四个左括号中的任意一个，生成的字符串是一样的，均为 "((())"，因此我们在尝试搜索时，只需去掉一个左括号进行下一轮搜索，不需要将前四个左括号都尝试一遍。

**1）代码**  (未自己实现)

​```python
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        res = []
        lremove, rremove = 0, 0
        for c in s:
            if c == '(':
                lremove += 1
            elif c == ')':
                if lremove == 0:
                    rremove += 1
                else:
                    lremove -= 1

        def isValid(str):
            cnt = 0
            for c in str:
                if c == '(':
                    cnt += 1
                elif c == ')':
                    cnt -= 1
                    if cnt < 0:
                        return False
            return cnt == 0

        def helper(s, start, lcount, rcount, lremove, rremove):
            if lremove == 0 and rremove == 0:
                if isValid(s):
                    res.append(s)
                return

            for  i in range(start, len(s)):
                if i > start and s[i] == s[i - 1]:
                    continue
                # 如果剩余的字符无法满足去掉的数量要求，直接返回
                if lremove + rremove > len(s) - i:
                    break
                # 尝试去掉一个左括号
                if lremove > 0 and s[i] == '(':
                    helper(s[:i] + s[i + 1:], i, lcount, rcount, lremove - 1, rremove);
                # 尝试去掉一个右括号
                if rremove > 0 and s[i] == ')':
                    helper(s[:i] + s[i + 1:], i, lcount, rcount, lremove, rremove - 1);
                # 统计当前字符串中已有的括号数量
                if s[i] == ')':
                    lcount += 1
                elif s[i] == ')':
                    rcount += 1
                # 当前右括号的数量大于左括号的数量则为非法,直接返回.
                if rcount > lcount:
                    break

        helper(s, 0, 0, 0, lremove, rremove)
        return res
```

### 



# 二叉树

## 二叉树构建

### 基本构建

```python
class BinaryTree:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def get(self):
        return self.data

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

    def setLeft(self, node):
        self.left = node

    def setRight(self, node):
        self.right = node
```

```python
# 实例
def init_bst():
    binaryTree = BinaryTree(0)
    binaryTree.setLeft(BinaryTree(1))
    binaryTree.setRight(BinaryTree(2))
    binaryTree.getLeft().setLeft(BinaryTree(3))
    binaryTree.getLeft().setRight(BinaryTree(4))
    binaryTree.getRight().setLeft(BinaryTree(5))
    binaryTree.getRight().setRight(BinaryTree(6))
    return binaryTree
```



### [从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

根据一棵树的中序遍历与后序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

```reStructuredText
例如，给出
中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]
构建二叉树结果：[3,9,20,null,null,15,7]
```

1）思路：根据中序遍历和后续遍历特性划分出左右子树

![树的特性.png](https://pic.leetcode-cn.com/3293e7ccb41baaf52adca7e13cc0f258e1c83a4c588f9b6cb3e86410a540f298-%E6%A0%91%E7%9A%84%E7%89%B9%E6%80%A7.png)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if postorder == []:
            return None
        val = postorder.pop()

        # 1. 根节点的val属性
        root = TreeNode(val)
        rt_idx = inorder.index(val)

        # 中序遍历的左子树节点和右子树节点
        inorder_left = inorder[:rt_idx]
        inorder_right = inorder[rt_idx + 1:]

        # 后续遍历的左子树节点和右子树节点
        postorder_left = postorder[:rt_idx]
        postorder_right = postorder[rt_idx: rt_idx + len(inorder_right)]

        # 2. 根节点的 left 属性 和 right 属性
        # 利用左子树的中序遍历和后续遍历结果构建root的左子树，递归实现
        root.left = self.buildTree(inorder_left, postorder_left)
        # 利用右子树的中序遍历和后续遍历结果构建root的右子树，递归实现
        root.right = self.buildTree(inorder_right, postorder_right)

        return root
```



### 从前序和中序遍历序列构造二叉树

思路：类似3.1.2， 确定 root 索引后划分左子树和右子树，递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder:
            return None
        val = preorder[0]
        root = TreeNode(val)

        rt_idx = inorder.index(val)

        preorder_left = preorder[1: rt_idx + 1]
        preorder_right = preorder[rt_idx + 1:]

        inorder_left = inorder[:rt_idx]
        inorder_right = inorder[rt_idx + 1:]

        root.left = self.buildTree(preorder_left, inorder_left)
        root.right = self.buildTree(preorder_right, inorder_right)

        return root
```

### [ 填充每个节点的下一个右侧节点指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)

Ⅱ 与该题一样

示例：

![img](https://assets.leetcode.com/uploads/2019/02/14/116_sample.png)

```
输入：root = [1,2,3,4,5,6,7]
输出：[1,#,2,3,#,4,5,6,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化的输出按层序遍历排列，同一层节点由 next 指针连接，'#' 标志着每一层的结束。
```

思路：**BFS 层序遍历过程中添加 next 关系**

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        """ BFS 层序遍历 """
        queue = collections.deque()
        queue.append(root)
        while queue:
            cur_layer_length = len(queue)
            for i in range(cur_layer_length):
                cur_node = queue.popleft()
                # 注意 None 节点判断
                if not cur_node:
                    continue
                if i == cur_layer_length - 1:
                    cur_node.next = None
                else:
                    cur_node.next = queue[0]
                if cur_node.left:
                	queue.append(cur_node.left)
                if cur_node.right:
                	queue.append(cur_node.right)
        return root
```



## 遍历二x叉树

```python
# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/10/17 20:29
# @FileName : tranversal.py
# @SoftWare : PyCharm

from collections import deque

from binary_tree_init import init_bst


class Traversal:
    def __init__(self):
        self.bst_values = []

    @staticmethod
    def layer_order_traversal(root):
        """
        BFS层序遍历, 使用队列
        :param root:
        :return:
        """
        result = []
        dq = deque()
        dq.append(root)
        while dq:
            cur_length = len(dq)
            level = []
            for _ in range(cur_length):
                cur = dq.popleft()
                if not cur:
                    continue
                dq.append(cur.left)
                dq.append(cur.right)
                level.append(cur.data)
            if level:
                result.append(level)
        return result

    def pre_order_traversal(self, root, method=None):
        if method == 'recurse':
            return self.pre_order_traversal__recurse(root)
        if method == 'dfs':
            return self.pre_order_traversal__dfs(root)
        return self.pre_order_traversal__stack(root)

    def mid_order_traversal(self, root, method=None):
        if method == 'recurse':
            return self.mid_order_traversal__recurse(root)
        if method == 'dfs':
            return self.mid_order_traversal__dfs(root)
        return self.mid_order_traversal__stack(root)

    def post_order_traversal(self, root, method=None):
        if method == 'recurse':
            return self.post_order_traversal__recurse(root)
        if method == 'dfs':
            return self.post_order_traversal__dfs(root)
        return self.post_order_traversal__stack(root)

    def pre_order_traversal__recurse(self, root):
        """
        前序遍历， 递归实现, 二叉树遍历最易理解和实现的版本
        三部曲：
        1)  遍历根节点值
        2） 左子节点递归调用
        3） 右子节点递归调用
        :param root: 二叉搜索树
        :return:
        """
        if not root:
            return []
        return [root.data] \
               + self.pre_order_traversal__recurse(root.left) \
               + self.pre_order_traversal__recurse(root.right)

    @staticmethod
    def pre_order_traversal__dfs(root):
        """ 通用模板：dfs 递归实现前序遍历 """

        def dfs(cur_node):
            if not cur_node:
                return
            result.append(cur_node.data)
            dfs(cur_node.left)
            dfs(cur_node.right)

        result = []
        dfs(root)
        return result

    @staticmethod
    def pre_order_traversal__stack(root):
        """ 最常用模板: 栈实现，非递归，性能最优 """
        if not root:
            return []
        result = []
        stack = []
        cur_node = root
        while stack or cur_node:
            while cur_node:
                # 添加根节点到结果表的同时当前节点遍历入栈，并指针移动到当前节点的左节点
                result.append(cur_node.data)
                stack.append(cur_node)
                cur_node = cur_node.left
            cur_node = stack.pop()
            cur_node = cur_node.right
        return result

    @staticmethod
    def mid_order_traversal__stack(root):
        if not root:
            return []
        result = []
        stack = []
        cur_node = root
        while stack or cur_node:
            while cur_node:
                # 当前节点遍历入栈，同时指针移动到当前节点的左节点
                stack.append(cur_node)
                cur_node = cur_node.left
            cur_node = stack.pop()
            result.append(cur_node.data)
            cur_node = cur_node.right
        return result

    @staticmethod
    def post_order_traversal__stack(root):
        if not root:
            return []
        result = []
        stack = []
        cur_node = root
        while stack or cur_node:
            while cur_node:
                result.append(cur_node.data)
                stack.append(cur_node)
                cur_node = cur_node.right
            cur_node = stack.pop()
            cur_node = cur_node.left
        return result[::-1]

    def mid_order_traversal__recurse(self, root):
        if not root:
            return []
        return self.mid_order_traversal__recurse(root.left) \
               + [root.data] \
               + self.mid_order_traversal__recurse(root.right)

    def post_order_traversal__recurse(self, root):
        if not root:
            return []
        return self.post_order_traversal__recurse(root.left) \
               + self.post_order_traversal__recurse(root.right) + [root.data]

    @staticmethod
    def mid_order_traversal__dfs(root):

        def dfs(cur_node):
            if not cur_node:
                return
            dfs(cur_node.left)
            result.append(cur_node.data)
            dfs(cur_node.right)

        result = []
        dfs(root)
        return result

    @staticmethod
    def post_order_traversal__dfs(root):

        def dfs(cur_node):
            if not cur_node:
                return
            dfs(cur_node.left)
            dfs(cur_node.right)
            result.append(cur_node.data)

        result = []
        dfs(root)
        return result


if __name__ == '__main__':
    b_tree = init_bst()
    print('前序：', Traversal().pre_order_traversal__stack(b_tree))
    print('中序：', Traversal().mid_order_traversal__recurse(b_tree))
    print('中序stack：', Traversal().mid_order_traversal__stack(b_tree))
    print('后序：', Traversal().post_order_traversal__dfs(b_tree))
    print('后序rec：', Traversal().post_order_traversal__recurse(b_tree))
    print('后序stack：', Traversal().post_order_traversal__stack(b_tree))
    print('---'*50)
    print(Traversal().pre_order_traversal(b_tree))
    print(Traversal().layer_order_traversal(b_tree))
    # print(pre_order(b_tree))
```

### 前序遍历

#### 递归（通用模板）

```python
def pre_order_traversal__dfs(root):
    """ 通用模板：dfs 递归实现前序遍历 """

    def dfs(cur_node):
        if not cur_node:
            return
        result.append(cur_node.data)
        dfs(cur_node.left)
        dfs(cur_node.right)

    result = []
    dfs(root)
    return result
```

#### 性能模板 （栈 -- 非递归）

```python
def pre_order_traversal__stack(root):
    """ 最常用模板: 栈实现，非递归，性能最优 """
    if not root:
        return []
    result = []
    stack = []
    cur_node = root
    while stack or cur_node:
        while cur_node:
            # 添加根节点到结果表的同时当前节点遍历入栈，并指针移动到当前节点的左节点
            result.append(cur_node.data)
            stack.append(cur_node)
            cur_node = cur_node.left
        cur_node = stack.pop()
        cur_node = cur_node.right
    return result
```



#### 最简单模板

```python
def pre_order_traversal__recurse(root):
    """
    前序遍历， 递归实现, 二叉树遍历最易理解和实现的版本
    三部曲：
    1)  遍历根节点值
    2） 左子节点递归调用
    3） 右子节点递归调用
    :param root: 二叉搜索树
    :return:
    """
    if not root:
        return []
    return [root.data] \
           + pre_order_traversal__recurse(root.left) \
           + pre_order_traversal__recurse(root.right)
```



## 实际应用

### 对称二叉树（递归/层序遍历）

- 题目

  给定一个二叉树，检查它是否是镜像对称的。

- 代码

  ```python
  # Definition for a binary tree node.
  class TreeNode:
      def __init__(self, val=0, left=None, right=None):
      self.val = val
      self.left = left
      self.right = right
      
  class Solution:
      def isSymmetric(self, root: TreeNode) -> bool:
          
          def check(l, r):
              if (l == None and r != None) or (l != None and r == None):
                  return False
              elif l == None and r == None:
                  return True
              else:
                  return l.val == r.val and check(l.right, r.left) and check(l.left, r.right)
          
          if not root:
              return True
          return check(root.left, root.right)
  ```

### [路径总和](https://leetcode-cn.com/problems/path-sum/) （二叉树 BFS/DFS/递归/栈）

#### 题目

给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。叶子节点 是指没有子节点的节点

示例：

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)

```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true
```

#### 解题思路 （BFS/DFS/递归/栈）

1）代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        
        """ 递归 """
        # if not root:
        #     return False
        # if root.left is None and root.right is None:
        #     return root.val == targetSum
        # return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)

        """ bfs """
        if not root:
            return False
        queue = collections.deque()
        queue.append((root, root.val))
        while queue:
            node, presum = queue.popleft()
            if not node.left and not node.right and presum == targetSum:
                return True
            if node.left:
                queue.append((node.left, presum + node.left.val))
            if node.right:
                queue.append((node.right, presum + node.right.val))
        return False
            
        """ 栈 非递归 """
        # if not root:
        #     return False
        # stack = [(root, root.val)]
        # while stack:
        #     node, t_sum = stack.pop()
        #     if not node.left and not node.right and t_sum == targetSum:
        #         return True
        #     if node.left:
        #         stack.append((node.left, node.left.val + t_sum))
        #     if node.right:
        #         stack.append((node.right, node.right.val + t_sum))
        # return False

        
        """ dfs, 输出所有路径 """
        # if not root:
        #     return False
        # def dfs(node, presum, res,  path):
        #     # res: 存储所有符合条件的结点路径    <-----
        #     # path: 节点路径					<-----
        #     # presum: 前缀和
        #     if not node:
        #         return
        #     if not node.left and not node.right and presum == targetSum:
        #         res.append(path)
        #     if node.left:
        #         dfs(node.left, presum + node.left.val, res, path + [node.left.val])
        #     if node.right:
        #         dfs(node.right, presum + node.right.val, res, path + [node.right.val])
        # res = []
        # dfs(root, root.val, res, [root.val])
        # return len(res) > 0

        # def dfs(node, presum):
        #     if not node:
        #         return False
        #     if not node.left and not node.right and presum == targetSum:
        #         return True
        #     if node.left:
        #         status = dfs(node.left, presum + node.left.val)
        #         if status:
        #             return True
        #     if node.right:
        #         status = dfs(node.right, presum + node.right.val)
        #         if status:
        #             return True
        #     return False

        # return dfs(root, root.val)

```

### [路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

#### 题目

给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。

路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

示例：

![img](https://assets.leetcode.com/uploads/2021/04/09/pathsum3-1-tree.jpg)

```
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条，如图所示。
```

#### 解题思路

### 



# 优先队列/堆

## 使用场景

- 最大k / 最小k 问题，需要动态查找每一步的最小值或最大值
- 动态维护数据有序性

## 堆构建/属性/方法

- python库： heapq

## 实际应用

### 接雨水Ⅱ (优先队列/小顶堆)

#### 题目

给你一个 `m x n` 的矩阵，其中的值均为非负整数，代表二维高度图每个单元的高度，请计算图中形状最多能接多少体积的雨水。

 示例：

![img](https://assets.leetcode.com/uploads/2021/04/08/trap1-3d.jpg)

```
输入: heightMap = [[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]
输出: 4
解释: 下雨后，雨水将会被上图蓝色的方块中。总的接雨水量为1+2+1=4。
```

#### 解题思路（小顶堆）

1）思路

​	维护周围一个圈，用堆来维护周围这一圈中的最小元素。为什么是维护最小的元素不是最大的元素呢，因为木桶原理呀。这个最小的元素从堆里弹出来，和它四个方向的元素去比较大小，看能不能往里灌水，怎么灌水呢，如果用方向就比较复杂了，我们可以用visited数组来表示哪些遍历过，哪些没遍历过。如果当前弹出来的高度比它周围的大，他就能往矮的里面灌水了，灌水后要把下一个柱子放进去的时候，放的高度要取两者较大的，也就是灌水后的高度，不是它原来矮的时候的高度了，如果不能灌水，继续走。

​	示例如：先把第一圈都放进去，然后开始从堆中弹出，第一圈，最小值是1（遍历时候标记为访问过），1从堆里弹出来，比如弹出来1(坐标[0,3])，它下方的3没有被访问过，尝试灌水，发现不能灌水，3入堆，然后继续弹。比如说，我此时弹出来一个3（坐标[1,0]），它能向右边2(坐标[1,1])灌水，那这边就可以统计了，然后我们要插入2(坐标[1,1])这个位置，但是插入的时候，要记得你得是插入的高度得是灌水后的高度，而不是原来的高度了

2）代码

```python
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        row = len(heightMap)
        col = len(heightMap[0])
        
        # 特殊情况
        if row < 3 or col < 3:
            return 0

        visited = [[0] * col for _ in range(row)]
        # 最外层元素构建优先队列(小顶堆)--  队列元素结构为 (h, i, j), 排序顺序优先级按索引从小到大
        priority_queue = []
        for j in range(col):
            priority_queue.append((heightMap[0][j], 0, j))
            priority_queue.append((heightMap[row - 1][j], row - 1, j))
        for i in range(1, row - 1):
            priority_queue.append((heightMap[i][0], i, 0))
            priority_queue.append((heightMap[i][col - 1], i, col - 1))
        heapq.heapify(priority_queue)

        # 遍历优先队列, 每次pop最小高度元素，找找四周元素的内部节点，添加内部节点到队列中且该节点高度为 当前系欸但和内部节点高度的较大值
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        volume = 0
        while priority_queue:
            h, x, y = heapq.heappop(priority_queue)
            for dx, dy in directions:
                nx = x + dx
                ny = y + dy
                if 0 < nx < row - 1 and 0 < ny < col - 1 and not visited[nx][ny]:
                    # 内部节点注水且将内部节点替换当前节点作为外围节点，内部节点标记1
                    volume += max(0, h - heightMap[nx][ny])
                    visited[nx][ny] = 1
                    heapq.heappush(priority_queue, (max(h, heightMap[nx][ny]), nx, ny))
        return volume
```

#### [合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

#### 题目

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

#### 解题思路（优先队列）

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return
        
        newHead = ListNode(None)
        heap = []

        # 依次取出 所有 ListNode 中的元素，每个节点val添加到优先队列heap中
        for node in lists:
            while node:
                heapq.heappush(heap, node.val)
                node = node.next

        # 从优先队列heap中依次弹出堆顶元素重构 ListNode
        cur = newHead
        while heap:
            cur.next = ListNode(heapq.heappop(heap))
            cur = cur.next
        return newHead.next
```



#### 解题思路（优先队列 / 构建可比较的 ListNode 对象）

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class ListNodeCompare:
    def __init__(self, listNode):
        self.node = listNode
    
    # 定义富比较函数
    def __lt__(self, other):
        return self.node.val < other.node.val


class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:

        new_lists = [item for item in lists if item]
        if not new_lists:
            return 

        newHead = ListNode(None)

        heap = []
        for head in new_lists:
            new_node = ListNodeCompare(head)
            heap.append(ListNodeCompare(head))
        heapq.heapify(heap)
        
        cur = newHead
        while heap:
            org_node = heapq.heappop(heap).node
            cur.next = org_node
            cur = cur.next
            # 原链表中将 org_node 的next 构成的 ListNodeCompare 推入优先队列
            if org_node.next:
                new_node = ListNodeCompare(org_node.next)
                heapq.heappush(heap, new_node)
        return newHead.next
```



# 字符串

## 双指针

# 动态规划

## 实际应用

### [最长定差子序列](https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/)

#### 题目

给你一个整数数组 arr 和一个整数 difference，请你找出并返回 arr 中最长等差子序列的长度，该子序列中相邻元素之间的差等于 difference 。

子序列 是指在不改变其余元素顺序的情况下，通过删除一些元素或不删除任何元素而从 arr 派生出来的序列。

#### 解题思路（DP巧解）

1）技巧

​	从左往右遍历 \textit{arr}arr，并计算出以 \textit{arr}[i]arr[i] 为结尾的最长的等差子序列的长度，取所有长度的最大值，即为答案

​	由于总是在左侧找一个最近的等于arr[i]−d 元素并取其对应 dp 值，因此我们直接用dp[v] 表示以 v 为结尾的最长的等差子序列的长度，这样 **dp[v−d]  就是我们要找的左侧元素对应的最长的等差子序列的长度**，因此转移方程可以改为 <font color=red>**dp[v]=dp[v−d]+1**</font>
2）代码

```python
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        # defaultdict占用内存比res.get(num - difference, 0)大, 推荐后者
        # res = collections.defaultdict(int)  
        res = dict()
        for num in arr:
            res[num] = res.get(num - difference, 0) + 1
        return max(res.values())
```

# 字典树/前缀树Trie

## Trie树实现 （dict / TrieNode）

- 题目

  Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如<font color=red>**自动补完和拼写检查**</font>。

  请你实现 Trie 类：

  Trie() 初始化前缀树对象。
  void insert(String word) 向前缀树中插入字符串 word 。
  boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
  boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。

- 思路

  以嵌套字典替代树，我们从字典树的根开始，查找前缀。对于当前字符对应的子节点，有两种情况：

  - 子节点存在。沿着指针移动到子节点，继续搜索下一个字符。
  - 子节点不存在。说明字典树中不包含该前缀，返回空指针。

  重复以上步骤，直到返回空指针或搜索完前缀的最后一个字符。

  若搜索到了前缀的末尾，就说明字典树中存在该前缀。此外，若前缀末尾对应节点的 end 属性 为真，则说明字典树中存在该字符串。

- Trie示例

  ```python
  # {'a': {'p': {'p': {'l': {'e': {'end': True}}, 'end': True}, 'o': {'l': {'o': {'g': {'e': {'end': True}}}}}}}}
  
  {
    "a": {
      "p": {
        "p": {
          "l": {
            "e": {
              "end": true
            }
          },
          "end": true
        },
        "o": {
          "l": {
            "o": {
              "g": {
                "e": {
                  "end": true
                }
              }
            }
          }
        }
      }
    }
  }
  ```


### dict实现

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = dict()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur_node = self.root
        for char in word:
            if char not in cur_node:
                cur_node[char] = dict()
            # 移动指针当前节点到下一个节点
            cur_node = cur_node[char]
        # 最后一个节点，即单词结束的节点，end 为key, value为True, 表示单词在该节点结束
        cur_node['end'] = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        cur_node = self.root
        for char in word:
            if char not in cur_node:
                return False
            cur_node = cur_node[char]
        return 'end' in cur_node

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        cur_node = self.root
        for char in prefix:
            if char not in cur_node:
                return False
            cur_node = cur_node[char]
        return True
```

### TrieNode实现

本质还是用dict实现

Trie，又称前缀树或字典树，是一棵有根树，其每个节点包含以下字段：

- 指向子节点的指针数组 <font color=red>children</font>。对于本题而言，数组长度为 26，即小写英文字母的数量。此时 children[0] 对应小写字母 a，children[1] 对应小写字母 b，…，children[25] 对应小写字母 z。
- 布尔字段 <font color=red>isEnd</font>，表示该节点是否为字符串的结尾。



```python
class TrieNode:
    def __init__(self):
        self.children = dict()
        self.isEnd = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.isEnd = True

    def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.isEnd

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
```

## [添加与搜索单词 - 数据结构设计](https://leetcode-cn.com/problems/design-add-and-search-words-data-structure/) (字典树搜索)

- 题目

  ```
  请你设计一个数据结构，支持 添加新单词 和 查找字符串是否与任何先前添加的字符串匹配 。
  
  实现词典类 WordDictionary ：
  
  WordDictionary() 初始化词典对象
  void addWord(word) 将 word 添加到数据结构中，之后可以对它进行匹配
  bool search(word) 如果数据结构中存在字符串与 word 匹配，则返回 true ；否则，返回  false 。word 中可能包含一些 '.' ，每个 . 都可以表示任何一个字母。
  ```

- 思路

  字典树+DFS：DFS递归遍历节点

  递归分成两种情况讨论:

  - 当前字符不为.
  - 当前字符为.

  对于第一种情况，如果当前字符不为., 我们正常判断即可：

  - 如果当前字符存在于前缀树中，那么把当前的指针指向其孩子对应字符的节点，并且起始索引+1（指向下一个字符）。
  - 如果不存在前缀树中，可以直接返回False

  那么对于第二种情况，因为.是一个通配符，我们需要遍历当前层所有可能的字符：

  - 如果该层存在字符，往下一层搜索，如果下一层不存在满足的字符（比如下一层没有字符了，或者下一层所有的字符都不满足，或者下一层不存在指定的字符）则返回False
  - 如果已经发现了满足条件的单词，我们直接剪枝，避免后续无效迭代（见代码注释）
  - 如果没有满足的情况再返回False

### TrieNode+DFS

```PYTHON
class TrieNode:
    def __init__(self):
        self.children = dict()
        self.isEnd = False


class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.isEnd = True

    def search(self, word: str) -> bool:
        
        def dfs(node, word, cur_idx):
            # 终止条件
            if cur_idx == len(word):
                return True and node.isEnd
            
            for idx in range(cur_idx, len(word)):
                # 递归，不同分支不同递归逻辑
                if word[idx] != '.':
                    if word[idx] in node.children:
                        return dfs(node.children[word[idx]], word, idx + 1)
                    else:
                        return False
                else:
                    # 通配符 . 的情况下, 遍历每一个children进行递归，遇到True直接中断迭代
                    for ch_key in node.children:
                        ch_res = dfs(node.children[ch_key], word, idx + 1)
                        if ch_res:
                            return True
                    return False
        
        return dfs(self.root, word, 0)
```



### Trie+BFS

```PYTHON
class WordDictionary:

    def __init__(self):
        self.trie = dict()

    def addWord(self, word: str) -> None:
        node = self.trie
        for c in word:
            if c not in node:
                node[c] = dict()
            node = node[c]
        node['#'] = dict()					# BFS 中 这里与之前构建Trie不同

    def search(self, word: str) -> bool:
        word += '#'							# 手动架上结尾字符
        queue = collections.deque()
        queue.append((0, self.trie))
        while queue:
            idx, node = queue.popleft()
            if idx == len(word):			# 终止条件 idx == len(word)
                return True
            if word[idx] != '.':
                if word[idx] in node:
                    queue.append((idx + 1, node[word[idx]]))
            else:
                for ch_node in node.values():
                    queue.append((idx + 1, ch_node))
        return False
```



# 系统设计类

## LRU缓存 （哈希-双向链表）

- 题目：[LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/)

  运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制 。
  实现 LRUCache 类：

  LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
  int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
  void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。


  进阶：你是否可以在 O(1) 时间复杂度内完成这两种操作？

- 示例

  ```
  输入
  ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
  [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
  输出
  [null, null, null, 1, null, -1, null, -1, 3, 4]
  
  解释
  LRUCache lRUCache = new LRUCache(2);
  lRUCache.put(1, 1); // 缓存是 {1=1}
  lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
  lRUCache.get(1);    // 返回 1
  lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
  lRUCache.get(2);    // 返回 -1 (未找到)
  lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
  lRUCache.get(1);    // 返回 -1 (未找到)
  lRUCache.get(3);    // 返回 3
  lRUCache.get(4);    // 返回 4
  ```

- 思路;

  哈希-双向链表，缓存cache的哈希表(value是双向链表节点) 与 双向链表的节点移动需要同步进行

- 代码

  ```python
  class LRUCache:
  
      def __init__(self, capacity: int):
          self.capacity = capacity
          self.cache = dict()                     # value是 DoubleLinkedList
          self.__head = DoubleLinkedList()        # 最近刚使用的
          self.__tail = DoubleLinkedList()        # 最久未使用的
          self.__head.next = self.__tail
          self.__tail.prev = self.__head        
  
      def get(self, key: int) -> int:
          if key not in self.cache:
              return -1
          # key-val 节点 移动到链表的头结点
          self.move_node_to_head(self.cache[key])
          return self.cache[key].val
  
      def put(self, key: int, value: int) -> None:
          if key not in self.cache:
              # key不存在 如果长度不小于cap，删除尾节点key对应的缓存, 同时删除尾节点
              # 新增节点到 头结点,同时新增缓存key-val
              if len(self.cache) >= self.capacity:
                  k = self.delete_tail_node()
                  self.cache.pop(k)
              node = self.add_new_node_to_head(DoubleLinkedList(key, value))
              self.cache[key] = node
          else:
              # 更新缓存 key-val, 同时更新链表节点值且移动该节点到头结点
              node = self.move_node_to_head(self.cache[key], value)
              self.cache[key] = node
  
      def move_node_to_head(self, node, value=None):
          """
          已存在的节点更新value后移动到链表头结点
          """
          if value:
              node.val = value
          node.next.prev = node.prev
          node.prev.next = node.next
          new_node = self.add_new_node_to_head(node)
          return new_node
  
      def add_new_node_to_head(self, node):
          """
          新的节点直接添加到链表头结点
          """
          node.next = self.__head.next
          node.prev = self.__head
          self.__head.next.prev = node
          self.__head.next = node
          return node
      
      def delete_tail_node(self):
          """
          删除尾节点，即 LRU 节点, 返回key用来删除cache对应的key-val
          """
          node = self.__tail.prev
          self.__tail.prev = node.prev
          node.prev.next = self.__tail
          return node.key
  ```

## [设计搜索自动补全系统](https://leetcode-cn.com/problems/design-search-autocomplete-system/)  (前缀树Trie)  -- 未完成

- 题目

  ```reStructuredText
  为搜索引擎设计一个搜索自动补全系统。用户会输入一条语句（最少包含一个字母，以特殊字符 '#' 结尾）。除 '#' 以外用户输入的每个字符，返回历史中热度前三并以当前输入部分为前缀的句子。下面是详细规则：
  
  一条句子的热度定义为历史上用户输入这个句子的总次数。
  返回前三的句子需要按照热度从高到低排序（第一个是最热门的）。如果有多条热度相同的句子，请按照 ASCII 码的顺序输出（ASCII 码越小排名越前）。
  如果满足条件的句子个数少于 3，将它们全部输出。
  如果输入了特殊字符，意味着句子结束了，请返回一个空集合。
  你的工作是实现以下功能：
  
  构造函数：
  
  AutocompleteSystem(String[] sentences, int[] times): 这是构造函数，输入的是历史数据。 Sentences 是之前输入过的所有句子，Times 是每条句子输入的次数，你的系统需要记录这些历史信息。
  
  现在，用户输入一条新的句子，下面的函数会提供用户输入的下一个字符：
  
  List<String> input(char c): 其中 c 是用户输入的下一个字符。字符只会是小写英文字母（'a' 到 'z' ），空格（' '）和特殊字符（'#'）。输出历史热度前三的具有相同前缀的句子。
  ```

- 思路

- 代码

  ```python
  
  ```

## [设计推特](https://leetcode-cn.com/problems/design-twitter/) （合并K个有序链表--多路归并 / 哈希表）

- 题目

  ```
  设计一个简化版的推特(Twitter)，可以让用户实现发送推文，关注/取消关注其他用户，能够看见关注人（包括自己）的最近 10 条推文。
  
  实现 Twitter 类：
  
  Twitter() 初始化简易版推特对象
  void postTweet(int userId, int tweetId) 根据给定的 tweetId 和 userId 创建一条新推文。每次调用次函数都会使用一个不同的 tweetId 。
  List<Integer> getNewsFeed(int userId) 检索当前用户新闻推送中最近  10 条推文的 ID 。新闻推送中的每一项都必须是由用户关注的人或者是用户自己发布的推文。推文必须 按照时间顺序由最近到最远排序 。
  void follow(int followerId, int followeeId) ID 为 followerId 的用户开始关注 ID 为 followeeId 的用户。
  void unfollow(int followerId, int followeeId) ID 为 followerId 的用户不再关注 ID 为 followeeId 的用户。
  ```

- 思虑1： 所有好友的top10放在一起排序再取top10 （数据量小可以，海量数据不适合）

- 思路2：有序链表多路归并

### 哈希表

```python
class Twitter:

    def __init__(self):
        self.user_follow = collections.defaultdict(set)
        self.user_post = collections.defaultdict(list)
        self.auto_incre = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.user_post[userId].append((self.auto_incre, tweetId))
        self.auto_incre += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        friends = self.user_follow[userId]
        tweet_tops = self.user_post[userId][-10:]
        for friend in friends:
            tweet_tops.extend(self.user_post[friend][-10:])
        tweet_tops.sort(key=lambda x: -x[0])
        return [item[1] for item in tweet_tops[:10]]
        
    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.user_follow[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.user_follow[followerId]:
            self.user_follow[followerId].remove(followeeId)
```

### 哈希表+链表多路归并+优先队列  

链表多路归并采用优先队列获取top-N

- 思路

  这里需求 3 和需求 4，只需要维护「我关注的人的 id 列表」 即可，不需要维护「谁关注了我」，由于不需要维护有序性，为了删除和添加方便， 「我关注的人的 id 列表」需要设计成哈希表（HashSet），而每一个人的和对应的他关注的列表存在一个哈希映射（HashMap）里；
  最复杂的是需求 2 getNewsFeed(userId):
  每一个人的推文和他的 id 的关系，依然是存放在一个哈希表里；
  对于每一个人的推文，**只有顺序添加的需求，没有查找、修改、删除操作，因此可以使用线性数据结构**，链表或者数组均可；

  - 使用数组就需要在尾部添加元素，还需要考虑扩容的问题（使用动态数组）；

  - 使用链表就得在头部添加元素，由于链表本身就是动态的，无需考虑扩容；

  检索最近的十条推文，需要先把这个用户关注的人的列表拿出来（实际上不是这么做的，请看具体代码，或者是**「多路归并」**的过程），然后再合并，排序以后选出 Top10，这其实是非常经典的「多路归并」的问题（「力扣」第 23 题：合并K个排序链表），这里需要使用的数据结构是**优先队列**，所以在上一步在存储推文列表的时候使用单链表是合理的，并且应**该存储一个时间戳字段，用于比较哪一队的队头元素先出列**。

- 总结：

  如果需要维护数据的时间有序性，链表在特殊场景下可以胜任。因为时间属性通常来说是相对固定的，而不必去维护顺序性；
  如果需要<font color=red>动态维护数据有序性，「优先队列」（堆）是可以胜任的</font>，「力扣」上搜索「堆」（heap）标签，可以查看类似的问题；

  

```python
class Tweet:
    def __init__(self, tweetId, timestamp):
        self.id = tweetId
        self.timestamp = timestamp
        self.next = None
    
    # 定义富比较函数 小于，确保结构体Tweet 可直接用与 堆排列
    def __lt__(self, other):
        return self.timestamp > other.timestamp

class Twitter:

    def __init__(self):
        self.timestamp = 0
        self.tweets = collections.defaultdict(lambda: None)     # 默认没发推的返回值是None
        self.friends = collections.defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        tweet = Tweet(tweetId, self.timestamp)
        # 新Tweet节点添加到用户链表头
        tweet.next = self.tweets[userId]
        # 更新 tweets 中保存的用户链表头节点为当前新增的节点(只保留头结点，不保留所有文章)
        self.tweets[userId] = tweet
        self.timestamp += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        # 优先队列heapq每次比较自己与关注的用户的头结点中的最小Tweet(对应时间戳最大，即最新节点)
        # 取够十个就可以了
        tops = []               # 保存返回值
        hints = []              # 动态保存优先队列的堆顶元素
        cur_user_tweet = self.tweets[userId]        # 可能没发推即None
        if cur_user_tweet:
            hints.append(cur_user_tweet)
        for user in self.friends[userId]:
            cur_user_tweet = self.tweets[user]
            if cur_user_tweet:
                hints.append(cur_user_tweet)
        # 朋友圈所有人包括自己的tweet文章的头结点组成的列表hints堆化
        heapq.heapify(hints)
        
        # 动态从hints中取堆顶元素(这里的小顶堆堆顶元素的时间戳最大，即最新tweet)
        while hints and len(tops) < 10:
            # 堆顶元素取出添加到 tops 中
            top = heapq.heappop(hints)
            tops.append(top.id)
            # 取出后将top.next添加到优先队列hints中
            if top.next:
                heapq.heappush(hints, top.next)
        return tops

    def follow(self, followerId: int, followeeId: int) -> None:
        if followeeId != followerId:
            self.friends[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.friends[followerId]:
            self.friends[followerId].remove(followeeId)


# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)
```

说明：

- `self.tweets` 是用于存放用户与其发布的 tweet 的，默认发布的 tweet 是 `None`，但如果直接声明成 `self.tweets = defaultdict(None)` 其实相当于默认值为 None，即没有默认值，这样 defaultdict 还是和 dict 相同，会报 KeyError 的错。所以要声明成 `self.tweets = defaultdict(lambda: None)`，`lambda: None` 意味着会返回 None

  ```python
  >> a = collections.defaultdict(lambda: None)
  >> print(a['b'])
  None
  
  >> a = collections.defaultdict(None)
  >> print(a['b'])
  Traceback (most recent call last):
    File "C:\Users\Administrator\Anaconda3\envs\py36\lib\site-packages\IPython\core\interactiveshell.py", line 3343, in run_code
      exec(code_obj, self.user_global_ns, self.user_ns)
    File "<ipython-input-19-657ea3224092>", line 1, in <module>
      print(a['b'])
  KeyError: 'b'
  ```

  

- heapq 默认是最小堆，我们将用户的 Tweet 放入这个最小堆中为了获取最新的 Tweet 即 `timestamp` 最大的那些 Tweet，而最小堆会 pop 它认为「最小」的元素，通过重载 `__lt__` 我们定义了 Tweet 实例之间「更小」的概念，即 `timestamp` 更大，也就是「更新」。

## 设计地铁系统

- 题目

  ```
  请你实现一个类 UndergroundSystem ，它支持以下 3 种方法：
  
  1. checkIn(int id, string stationName, int t)
  
  编号为 id 的乘客在 t 时刻进入地铁站 stationName 。
  一个乘客在同一时间只能在一个地铁站进入或者离开。
  2. checkOut(int id, string stationName, int t)
  
  编号为 id 的乘客在 t 时刻离开地铁站 stationName 。
  3. getAverageTime(string startStation, string endStation) 
  
  返回从地铁站 startStation 到地铁站 endStation 的平均花费时间。
  平均时间计算的行程包括当前为止所有从 startStation 直接到达 endStation 的行程。
  调用 getAverageTime 时，询问的路线至少包含一趟行程。
  你可以假设所有对 checkIn 和 checkOut 的调用都是符合逻辑的。也就是说，如果一个顾客在 t1 时刻到达某个地铁站，那么他离开的时间 t2 一定满足 t2 > t1 。所有的事件都按时间顺序给出。
  ```

- 思路

  表1：根据 id 来保存上车 车站名字 和 时间
  表2：在下车车站计算累积线路乘车时间，和线路乘车次数
  取得平均时间时，只需要查询两个车站，得到总时间，和总次数

- 代码

  ```python
  class UndergroundSystem:
  
      def __init__(self):
          self.startInfo = dict()
          self.routines = dict()
  
      def checkIn(self, id: int, stationName: str, t: int) -> None:
          self.startInfo[id] = (stationName, t)
  
      def checkOut(self, id: int, stationName: str, t: int) -> None:
          start_sta, start_time = self.startInfo[id]
          if (start_sta, stationName) not in self.routines:
              t_consum = t - start_time
              self.routines[(start_sta, stationName)] = [t_consum, 1]
          else:
              self.routines[(start_sta, stationName)][0] += (t - start_time)
              self.routines[(start_sta, stationName)][1] += 1
  
      def getAverageTime(self, startStation: str, endStation: str) -> float:
      	route = (startStation, endStation)
          return self.routines[route][0] / self.routines[route][1]
  ```

#### [设计文件分享系统](https://leetcode-cn.com/problems/design-a-file-sharing-system/)

```python
class FileSharing:

    def __init__(self, m: int):
        self.users = collections.defaultdict(list)
        self.chunks = collections.defaultdict(set)
        self.id = 1
        self.pools = []

    def join(self, ownedChunks: List[int]) -> int:
        if self.pools:
            id = heapq.heappop(self.pools)
        else:
            id = self.id
            self.id += 1
        self.users[id] = ownedChunks
        for chunk in ownedChunks:
            self.chunks[chunk].add(id)
        return id

    def leave(self, userID: int) -> None:
        for _chunk in self.users[userID]:
            self.chunks[_chunk].remove(userID)
        self.users.pop(userID)
        heapq.heappush(self.pools, userID)

    def request(self, userID: int, chunkID: int) -> List[int]:
        ans = list(self.chunks[chunkID])
        if not ans:
            return ans
        ans.sort()
        if userID not in self.chunks[chunkID]:
            # 如果用户收到 ID 的非空列表，就表示成功接收到请求的文件块, 对应更新users和chunks
            self.chunks[chunkID].add(userID)
            self.users[userID].append(chunkID)
        return ans
```



# 附录

## 常识性问题

### 判断一个数是否为2的幂次方？

二进制表示的2的幂次方数中只有一个1，后面跟的是n个0； 因此问题可以转化为判断1后面是否跟了n个0。如果将这个数减去1后会发现，仅有的那个1会变为0，而原来的那n个0会变为1；因此将原来的数与上(&)减去1后的数字，结果为零。

```python
def two_power_check(n):
	return n & (n - 1) == 0
```

- 判断是否为4的幂次方

  ```python
  def isPowerOfFour(n):
      if n <= 0:
      	return False
      m = int(math.sqrt(n))
      return m * m == n and (m & (m - 1) == 0)
  # 或者 n /= 4   n % 4 ==0
  ```

### 全排列 itertools.permutations

```python
In[41] : list(itertools.permutations([1,4,5], 3))
Out[41]: [(1, 4, 5), (1, 5, 4), (4, 1, 5), (4, 5, 1), (5, 1, 4), (5, 4, 1)]
```





