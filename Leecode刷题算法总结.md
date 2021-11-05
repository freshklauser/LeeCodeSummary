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

```python
class Solution:
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
```

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

```python
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





