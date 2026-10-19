---
title: "Fast & Slow Pointers"
date: 2025-10-27 13:15:03 +08:00
categories:
  - 算法
---

## condition
null

## theoretical key point
双指针的一个分支，两个指针同时从左边开始，满足条件的slow point可以+=1，fast point在通常情况下需要不断的加1

## coding key point
1. 一个大while，外层保证fast指针+=1，内层需要满足一点的条件再加1
2. while循环的终止条件需要用到fast point，因为fast往往可以走完全程，如果用slow point充当判断条件的话会导致out of index问题

## example
### 1. remove element
link：https://leetcode.cn/problems/remove-element/description/

original code：
```
class Solution:
    def removeElement(self, nums: List[int], val: int) -&gt; int:
        SlowIndex = 0
        FastIndex = 0
        while (FastIndex &lt; (len(nums))):
            if nums[FastIndex] != val:
                nums[SlowIndex] = nums[FastIndex]
                SlowIndex += 1
            FastIndex += 1

        return SlowIndex
```
这是一道经典的可用暴力解的问题，快慢指针实际上改变了其复杂度``O(n^2)->O(n)``
1. while判断条件为<，这是一个边界问题，目的在于不能out of index
2. if判断语句一般为!=，接着将fast point的值传递为slow point，可以理解为前者是一个中间变量，后者为一个空数组，fast point做的仅仅是append，最后的输出仅为list[],index=slow

### 2. remove duplicates from sorted array
link：https://leetcode.cn/problems/remove-duplicates-from-sorted-array/description/

original code：
```
class Solution:
    def removeDuplicates(self, nums: List[int]) -&gt; int:
        fast = 0
        slow = 0
        while(fast &lt; len(nums)):
            if (nums[slow] != nums[fast]):
                slow += 1
                nums[slow] = nums[fast]
            fast += 1
        return slow + 1
```
大同小异，只是if判断不一样，需要找到不同的nums[slow]和nums[fast]。不满足一定条件才能移动slow point

### 3. move zeros
link：https://leetcode.cn/problems/move-zeroes/

origin code：
```
class Solution:
    def moveZeroes(self, nums: List[int]) -&gt; None:
        fast = 0
        slow = 0
        while (fast &lt; len(nums)):
            if nums[fast] != 0:
                a = nums[slow]
                nums[slow] = nums[fast]
                nums[fast] = a
                slow += 1
            fast += 1
```
依旧快慢指针，但是需要存中间变量，不然会被覆盖

## 4. backspace string compare
link：https://leetcode.cn/problems/backspace-string-compare/description/

origin code：
```
class Solution:
    def backspaceCompare(self, S: str, T: str) -&gt; bool:
        i, j = len(S) - 1, len(T) - 1
        skipS = skipT = 0

        while i &gt;= 0 or j &gt;= 0:
            while i &gt;= 0:
                if S[i] == &quot;#&quot;:
                    skipS += 1
                    i -= 1
                elif skipS &gt; 0:
                    skipS -= 1
                    i -= 1
                else:
                    break
            while j &gt;= 0:
                if T[j] == &quot;#&quot;:
                    skipT += 1
                    j -= 1
                elif skipT &gt; 0:
                    skipT -= 1
                    j -= 1
                else:
                    break
            if i &gt;= 0 and j &gt;= 0:
                if S[i] != T[j]:
                    return False
            elif i &gt;= 0 or j &gt;= 0:
                return False
            i -= 1
            j -= 1
        
        return True
```
主要是字符串的比较，注意python里不能直接s[0]='r',字符串不是数组是不能被覆盖的
