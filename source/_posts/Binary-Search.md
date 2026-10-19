---
title: "Binary Search"
date: 2025-10-27 13:16:05 +08:00
categories:
  - 算法
---

## condition
数组为有序数组，同时题目数组中无重复元素

## theoretical key point
将一个数组一分为二，每次进行左/右查找

## coding key point
边界需要注意（左开右闭or左右皆闭），中间指针需要注意

## example
link：https://leetcode.cn/problems/binary-search/description/

original code:
```
# 1 2 3 4 5 6 7 #array
# 0 1 2 3 4 5 6 #address
class Solution:
    def search(self, nums: List[int], target: int) -&gt; int:
        left_p = now_p = 0
        right_p = len(nums) - 1
        
        while(left_p &lt;= right_p):
            now_p = (left_p + right_p) // 2
            if (nums[now_p] == target):
                return now_p
            elif (nums[now_p] &lt; target):
                left_p = now_p + 1
            elif (nums[now_p] &gt; target):
                right_p = now_p - 1
                
        return -1
```
1. 定义左右指针，注意右指针为``len()-1``，因为数组实际长度是大于地址一个单位的
2. while的循环条件为<=（小于说明没有搜索完，还有可以搜索的下标；等于通常为最后一步，即两指针重合的情况，**不可忽略**)
3. 循环内首先定义中间指针，一般直接相加除2即可。接着需要做分类讨论，定义新的左/右指针，需要注意的是**新指针需要+-1，因为中间指针已经搜索过了，不需要再reseek一次，不然会导致死循环无法退出**
4. 搜到了就马上返回，搜不到就返回-1
