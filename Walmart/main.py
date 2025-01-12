#Find subsequence of length k with largest sum
class Solution:
    def maxSubsequence(self, nums: List[int], k: int) -> List[int]:
        nums = list(enumerate(nums))
        # print(nums)
        nums.sort(reverse = True, key = lambda x : x[1])
        nums = nums[:k]
        nums.sort(key = lambda x : x[0])
        return [i[1] for i in nums]

#Amount of time for binary tree to be infected
