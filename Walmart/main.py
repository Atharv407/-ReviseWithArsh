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
class Solution: 		
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        graph = defaultdict(list)
        
        stack = [(root, None)]
        while stack: 
            n, p = stack.pop()
            if p: 
                graph[p.val].append(n.val)
                graph[n.val].append(p.val)
            if n.left: stack.append((n.left, n))
            if n.right: stack.append((n.right, n))
        
        ans = -1
        seen = {start}
        queue = deque([start])
        while queue: 
            for _ in range(len(queue)): 
                u = queue.popleft()
                for v in graph[u]: 
                    if v not in seen: 
                        seen.add(v)
                        queue.append(v)
            ans += 1
        return ans 

#Largest divisible subset
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        res = 0
        nums.sort()
        l = [set() for i in nums]
        # l[]
        for i in range(0, len(nums)) :
            # print(l)
            for j in range(i-1, -1, -1) :
                # print(nums[i], nums[j])
                if nums[i] % nums[j] == 0  and len(l[i]) < len(l[j]):
                    # print('in')
                    l[i] = l[j].copy()
                    # break
            l[i].add(nums[i])
            if len(l[i]) > len(l[res]) : res = i
        return list(l[res])