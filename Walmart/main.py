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
    
#K-diff pairs in an array
class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        if k == 0 :
            nums = Counter(nums)
            res = 0
            for i in nums :
                if nums[i] > 1 : 
                    res += 1
            return res
        nums = set(nums)
        res = 0
        cnums = nums.copy()
        for i in nums : 
            if i+k in cnums :
                # print(i, i+k)
                res += 1
            if i-k in  cnums : 
                # print(i, i+k)
                res += 1
            cnums.remove(i)
        # print(res)
        return res
    
#Rotate function
class Solution:
    def maxRotateFunction(self, nums: List[int]) -> int:
        res = 0
        for i in range(len(nums)) :
            res += nums[i]*i
        curr = res
        s = sum(nums)
        for i in range(len(nums)-1, 0, -1) :
            # nums = nums[-1:] + nums[:-1]
            curr = curr + s - nums[i]*len(nums)
            if curr > res : res = curr
            # res = max(res, curr)
        return res

#Get Equal Substring Within Budget
class Solution :
    def equalSubstring(self, s, t, cost):
        i = 0
        for j in range(len(s)):
            cost -= abs(ord(s[j]) - ord(t[j]))
            if cost < 0:
                cost += abs(ord(s[i]) - ord(t[i]))
                i += 1
        return j - i + 1

#Word break
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool :
        ld = len(wordDict)
        fallacies = []
        def wb(s) -> bool :
            ls = len(s)
            if ls == 0 :
                return True
            elif s in fallacies :
                return False
            # print(wordDict)
            for ss in wordDict :
                lss = len(ss)
                if ss == s[:lss] :
                    if wb(s[lss:]) == True :
                        return True
            fallacies.append(s)
            return False
        return wb(s)

#Top k Frequent Words
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        c = Counter(words)
        for ki in c : 
            c[ki] = [-c[ki], ki]
        words = list(c.keys())
        words.sort(key = lambda x : c[x])
        return words[:k]
    
#Sort Characters by Frequency 
class Solution:
    def frequencySort(self, s: str) -> str:
        s = list(s)
        # print(s)
        c = Counter(s)
        # print(c)
        s = list(set(s))
        s.sort(key = lambda x : c[x], reverse = True)
        res = ''
        for i in s : 
            res += i*c[i]
        return res

#Friends Of Appropriate Ages
class Solution:
    def numFriendRequests(self, ages):
        def request(x, y):
            return not (y <= 0.5 * x + 7 or y > x or y > 100 and x < 100)
        c = Counter(ages)
        return sum(request(a, b) * c[a] * (c[b] - (a == b)) for a in c for b in c)