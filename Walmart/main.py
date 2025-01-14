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

#Maximum length of Repeated Subarray
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        res = 0
        dp = [[0]*len(nums2) for i in nums1]
        for i in range(len(nums1)) :
            for j in range(len(nums2)) :
                if nums1[i] == nums2[j]: 
                    if i-1 > -1 and j - 1 > -1 : 
                        dp[i][j] = dp[i-1][j-1]
                    dp[i][j] += 1
                    if dp[i][j] > res :
                        res = dp[i][j]
        # print(dp)
        return res
    
#Battleships in a Board
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        res = 0
        v = [0]*len(board[0])
        # vr = [0]*len(board)
        h = 0
        for i in range(len(board)) :
            h = 0
            print(res, v)
            for j in range(len(board[0])) :
                if board[i][j] == 'X' :
                    if h == 1 :
                        v[j-1] = 0
                    elif h == 0 :
                        v[j] += 1
                    h += 1
                else : 
                    if h > 1 : 
                        res += 1
                    h = 0
                    if v[j] > 0 : 
                        res += 1
                        v[j] = 0
            if h > 1 : 
                res += 1
                v[-1] = 0
        print(res, h, v)
        
        # else :
        for j in v : 
            if j > 0 : res += 1

        return res 
                    
            
#Verify Preorder Serialization of a Binary Tree
class Solution:
    def isValidSerialization(self, p: str) -> bool:
        def chk(i, j) :
            if p[i] == '#' : return i + 1
            if j==i : return j
            if j-i < 3 : return i

            le = chk(i+1, j)
            i = le

            re = chk(i, j)

            return re
            
        p = p.split(',')
        if chk(0, len(p)) != len(p) :
            return False
        return True


#Count the Number of Square-Free Subsets
class Solution:
    def squareFreeSubsets(self, nums):
        chk = {1:0, 2:1, 3:2, 5:4, 6:3, 7:8, 10:5, 11:16, 13:32, 14:9, 15:6, 17:64, 19:128, 21:10, 22:17, 23:256, 26:33, 29:512, 30:7}
        count = defaultdict(int)
        for n in nums:
            if n in chk:
                for k in count.copy():
                    if chk[n] & k == 0:
                        count[chk[n]|k] += count[k]
                count[chk[n]] += 1
        return sum(count.values()) % (10 ** 9 + 7)
        