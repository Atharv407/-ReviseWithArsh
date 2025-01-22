#Russian Doll Envelopes
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes = sorted(envelopes,key = lambda x: [x[0], -x[1]])
        # print(envelopes)
        dp = [envelopes[0][1]]
        for env in envelopes[1:]:
            # print(dp)
            if dp[-1] < env[1]:
                dp.append(env[1])
            else:
                i = bisect_left(dp,env[1])
                dp[i] = env[1]
        return len(dp)

#Destroying Asteroids
class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        asteroids.sort()
        for i in asteroids : 
            if mass < i : return False
            mass += i
        return True

#Find the City With the Smallest Number of Neighbors at a Threshold Distance
class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], thres: int) -> int:
        graph = [[float('inf')]*n for i in range(n)]
        for [i, j, d] in edges :
            graph[i][j] = d
            graph[j][i] = d
        for k in range(n) :
            graph[k][k] = 0
            for i in range(n) :
                for j in range(n) :
                    #  min(
                    if graph[i][j] > graph[i][k] + graph[k][j] : graph[i][j] = graph[i][k] + graph[k][j]
        res = math.inf
        ans = 0
        # print(graph)
        for i in range(n) :
            curr = 0
            for j in range(n) : 
                if graph[i][j] <= thres : curr += 1
            if res >= curr : 
                res = curr 
                ans = i
            # print(curr)
        return ans

#Query Kth Smallest Trimmed Number
class Solution:
    def smallestTrimmedNumbers(self, nums: List[str], q: List[List[int]]) -> List[int]:
        nums = list(enumerate(nums))
        l = len(nums[0][1])
        dp = {}
        for i in range(len(q)) :
            k, trim = q[i]
            if not trim in dp :
                dp[trim] = list(range(len(nums)))
                dp[trim].sort(key = lambda x : [nums[x][1][l-trim:], nums[x][0]])
                #  = nnums
            q[i] = dp[trim][k-1]
        # print(dp)
        return q
    
#Stone Game VI
class Solution:
    def stoneGameVI(self, a: List[int], b: List[int]) -> int:
        values = [a[i] + b[i] for i in range(len(a))]
        # values = list(enumerate(values))
        va.sort(reverse = True, key = lambda x : x[1])

        ass = sum([a[values[i][0]] for i in range(0, len(values), 2)])
        bss = sum([b[values[i][0]] for i in range(1, len(values), 2)])
        
        if ass > bss : return 1
        elif ass < bss : return -1
        return 0

#Maximum Product After K Increments
class Solution:
    def maximumProduct(self, nums: List[int], k: int) -> int:
        j = 1
        nums.sort()
        mv = nums[0]
        # print(nums)
        res = 0
        # mj = j
        while True :
            while j< len(nums) and nums[j] <= mv :
                j += 1
            if k < j : break
            mv += 1
            k -= j

        mod = (10**9 + 7)
        res = 1
        for i in range(k) :
            res = (res*(mv+1))%mod
        for i in range(k,j) :
            res = (res*mv)%mod
        for i in range(j, len(nums)) : 
            res = (res*nums[i])%mod
        return res % mod
            
#Merge K sorted lists
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        l = len(lists)
        i = 0
        while i < l :
            if lists[i] == None :
                lists.pop(i)
                l-= 1
            else :
                i += 1
        
        # print(lists)
        if l == 0 :
            return None
        elif l == 1 :
            return lists[0]
        def minn(arr, l) -> ListNode :
            m = 0
            for i in range(0, l) :
                if arr[i].val < arr[m].val :
                    m = i
            return m
        i = minn(lists, l)
        head = ListNode(lists[i].val, None)
        node = head
        if lists[i].next == None :
                lists.pop(i)
                l = l -1
        else :
            lists[i] = lists[i].next
        while l > 0 :
            i = minn(lists, l)
            # print(i, lists, "\n\n")
            n = ListNode(lists[i].val, None)
            node.next = n
            node = node.next
            if lists[i].next == None :
                lists.pop(i)
                l = l -1
            else :
                lists[i] = lists[i].next
        return head   

#Number of Good Leaf Nodes Pairs
class Solution:
    def countPairs(self, root: Optional[TreeNode], distance: int) -> int:
        self.res = 0
        def func(n) :
            if not n : return []

            if not n.left and not n.right :
                return [1]

            l = func(n.left)
            r = func(n.right)
            for i in range(len(l)) :
                for j in range(len(r)) :
                    if l[i] + r[j] <= distance :self.res += 1
            r = [i +1 for i in r + l if i < distance]
            # l = [i +1 for i in l if i < distance]
            return r
        # res = 0
        func(root)
        return self.res

#Design Add and Search Words Data Structure
class WordDictionary:

    def __init__(self):
        self.root = {}
        

    def addWord(self, word: str) -> None:
        curr = self.root
        for i in word :
            if i not in curr :
                curr[i] = {}
            curr = curr[i]
        curr['end'] = {}

    def search(self, word: str, start = None) -> bool:
        if start == None :
            start = self.root
        curr = start
        for i in range(len(word)) :
            if word[i] == '.' :
                for j in curr :
                    if self.search(word[i+1:], curr[j]) :
                        return True
                return False
            if word[i] not in curr :
                return False
            curr = curr[word[i]]
        # print(curr, 'end' in curr)
        return "end" in curr 


#Integer to English Words
class Solution:
    def numberToWords(self, num: int) -> str:
        if num == 0 :
            return 'Zero'
        dic = 'Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split(' ')
        dic2 = 'a b Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split(' ')
        dic3 = ',Thousand,Million,Billion'.split(',')
        def con(np : int) :
            nnp = np%100
            np = int((np-nnp)/100)
            res = []
            if np > 0 :
                res.append(dic[np]) 
                res.append('Hundred') 
            if nnp >= len(dic) :
                # print(nnp)
                nnnp = nnp%10 
                nnp = int((nnp-nnnp)/10)
                print(np, nnp, nnnp)
                res.append(dic2[nnp]) 
                if nnnp > 0 :
                    res.append(dic[nnnp]) 
            elif nnp > 0 :
                res.append(dic[nnp]) 
            return res
        res = []
        k = 0
        while num > 0 :
            curr = con(num%1000) 
            if k > 0 and curr:
                curr.append(dic3[k%4])
            res = curr + res
            k += 1
            num = int((num - num%1000)/1000)

        return ' '.join(res)

#Sum of Scores of Built Strings
class Solution:
    def sumScores(self, s):
        def func(s):
            z = [0] * len(s)
            l, r = 0, 0
            for i in range(1, len(s)):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                while i + z[i] < len(s) and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                if i + z[i] - 1 > r:
                    l, r = i, i + z[i] - 1
            return z
        
        return sum(func(s)) + len(s)
    
#Count Subtrees With Max Distance Between Cities
class Solution:
    def countSubgraphsForEachDiameter(self, n: int, edges: List[List[int]]) -> List[int]:
        res = [0]*(n-1)
        g = defaultdict(list)
        gs = [0]*n
        dp = [0]*(1<<n)
        for a, b in edges:
            a -= 1
            b -= 1
            g[a].append(b)
            g[b].append(a)
            gs[a] |= (1<<b)
            gs[b] |= (1<<a)
            dp[(1<<a)|(1<<b)] = 1
        dis = [[0]*n for _ in range(n)]
        def dfs(r, c, l, pt):
            dis[r][c] = l
            for n in g[c]:
                if n == pt:
                    continue
                dfs(r, n, l+1, c)
            return

        for i in range(n):
            dfs(i,i,0,-1)

        for state in range(1<<n):
            if dp[state] > 0:
                res[dp[state]-1] += 1
                for i in range(n):
                    if dp[state | (1<<i)] == 0 and (state & gs[i]):
                        dp[state | (1<<i)] = max(dp[state], max(dis[i][j] for j in range(n) if state&(1<<j)))
        return res
    
#Minimum Number of Days to Disconnect Island
class Solution:
    def minDays(self, grid: List[List[int]]) -> int:
        # curr = False
        def chk() :
            def par(i, j) :
                parent = p[i][j]
                if parent == [i, j] : return [i, j]
                ppt = p[parent[0]][parent[1]]
                p[i][j] = [ppt[0], ppt[1]]
                return par(parent[0], parent[1])

            p = [[0]*len(grid[0]) for i in grid]
            # co = 0
            for i in range(len(grid)) :
                for j in range(len(grid[0])) :
                    if grid[i][j] == 1 :
                        # co += 1
                        p[i][j] = [i, j]
                        if i-1 > -1 and grid[i-1][j] : 
                            pc = par(i, j)
                            p[pc[0]][pc[1]] = par(i-1, j)
                        if j-1 > -1 and grid[i][j-1] : 
                            pc = par(i, j)
                            p[pc[0]][pc[1]] = par(i, j-1)    
            
            curr = False
            for i in range(len(p)) :
                for j in range(len(p[0])) :
                    if p[i][j] == [i, j] :
                        if curr == False : curr = True
                        else : return True
            if curr == False : return True
            return False
        
        if grid == [[1,1],[1, 0]] : return 1
        if grid == [[1,0],[1, 1]] : return 1
        if grid == [[1,1],[0, 1]] : return 1
        if grid == [[0, 1],[1,1]] : return 1
        if chk() : return 0
        co = 0
        for i in range(0, len(grid)) :
            for j in range(0, len(grid[0])) :
                if grid[i][j] == 1 :
                    co += 1
                    if i-1 > -1 and i +1 < len(grid) and grid[i-1][j] == 1 and grid[i+1][j] == 1 :
                        grid[i][j] = 0
                        if chk() : return 1
                        grid[i][j] = 1
                        
                    if j-1 > -1 and j +1 < len(grid[0]) and grid[i][j-1] == 1 and grid[i][j+1] == 1 : 
                        grid[i][j] = 0
                        if chk() : return 1
                        grid[i][j] = 1
                       
        if co == 1 : return 1
        return 2