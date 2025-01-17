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

        