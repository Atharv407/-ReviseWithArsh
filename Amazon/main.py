# First non-repeating character in a stream
class Solution:
    def firstUniqChar(self, s: str) -> int:
        chars = Counter(s)
        for k in chars : 
            if chars[k] == 1 : 
                return s.find(k)
        return -1
            
#Longest Mountain
class Solution:
    def longestMountain(self, arr: List[int]) -> int:
        curr = 1
        res = 0
        incr = True
        if len(arr) < 3 : return 0
        for i in range(1, len(arr)) :
            if arr[i] > arr[i-1] :
                if incr == True : curr += 1
                else : 
                    res = max(res, curr)
                    curr = 2
                    incr = True
            elif arr[i] < arr[i-1] : 
                if incr == False : curr += 1
                else : 
                    if curr > 1 :
                        curr += 1
                        incr = False
                    else : curr = 1
            else : 
                if incr == False :
                    res = max(res, curr)
                incr = True
                curr = 1
        if incr == False : res = max(res, curr)
        return res

#Maximum sum of distinct subarrays with length k
class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        subarr = {}
        i = 0
        res = 0
        j = 0
        s = 0
        while i < len(nums)-k+1 : 
            if j < i + k :
                # j += 1
                if nums[j] in subarr :
                    # subarr = set()
                    i = subarr[nums[j]]
                    subarr = {}
                    s = 0
                    j = i - 1
                else : 
                    subarr[nums[j]] = j
                    s += nums[j]
                j += 1
            else : 
                # print(i, j, subarr)
                res = max(res, s)
                subarr.pop(nums[i])
                s -= nums[i]
                i = i + 1
        return res
                
#Delete N nodes after M nodes of a linked list

'''
class Node:
    # Constructor to initialize the node object
    def __init__(self, data):
        self.data = data
        self.next = None
'''
class Solution:
    def linkdelete(self, head, n, m):
        curr = head
        while True :
            # curr = head
            for i in range(m-1) :
                curr = curr.next
                if curr == None : return
            # print(curr.data)
            nex = curr.next
            for j in range(n) :
                nex = nex.next
                if nex == None : return
            curr.next = nex
            # print(nex.data)
            curr = nex



#{ 
 # Driver Code Starts
class Node:
    # Constructor to initialize the node object
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    # Function to initialize head
    def __init__(self):
        self.head = None

    # Function to insert a new node at the beginning
    def push(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node

    # Utility function to print the linked list
    def printList(self):
        temp = self.head
        while temp:
            print(temp.data, end=" ")
            temp = temp.next
        print("")  # Newline after printing the linked list


if __name__ == '__main__':
    t = int(input())  # Number of test cases
    while t > 0:
        llist = LinkedList()
        values = input().strip().split()
        for i in reversed(values):  # Reversed input list to preserve order
            llist.push(i)
        n, m = map(int, input().strip().split())  # n: keep, m: delete
        Solution().linkdelete(llist.head, n,
                              m)  # Call the method to modify the list
        llist.printList()
        t -= 1
        print("~")  # Separator for test cases

#Maximum of all Subarrays of Size k
#User function Template for python3
import heapq
from collections import Counter
class Solution:
    #Function to find maximum of each subarray of size k.
    def maxOfSubarrays(self, arr, k):
        i = 0
        j = i + k
        q = [-i for i in arr[i:j]]
        heapq.heapify(q)
        res = [-q[0]]
        sa = Counter(q)
        for i in range(1, len(arr)-k+1) :
            # print(sa, q)
            heapq.heappush(q,-arr[i+k-1])
            sa[-arr[i-1]] -= 1
            sa[-arr[i+k-1]] += 1
            while sa[q[0]] <= 0 :
                heapq.heappop(q)
                # if len(q) == 0 : return res
            res.append(-q[0])
        return res
            
            


#{ 
 # Driver Code Starts
#Initial Template for Python 3

import atexit
import io
import sys
from collections import deque

#Contributed by : Nagendra Jha

_INPUT_LINES = sys.stdin.read().splitlines()
input = iter(_INPUT_LINES).__next__
_OUTPUT_BUFFER = io.StringIO()
sys.stdout = _OUTPUT_BUFFER


@atexit.register
def write():
    sys.__stdout__.write(_OUTPUT_BUFFER.getvalue())


if __name__ == '__main__':
    test_cases = int(input())
    for cases in range(test_cases):
        arr = list(map(int, input().strip().split()))
        k = int(input())
        ob = Solution()
        res = ob.maxOfSubarrays(arr, k)
        for i in range(len(res)):
            print(res[i], end=" ")
        print()
        print("~")

#Which among them forms a perfect sudoku pattern
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        cols = [set() for i in range(9)]
        row = set()
        sb = [set() for i in range(3)]
        for i in range(9) :
            row = set()
            if i %3 == 0 : sb =  [set() for i in range(3)]
            for j in range(9) :
                if board[i][j] == '.' : continue
                if board[i][j] in row : return False
                row.add(board[i][j])
                if board[i][j] in cols[j] : return False
                cols[j].add(board[i][j])
                ind = int(j /3)
                if board[i][j] in sb[ind] : return False
                sb[ind].add(board[i][j])
        return True

#Rotten Oranges 
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        fresh = 0
        for i in grid :
            for j in i :
                if j == 1 : fresh += 1
        res = 0
        while fresh != 0 :
            res += 1
            pfresh = fresh
            # print(fresh)
            ngrid = [c[:] for c in grid]
            for i in range(len(grid)) :
                for j in range(len(grid[0])) :
                    if grid[i][j] != 2 : continue
                    if i -1 > -1 and ngrid[i-1][j] == 1 : 
                        ngrid[i-1][j] = 2
                        fresh -=1
                    if i +1 < len(grid) and ngrid[i+1][j] == 1 : 
                        ngrid[i+1][j] = 2
                        fresh -=1
                    if j -1 > -1 and ngrid[i][j-1] == 1 : 
                        ngrid[i][j-1] = 2
                        fresh -=1
                    if j+1 < len(grid[0]) and ngrid[i][j+1] == 1 : 
                        ngrid[i][j+1] = 2
                        fresh -=1
            grid = ngrid
            if fresh == pfresh : return -1
            
        return res

#Tree Burning
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

#Calculating max profit
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int :
        buy = [-math.inf]*(k+1)
        sell = [0]*(k+1)
        buy[0] = -prices[0]
        i = 1
        # print(buy)
        # print(sell)
        for p in prices :
            print(i, buy, sell)
            for j in range(min(k, i+1)-1, -1, -1) :
                sell[j] = max(sell[j], buy[j]+p)
            for j in range(min(k, i+1)-1, -1, -1) :
                buy[j] = max(buy[j], sell[j-1]-p)
            i += 1
        print(buy, sell)
        return max(sell)

#tree serialization and deserialization
class Codec:

    def serialize(self, root):
        # use level order traversal to match LeetCode's serialization format
        flat_bt = []
        queue = collections.deque([root])
        while queue:
            node = queue.pop()
            if node:
                flat_bt.append(str(node.val))
                queue.appendleft(node.left)
                queue.appendleft(node.right)
            else:
                # you can use any char to represent null
                # empty string means test for a non-null node is simply: flat_bt[i]
                flat_bt.append('')
        return ','.join(flat_bt)
    # time:  O(n)
    # space: O(n)

    def deserialize(self, data):
        if not data:
            return
        flat_bt = data.split(',')
        ans = TreeNode(flat_bt[0])
        queue = collections.deque([ans])
        i = 1
        while queue:
            node = queue.pop()
            if i < len(flat_bt) and flat_bt[i]:
                node.left = TreeNode(int(flat_bt[i]))
                queue.appendleft(node.left)
            i += 1
            if i < len(flat_bt) and flat_bt[i]:
                node.right = TreeNode(int(flat_bt[i]))
                queue.appendleft(node.right)
            i += 1
        return ans
    