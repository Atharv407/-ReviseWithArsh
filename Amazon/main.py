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

#Phone directory
#User function Template for python3
import bisect
class Solution:
    def displayContacts(self, n, contact, s):
        def bl(search, l, h) : 
            if l > h : return l
            while l < h : 
                m = (l+h)//2
                ms = contact[m][:len(search)]
                if ms < search : l = m + 1
                elif ms > search : h = m - 1
                else : h = m-1
            # print('bl - ', l, h)
            ms = contact[l][:len(search)]
            if ms == search : return l
            elif l+1 < len(contact) and contact[l+1][:len(search)] == search : return l+1
            # elif 
            return len(contact)
        def hl(search, l, h) : 
            if h < l : return h
            while l < h : 
                m = (l+h)//2
                ms = contact[m][:len(search)]
                if ms < search : l = m + 1
                elif ms > search : h = m - 1
                else : l = m+1
            # print('hl - ', l, h)
            ms = contact[h][:len(search)]
            if ms == search : return h
            elif h-1 > -1 and contact[h-1][:len(search)] == search : return h-1
            return -1
        contact = list(set(contact))
        contact.sort()
        # print(contact)
        search = ''
        res = []
        l, h = 0, len(contact)-1
        for a in s : 
            search += a
            l = bl(search, l, h)
            h = hl(search, l, h)
            # print(l,h)
            if h < l : res.append(['0'])
            else : res.append(contact[l: h+1])
            
        return res


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        n = int(input())
        contact = input().split()
        s = input()
        
        ob = Solution()
        ans = ob.displayContacts(n, contact, s)
        for i in range(len(s)):
            for val in ans[i]:
                print(val, end = " ")
            print()
        print("~")
# } Driver Code Ends

#Count ways to nth stair
class Solution:
    def waysToReachStair(self, k: int) -> int:
        def sm(c = 1, i = 0, back=True) :
            print(c,i)
            if c > k +1 : return 0
            res = sm(c+(2**i), i + 1, True)
            if back : res += sm(c-1, i, False)
            if c == k : res += 1
            return res
        def com(n, r) :
            den = 1
            num = 1
            # print(n, r)
            for i in range(2, n+1) :
                num*= i
                if i == r : 
                    den*= num
                if i == n-r :
                    den*= num
            # print(num, den)
            return num/den

        if k < 5 : return sm()
        for i in range(30) : 
            if k == 2**i : return 1
            if k < 2**i : 
                # print(i, k)
                k = 2**i - k
                # print(k)
                if k > i+1 : return 0
                else : return int(com(i+1, k))
        return 0

#Nuts and bolts problem
class Solution:
    def waysToReachStair(self, k: int) -> int:
        def sm(c = 1, i = 0, back=True) :
            print(c,i)
            if c > k +1 : return 0
            res = sm(c+(2**i), i + 1, True)
            if back : res += sm(c-1, i, False)
            if c == k : res += 1
            return res
        def com(n, r) :
            den = 1
            num = 1
            # print(n, r)
            for i in range(2, n+1) :
                num*= i
                if i == r : 
                    den*= num
                if i == n-r :
                    den*= num
            # print(num, den)
            return num/den

        if k < 5 : return sm()
        for i in range(30) : 
            if k == 2**i : return 1
            if k < 2**i : 
                # print(i, k)
                k = 2**i - k
                # print(k)
                if k > i+1 : return 0
                else : return int(com(i+1, k))
        return 0

#column name from col number
class Solution:
    def convertToTitle(self, n: int) -> str:
        m="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        res=""
        while n:
            n=n-1
            res=m[n%26]+res
            n=n//26
        return res    

# Brackets in matrix chain multiplication
class Solution:

    def __init__(self):
        self.st = ""
        self.name = 'A'

    # Function to print the parenthesis of matrix chain multiplication
    def printParenthesis(self, i, j, n, bracket):
        # If i and j are equal, it means only one matrix is remaining
        if i == j:
            self.st += self.name  # add the name of the matrix to the string
            self.name = chr(ord(self.name) +
                            1)  # increment the name for the next matrix
            return
        self.st += '('  # add '(' to the string
        self.printParenthesis(
            i, bracket[i][j], n, bracket
        )  # recursively print the parenthesis for the left side of the split
        self.printParenthesis(
            bracket[i][j] + 1, j, n, bracket
        )  # recursively print the parenthesis for the right side of the split
        self.st += ')'  # add ')' to the string

    # Function to calculate the minimum number of operations needed to multiply the matrices
    def matrixChainOrder(self, arr):
        n = len(arr)
        m = [[0] * n for _ in range(n)
             ]  # create a 2D array to store the minimum number of operations
        bracket = [
            [0] * n for _ in range(n)
        ]  # create a 2D array to store the split position for each matrix multiplication

        for i in range(1, n):
            m[i][i] = 0  # initialize the diagonal elements to 0

        for L in range(2, n):  # iterate over the lengths of the sequences
            for i in range(
                    1, n - L +
                    1):  # iterate over the starting indices of the sequences
                j = i + L - 1  # calculate the ending index of the sequence
                m[i][j] = float(
                    'inf'
                )  # set the minimum number of operations to a large value at first
                for k in range(i, j):  # iterate over possible split positions
                    q = m[i][k] + m[k + 1][j] + arr[i - 1] * arr[k] * arr[
                        j]  # calculate the number of operations needed for this split
                    if q < m[i][
                            j]:  # if this split has fewer operations than the current minimum, update the values
                        m[i][j] = q
                        bracket[i][j] = k

        self.printParenthesis(
            1, n - 1, n, bracket)  # call the function to print the parenthesis
        return self.st  # return the string containing the parenthesis