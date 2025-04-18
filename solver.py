import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    print(len(temp))
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) # Lấy vị trí ban đầu của các thùng
    beginPlayer = PosOfPlayer(gameState) # Lấy vị trí ban đầu crua nhân vật

    startState = (beginPlayer, beginBox) # Khởi tạo vị trí bắt đầu của nhân vật và thùng
    frontier = collections.deque([[startState]]) # Khởi tạo hàng đợi frontier gồm 1 giá trị bàn đầu là startState
    exploredSet = set() # Khởi tạo tập hợp dùng để chức các trạng thái đã được xét
    actions = collections.deque([[0]]) # Khởi tạo hàng đợi chứa mỗi phần tử là một dãy hành động tạo thành đường đi của nhân vật
    temp = [] # List chứa dãy hành động từ lúc bắt đầu đến trạng thái kết thúc
    ### CODING FROM HERE ###
    while frontier: # Bắt đầu vòng lặp BFS đến khi frontier rỗng
        node = frontier.popleft() # Láy trạng thái đầu tiên từ hàng đợi frontier
        node_action = actions.popleft() # Lấy chuỗi hành động tương ứng với trạng thái node
        if isEndState(node[-1][-1]): # Kiểm tra xem trạng thái hiện tại có phải trạng thái kết thúc không
            temp += node_action[1:] # Nếu đúng, thêm các hành động từ node_action vào temp và kết thúc
            break
        if node[-1] not in exploredSet: # Kiểm tra xem trạng thái hiện tại đã được duyệt chưa
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]): # Duyệt qua tất cả các hành động hợp lệ mà người chơi có thể thực hiện từ trạng thái hiện tại
                # Cập nhật trạng thái mới dựa trên hành động hiện tại
                # newPosPlayer: vị trí mới của người chơi
                # newPosBox: vị trí mới của các hộp
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) 
                if isFailed(newPosBox): #Kiểm tra xem trạng thái mới có dẫn đến thất bại hay không
                    continue # Nếu có thì bỏ qua trạng thái này và tiếp tục vòng lặp
                frontier.append(node + [(newPosPlayer, newPosBox)]) # Nếu trạng thái mới hợp lệ, thêm nó vào frontier để duyệt sau
                actions.append(node_action + [action[-1]]) # Thêm chuỗi hành động mới vào actions
    print(len(temp)) # In độ dài của chuỗi hành động
    return temp
def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) # Lấy vị trí ban đầu crua thùng
    beginPlayer = PosOfPlayer(gameState) # Lấy vị trí ban đầu của nhân vật

    startState = (beginPlayer, beginBox) # startState khởi tạo với vị trí bắt đầu của nhân vật và thùng
    frontier = PriorityQueue() # Khởi tạo mảng frontier là một hàng đợi ưu tiên, với độ ưu tiên là chi phí đường đi từ tráng thái bắt đầu đến lúc trạng thái kết thúc
    frontier.push([startState], 0) # Thêm vào node startState vào frontier với chi phí là 0
    exploredSet = set() # Khởi tạo mảng exploredSet là một set chứa các trạng thái đã xét
    actions = PriorityQueue() # Khởi tạo action là một hàng đợi ưu tiên, dùng để lưu trữ đường đi tương ứng với node đó trong frontier
                              # Độ ưu tiên cũng chính là chi phí đường đi đó
    actions.push([0], 0) # thêm đường đi (chưa có hành động nào) vào hàng đợi actions với chi phí là 0
    temp = [] # Khởi tạo 1 list rỗng dùng để chứa đường đi từ trạng thái bắt đầu đến trạng thái kết thúc
    ### CODING FROM HERE ###
    while frontier: # Bắt đầu vòng lặp chính của thuật toán UCS cho đến khi frontier rỗng
        node = frontier.pop() #Lấy trạng thái có chi phí thấp nhất từ hàng đợi frontier
        node_action = actions.pop() # Lấy chuỗi hành động tương ứng với trạng thái node
        if isEndState(node[-1][-1]): #Kiểm tra xem trạng thái hiện tại có phải là trạng thái kết thúc không
            temp += node_action[1:] # Nếu đúng, thêm các hành động từ node_action vào temp và kết thúc vòng lặp
            break
        if node[-1] not in exploredSet: # Kiểm tra xem trạng thái hiện tại đã được duyệt chưa
            exploredSet.add(node[-1]) # Nếu chưa, thêm trạng thái này vào exploredSet để đánh dấu là đã duyệt
            for action in legalActions(node[-1][0], node[-1][1]): # Duyệt qua tất cả các hành động hợp lệ mà người chơi có thể thực hiện từ trạng thái hiện tại
                # Cập nhật trạng thái mới dựa trên hành động hiện tại
                # newPosPlayer: vị trí mới của người chơi
                # newPosBox: vị trí mới của các hộp
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox): # Kiểm tra xem trạng thái mới có dẫn đến thất bại hay không
                    continue
                # Nếu không thêm phần tử cuối cùng trong hàng đợi actions vào cuối node_actions ta được đường đi mới từ trạng thái bắt đầu đến trạng thái mới
                temp_node =  node_action + [action[-1]] 
                new_cost =  cost(temp_node[1:]) # Cập nhật chi phí đường đi mới ở trên
                frontier.push(node + [(newPosPlayer, newPosBox)], new_cost) # Nếu trạng thái mới hợp lệ, thêm nó vào frontier với chi phí được tính bằng hàm cost(node_action[1:]))
                actions.push(temp_node, new_cost) #Thêm 1 đường đi mới vào actions với độ ưu tiên là chi phí đường đi
    print(len(temp)) # In độ dài của chuỗi hành động
    return temp
"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    return result
