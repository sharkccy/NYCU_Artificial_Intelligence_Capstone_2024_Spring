import STcpClient
import numpy as np
import random

'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''
class GameEnvironment:
    def __init__(self):
        #self.client = None
        self.mapStat = None  # Initialize the current map state of the game
        self.sheepStat = None # Initialize the current map state of the game
        self.id_package = None 
        self.playerID = None
        self.done = None

    def reset(self):
        # Reset the game to its initial state
        # This function should communicate with the game server to start a new game
        # and initialize self.state with the initial game state
        # Get initial state

        self.id_package, self.playerID, self.mapStat = STcpClient.GetMap()

        init_pos = InitPos(self.mapStat, self.playerID)

        #要先找initPos才能得到 sheepStat
        STcpClient.SendInitPos(self.id_package, init_pos)
        # Get state after initial step

        return self.id_package, self.playerID, self.mapStat, self.sheepStat

    def step(self, action):
        # Send the chosen action to the game server
        STcpClient.SendStep(self.id_package, action)
        # Receive the new state, reward, and game over status

        return 
    def update(self):
        (self.done, self.id_package, self.mapStat, self.sheepStat) = STcpClient.GetBoard()
        return
    
class MCTSNode:
    global findBarrier_dir, all_move_dir, game_over
    findBarrier_dir = [(0, -1), (-1, 0), (0, 1), (1,0)]
    all_move_dir = [(-1, -1), (0,-1), (1, -1), (-1, 0), (1, 0), (-1,1), (0, 1), (1, 1)]
    def __init__(self, mapStat, sheepStat, parent=None, move=None):
        self.mapStat = mapStat
        self.sheepStat = sheepStat
        self.parent = parent
        self.move = move
        self.children = []
        self.score = 0  # 这里使用score来累计得分
        self.visits = 0

    def uct_score(self, total_visits, exploration_weight=1.44):
        if self.visits == 0:
            return float('inf') # 确保未访问的节点被优先探索
        average_score = self.score / self.visits  # 使用平均得分代替胜率
        uct = average_score + exploration_weight * (np.log(total_visits) / self.visits) ** 0.5
        return uct
    def uct_score_no_inf(self, total_visits, exploration_weight=1.44):
        if self.visits == 0:
            return 0 # 确保未访问的节点被优先探索
        average_score = self.score / self.visits  # 使用平均得分代替胜率
        uct = average_score + exploration_weight * (np.log(total_visits) / self.visits) ** 0.5
        return uct

    # 產生所有當前節點可以產生的合理行動
    def generate_legal_moves(self, playerID):
        moves = []
        for x in range(15):
            for y in range(15):
                if self.mapStat[x][y] == playerID and self.sheepStat[x][y] > 1:
                    for idx, (dx, dy) in enumerate(all_move_dir):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 15 and 0 <= ny < 15 and self.mapStat[nx][ny] == 0:
                            for split in range(1, int(self.sheepStat[x][y])):
                                moves.append(((x, y), split, idx + 1 if idx < 4 else idx + 2))  # Correct structure
                                
        return moves



    # 為該行動創造一個節點
    def simulate_move(self, move):
        x, y = move[0][0], move[0][1]
        dx, dy = all_move_dir[move[2] - 1 if move[2] <= 4 else move[2] - 2]
        nx, ny = x + dx, y + dy
        while 0 <= nx < 15 and 0 <= ny < 15 and self.mapStat[nx][ny] == 0:
            nx += dx
            ny += dy
        # Move back to the last valid position
        nx -= dx
        ny -= dy
        new_mapStat = self.mapStat.copy()
        new_sheepStat = self.sheepStat.copy()
        new_mapStat[nx][ny] = new_mapStat[x][y]
        new_sheepStat[nx][ny] += move[1]
        new_sheepStat[x][y] -= move[1]
        return new_mapStat, new_sheepStat

    


# DFS來計算模擬遊戲終局的分數
def explore_region(x, y, mapStat, visited, playerID):
    # 检查坐标是否越界或已访问或不属于当前玩家
    if not (0 <= x < 15 and 0 <= y < 15) or visited[x][y] or mapStat[x][y] != playerID:
        return 0
    visited[x][y] = True  # 标记为已访问
    size = 1  # 当前单元格也算入区域大小
    for dx, dy in findBarrier_dir:  # 上下左右四个方向
        nx, ny = x + dx, y + dy
        size += explore_region(nx, ny, mapStat, visited, playerID)
    return size

def calculate_score(mapStat, playerID):
    visited = [[False] * 15 for _ in range(15)]  # 访问状态矩阵
    total_score = 0
    for x in range(15):
        for y in range(15):
            if mapStat[x][y] == playerID and not visited[x][y]:
                region_size = explore_region(x, y, mapStat, visited, playerID)
                total_score += (region_size ** 1.25)
    return round(total_score)  # 求和后四舍五入

# 將所有possible_moves產生一個節點新增到樹上
def expand(node, playerID):
    possible_moves = node.generate_legal_moves(playerID)
    # print(f"Possible Moves: {possible_moves}")
    for move in possible_moves:
        # Assuming move is a tuple like (x, y, m, dir)
        new_mapStat, new_sheepStat = node.simulate_move(move)
        node.children.append(MCTSNode(new_mapStat, new_sheepStat, parent=node, move=move))


def select_best_child(node):
        # 首先，我们需要计算总访问次数，这是计算UCT分数所需的
                # 如果没有子节点，返回 None

        if len(node.children)==0:
            return None
        
        total_visits = sum(child.visits for child in node.children)
        

        
        # 选择具有最高UCT分数的子节点
        '''
        max_uct = 0
        best_child = node.children[0]
        for child in node.children:
            if max_uct < child.uct_score(total_visits):
                best_child = child
        '''
        best_child = max(node.children, key=lambda child: child.uct_score(total_visits))    
        return best_child

def select_best_step(node):
        # 首先，我们需要计算总访问次数，这是计算UCT分数所需的
                # 如果没有子节点，返回 None

        if len(node.children)==0:
            return None
        
        total_visits = sum(child.visits for child in node.children)
        
        
        # 选择具有最高UCT分数的子节点
        best_child = max(node.children, key=lambda child: child.uct_score_no_inf(total_visits))  
  
        return best_child


# 為該節點快速隨機選擇直到遊戲終局，並計算最終分數
def simulate_random_game(node, playerID):
    if(len(node.children)==0):
        return calculate_score(node.mapStat, playerID), node
    startNode = random.choice(node.children)
    simulate_tree_root = MCTSNode(startNode.mapStat, startNode.sheepStat, parent=node, move=startNode.move)
    while not is_game_over(simulate_tree_root.mapStat, simulate_tree_root.sheepStat, playerID):  # Include playerID
        #player myself
        legal_moves = simulate_tree_root.generate_legal_moves(playerID)
        if len(legal_moves)==0:
            continue  # No legal moves available, skip 
        move = random.choice(legal_moves)
        new_mapStat, new_sheepStat = simulate_tree_root.simulate_move(move)

        #player other
        for i in range(1,5):
            if(i == playerID):
                continue
            legal_moves = simulate_tree_root.generate_legal_moves(playerID)
            if len(legal_moves)==0:
                continue  # No legal moves available, skip 
            move = random.choice(legal_moves)
            new_mapStat, new_sheepStat = simulate_tree_root.simulate_move(move)

        #update state
        simulate_tree_root = MCTSNode(new_mapStat, new_sheepStat, parent=simulate_tree_root, move=move)

    return calculate_score(simulate_tree_root.mapStat, playerID), startNode


def backpropagate(node, result):
    # 从当前节点回溯到根节点，更新统计信息
    #print(f"from:{node.move}")
    while node.parent is not None:
        node.visits += 1
        node.score += result  # 更新节点的得分或胜利次数，这里假设使用了得分
        node = node.parent
    #print(f"end:{node.move}")
    return node

def mcts(root, playerID, iterations=1):
    # print(f"mapstat: {root.mapStat} sheepstat: {root.sheepStat}")
    now_node = root
    for iter in range(iterations): 
        # 选择 (試圖找出最佳解的過程節點，有利用和探索，)

        #print(iter+1)
        #print(f"now_node:{now_node.move}")
        while len(now_node.children):
            now_node = select_best_child(now_node)
            #print(f"select:{now_node.move}")

        # 扩展 

        expand(now_node, playerID)
        # 模拟
        result, now_node = simulate_random_game(now_node, playerID)
        # 回溯
        now_node = backpropagate(now_node, result)
        #print(f"real end:{now_node.move}")

'''
    產出指令
    
    input: 
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 15*15矩陣, 
              0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 15*15矩陣

    return Step
    Step : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
            4 X 6
            7 8 9
'''
def is_game_over(mapStat, sheepStat, playerID):

        # Check for each cell in the grid that belongs to the specified player
        for x in range(15):
            for y in range(15):
                if mapStat[x][y] == playerID and sheepStat[x][y] > 1:  # Check cells with player's sheep
                    # Check all possible directions for possible moves
                    for dx, dy in all_move_dir:
                        nx, ny = x + dx, y + dy
                        # Ensure the new position is within bounds and empty
                        if 0 <= nx < 15 and 0 <= ny < 15 and mapStat[nx][ny] == 0:
                            return False  # There is at least one legal move left for the player
        return True  # No legal moves left for the player, game is over for them

# 在GetStep函數中使用MCTS
def GetStep(mapStat, sheepStat, playerID):
    root = MCTSNode(mapStat, sheepStat)
    mcts(root, playerID, iterations=100)  # 執行100次MCTS迭代
    '''    
    #This pattern is for debug. Print whole mcts tree
    node = root
    cnt = 1

    while len(node.children):
        print(f"level:{cnt}")
        for i, child in enumerate(node.children):
            print(f"move:{child.move} score:{child.score} visit time:{child.visits} uct score:{child.uct_score(1000)}")
        node = select_best_child(node)
        print(f"select best child:{node.move}")
        cnt+=1
    '''
            
    
    # 從最佳子節點選擇行動
    best_child = select_best_step(root)
    #print(f"best_child is None:{best_child is None}")
    step = [(0, 0), 0, 1] if best_child.move is None else best_child.move
    #print(f"Generated Step: {step}")
    return step

def InitPos(mapStat, playerID):
    DOF_array = np.zeros((15, 15))
    DDOF_array = np.zeros((15, 15))
    init_pos = [0,0]
    max_DOF = 0
    max_CDOF = 0
    for i in range(15):
        for j in range(15):
            if mapStat[i][j] == 0:
                #initPos只能選board cell
                if any(0 <= i + dx <15 and 0 <= j + dy < 15 and mapStat[i + dx][j + dy] == -1 for dx, dy in findBarrier_dir):
                    DOF = 0
                    CDOF = 0
                    for dx, dy in all_move_dir:
                        nx, ny = i + dx, j + dy
                        while 0 <= nx < 15 and 0 <= ny < 15 and mapStat[nx][ny] == 0:
                            DOF += 1
                            if (dx, dy) in findBarrier_dir:
                                CDOF += 1
                            nx += dx
                            ny += dy
                    DOF_array[i][j] = DOF
                    DDOF_array[i][j] = CDOF
                    if DOF == max_DOF:
                        if CDOF > max_CDOF:
                            max_DOF = DOF
                            max_CDOF = CDOF
                            init_pos = [i, j]
                    elif DOF > max_DOF:
                        max_DOF = DOF
                        max_CDOF = CDOF
                        init_pos = [i, j]
    print(DOF_array)
    print("\n") 
    print(DDOF_array)
    return init_pos

# def calculate_DOF(mapStat, playerID):
    L1_DOF_array = np.zeros((15, 15))
    L1_CDOF_array = np.zeros((15, 15))
    distance_array = np.zeros((15, 15))
    enemy_choice = []
    for i in range(15):
        for j in range(15):
            if mapStat[i][j] == 0:
                L1_DOF = 0
                L1_CDOF = 0
                for dx, dy in all_move_dir:
                    nx, ny = i + dx, j + dy
                    while 0 <= nx < 15 and 0 <= ny < 15 and mapStat[nx][ny] == 0:
                        L1_DOF += 1
                        if (dx, dy) in findBarrier_dir:
                            L1_CDOF += 1
                        nx += dx
                        ny += dy
                L1_DOF_array[i][j] = L1_DOF
                L1_CDOF_array[i][j] = L1_CDOF

    L2_DOF_array = L1_DOF_array.copy()
    L2_CDOF_array = L1_CDOF_array.copy()
    max_L2_DOF = 0
    max_L2_CDOF = 0
    for i in range(15):
        for j in range(15):
            if mapStat[i][j] == 0 and L1_DOF_array[i][j] != 0:
                for dx, dy in all_move_dir:
                    nx, ny = i + dx, j + dy
                    while 0 <= nx < 15 and 0 <= ny < 15 and mapStat[nx][ny] == 0:
                        # print(f"i:{i} j: {j} nx:{nx} ny:{ny} L1_DOF_array[nx][ny]:{L1_DOF_array[nx][ny]}")
                        L2_DOF_array[i][j] += (L1_DOF_array[nx][ny])
                        if (dx, dy) in findBarrier_dir:
                            L2_CDOF_array[i][j] += (L1_CDOF_array[nx][ny])
                        nx += dx
                        ny += dy
                        # if any(0 <= i + dm <15 and 0 <= j + dn < 15 and mapStat[i + dm][j + dn] == -1 for dm, dn in findBarrier_dir):
                        #     if L2_DOF_array[i][j] == max_L2_DOF:
                        #         if L2_CDOF_array[i][j] > max_L2_CDOF:
                        #             max_L2_DOF = L2_DOF_array[i][j]
                        #             max_L2_CDOF = L2_DOF_array[i][j]
                        #             init_pos = [i, j]
                        #     elif L2_DOF_array[i][j] > max_L2_DOF:
                        #         max_L2_DOF = L2_DOF_array[i][j]
                        #         max_L2_CDOF = L2_CDOF_array[i][j]
                        #         init_pos = [i, j]
    
    for i in range(15):
        for j in range(15):
            if mapStat[i][j] != 0 and mapStat[i][j] != -1 and mapStat[i][j] != playerID:
                enemy_choice.append((i, j))
                
    Max_score = 0
    for i in range(15):
        for j in range(15):
            if L2_DOF_array[i][j] != 0:
                print(f"enemy_choice:{enemy_choice}")
                distance_array[i][j] = 5 * sum(np.sqrt((i - x) ** 2 + (j - y) ** 2) for x, y in enemy_choice)
                if any(0 <= i + dm <15 and 0 <= j + dn < 15 and mapStat[i + dm][j + dn] == -1 for dm, dn in findBarrier_dir):
                    if distance_array[i][j] + L2_DOF_array[i][j] > Max_score:
                        Max_score = distance_array[i][j] + L2_DOF_array[i][j]
                        init_pos = [i, j]

    print(np.round(distance_array).T)
    print("\n")
    # print(L1_DOF_array)
    print(L2_DOF_array.T)
    print("\n")
    print(np.round(distance_array + L2_DOF_array).T)
    print("\n")
    print(init_pos)
    return init_pos

# def InitPos(mapState, playerID):
    InitPos = [0, 0]
    InitPos = calculate_DOF(mapState, playerID)
    return InitPos
   
# Start the game loop
if __name__ == "__main__":
    env = GameEnvironment()
     # Initial connection and setup
    
    env.reset()

    # Game loop
    while True:

        env.update()
        if env.done:
            print("Game over or connection lost.")
            break
    
        # Get the next move from MCTS
        step = GetStep(env.mapStat, env.sheepStat, env.playerID)

        # Send the action to the server
        env.step(step)