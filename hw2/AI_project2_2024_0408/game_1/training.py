import STcpClient 
import numpy as np
import random
from collections import deque
from tensorflow.keras import layers, models, optimizers
from scipy.ndimage import label
import subprocess
import pygetwindow as gw
import time
import psutil
import matplotlib.pyplot as plt

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



class GameEnvironment:
    global findBarrier_dir, all_move_dir, game_over
    findBarrier_dir = [(0, -1), (-1, 0), (0, 1), (1,0)]
    all_move_dir = [(-1, -1), (0,-1), (1, -1), (-1, 0),  (1, 0), (-1,1), (0, 1), (1, 1)]
    def __init__(self):
        #self.client = None
        self.mapStat = None  # Initialize the current map state of the game
        self.sheepStat = None # Initialize the current map state of the game
        self.id_package = None 
        self.playerID = None
        self.previous_score = None
        self.previous_mapStat = None
        self.previous_sheepStat = None
        self.done = None

    def reset(self):
        # Reset the game to its initial state
        # This function should communicate with the game server to start a new game
        # and initialize self.state with the initial game state

        # Get initial state

        self.id_package, self.playerID, self.mapStat = STcpClient.GetMap()
        self.game_over = 0
        max_move_count = 0
        init_pos = [0,0]

        for i in range(12):
            for j in range(12):
                if self.mapStat[i][j] == 0:
                    #initPos只能選board cell
                    if any(0 <= i + dx <12 and 0 <= j + dy < 12 and self.mapStat[i+dx][j + dy] == -1 for dx, dy in findBarrier_dir):
                        move_count = 0
                        for dx, dy in all_move_dir:
                            nx, ny = i + dx, j + dy
                            if 0 <= nx < 12 and 0 <= ny < 12 and self.mapStat[nx][ny] == 0:
                                move_count += 1
                        #找到最多可移動的cell
                        if move_count > max_move_count:
                            max_move_count = move_count
                            init_pos = [i, j]
        #要先找initPos才能得到 sheepStat
        STcpClient.SendInitPos(self.id_package, init_pos)
        # Get state after initial step
        (self.done, self.id_package, self.mapStat, self.sheepStat) = STcpClient.GetBoard() 
        #不確定會不會initail完就直接結束(決定done用不用的到)

        return self.id_package, self.playerID, self.mapStat, self.sheepStat

    def step(self, action):
        self.previous_mapStat = np.copy(self.mapStat)
        self.previous_sheepStat = np.copy(self.sheepStat)
        self.previous_score = self.calculate_player_score(self.playerID)
        # Send the chosen action to the game server
        STcpClient.SendStep(self.id_package, action)
        # Receive the new state, reward, and game over status
        (done, new_id_package, new_mapStat, new_sheepStat) = STcpClient.GetBoard() 
        self.id_package = new_id_package
        self.mapStat = new_mapStat
        self.sheepStat = new_sheepStat
        self.done = done
        #TODO Implement reward strategy here(?)
        #print(done)
        reward = self.calculate_reward(action)
        # print(f"Reward: {reward}")


        return done, new_id_package, new_mapStat, new_sheepStat, reward
    
    def check_valid_move(self, action):
        if self.previous_mapStat.all() == self.mapStat.all() and self.previous_sheepStat.all() == self.sheepStat.all():
            return False
        return True

    def check_game_over(self):

        rows, cols = self.mapStat.shape
        for row in range(rows):
            for col in range(cols):
                if self.mapStat[row][col] == 0 or self.mapStat[row][col] == -1:
                    continue
                else:
                    # for playerID in range(1, 5):
                    #     if mapStat[row][col] == playerID:
                            for dx, dy in all_move_dir:
                                nx, ny = row + dx, col + dy
                                if 0 <= nx < rows and 0 <= ny < cols and self.mapStat[nx][ny] == 0 and self.sheepStat[row][col] > 1:
                                    return False
        self.game_over = 1
        return True

    def calculate_player_score(self, playerID):
        """
        Calculate the score for a given player based on the map state.
    
        Args:
        - mapStat: A 12x12 numpy array representing the map state.
        - playerID: The player's ID for whom to calculate the score.
    
        Returns:
            - The calculated score for the player.
        """
        # Create a binary map: 1 where the playerID matches, 0 elsewhere
        player_map = (self.mapStat == playerID).astype(int)
    
        # Find connected components (regions) in the binary map
        labeled_array, num_features = label(player_map)
        score = 0
        # Iterate through each connected region to calculate its contribution to the score
        for region_id in range(1, num_features + 1):
            region_size = np.sum(labeled_array == region_id)
            score += pow(region_size, 1.25)
    
        # Round the score according to the game rules
        return round(score)
    
    def calculate_previous_score(self, playerID):
        player_map = (self.previous_mapStat == playerID).astype(int)
        labeled_array, num_features = label(player_map)
        score = 0
        for region_id in range(1, num_features + 1):
            region_size = np.sum(labeled_array == region_id)
            score += pow(region_size, 1.25)
        return round(score)
    
    def check_win(self):
        score = [0, 0, 0, 0]
        for playerID in range(1, 5):
            score[playerID - 1] = self.calculate_previous_score(playerID)
        if score[self.playerID - 1] == max(score):
            return True
        
    def get_surround(self, x, y):
        legal_moves = []
        for dx, dy in all_move_dir:
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x <= 11 and 0 <= new_y <= 11:
                legal_moves.append((new_x, new_y))
            # 檢查新位置是否在0到11之间
        

        return legal_moves
    
    def valid_moves(self, playerID, mapStat, sheepStat, x, y):
        '''
        Get a vaild moves direction of (x, y) for player i.
        '''

        results = []
        if mapStat[x, y] != playerID or sheepStat[x, y] <= 1:
            return []
        surround = self.get_surround(x, y)
        for value in surround:
            (x_iter, y_iter) = value
            print(f"ID:{playerID} x:{x_iter} y:{y_iter}")
            if mapStat[x_iter, y_iter] == 0:
                results.append(value)

        return results
    
    def is_skip(self, playerID, mapStat, sheepStat):
        '''
        Confirm if we want to skip this player's turn
        '''
        for x in range(12):
            for y in range(12):
                if self.sheepStat[x, y] > 1 and playerID == mapStat[x, y]:
                    if len(self.valid_moves(playerID, mapStat, sheepStat, x, y)):
                        # if there exist at least one valid move return false
                        return False
        return True
        
    def is_done(self, mapStat, sheepStat):
        '''
        Check if the game is over
        '''

        for id in range(1, 1):
            if not self.is_skip(id, mapStat, sheepStat):
                #print(f"player:{id} skip:{False}")
                return False
            
        return True



    def calculate_reward(self, action):
        #TODO Implement reward strategy here
        reward = 0

        # new_occupied_cells = np.sum(self.mapStat == playerID) - np.sum(self.previous_mapStat == playerID)
        # if new_occupied_cells > 0:
        #     reward += 2000 
        #else:
            #reward -= 500
        
        '''
        if self.done:
            if self.check_win():
                reward += 1000  # Assuming a large reward for winning
            else:
                reward -= 1000
        
        if self.check_valid_move(action):
            reward += 200
        else:
            reward -= 500
        '''
        x, y = action[0][0], action[0][1] 
        print(f"\n{self.previous_mapStat[x,y]}")
        print(f"playerID:{self.playerID} previous_mapStat:{self.previous_mapStat[x,y]}")
        if self.previous_mapStat[x,y] == self.playerID:
            reward += 2000
            print("reason 1")
            print(f"sheepStat:{self.previous_sheepStat[x,y]} action:{action[1]}")
            # print(f"score: {self.calculate_player_score(self.playerID)}")
            if self.previous_sheepStat[x,y] > action[1]:
                reward += 2000 
                print("reason 2")
                try:
                    print(x + all_move_dir[action[2] - 1 if action[2] <= 4 else action[2] - 2][0])
                    print(y + all_move_dir[action[2] - 1 if action[2] <= 4 else action[2] - 2][1])
                except Exception as e:
                    print(f"Error: {e}")
                print(self.previous_mapStat[x + all_move_dir[action[2] - 1 if action[2] <= 4 else action[2] - 2][0],y + all_move_dir[action[2] - 1 if action[2] <= 4 else action[2] - 2][1]])
                if self.previous_mapStat[x + all_move_dir[action[2] - 1 if action[2] <= 4 else action[2] - 2][0],y + all_move_dir[action[2] - 1 if action[2] <= 4 else action[2] - 2][1]] == 0:
                    reward += 2000
                    print("reason 3")
                    print(f"score: {self.calculate_player_score(self.playerID)}")
                    if self.calculate_player_score(self.playerID) - self.previous_score > pow(1, 1.25):
                        print("reason 4")
                        reward += 5000
        if self.done:
            if self.check_win():
                reward += 20000  
        return reward
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size*2, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, mapStat,sheepStat, action, reward, next_mapStat, next_sheepStat, done):
        self.memory.append((mapStat,sheepStat, action, reward, next_mapStat, next_sheepStat, done))
    
    def act(self, mapStat, sheepStat, all_possible_actions, dir_values):
        '''
        axis_selects = []
        print(sheepStat.shape)
        for x in range(12):
            for y in range(12):
                if sheepStat[x, y] > 1 and 4 == mapStat[x, y]:
                    axis_selects.append((x,y))
        '''
        combined_state = np.concatenate((mapStat.flatten(), sheepStat.flatten()))

        # 使用隨機 epsilon 策略


   
        if np.random.rand() <= self.epsilon:
            # 以一定機率隨機選擇一个动作
            #axis_selecct = random.choice(axis_selects)
            #action = [axis_selecct, random.randrange(1,16), random.choice(dir_values)]
            action_key = random.randrange(self.action_size)
            #if(all_possible_actions[action_key][0] not in axis_selects):
                #action_key = random.randrange(self.action_size)
            return action_key

        combined_state = combined_state.reshape(1, -1)
        act_values = self.model.predict(combined_state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for mapStat, sheepStat, action, reward, next_mapStat, next_sheepStat, done in minibatch:
            next_combined_state = np.concatenate((next_mapStat.flatten(), next_sheepStat.flatten()))
            next_combined_state = next_combined_state.reshape(1, -1)
            combined_state = np.concatenate((mapStat.flatten(), sheepStat.flatten()))
            combined_state = combined_state.reshape(1, -1)
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_combined_state)[0])
            target_f = self.model.predict(combined_state)
            target_f[0][action] = target
            self.model.fit(combined_state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    score_graph = []
    # Assuming you've implemented a way to determine state and action sizes
    state_size = 144 # 12*12棋盤資訊 需要考慮
    action_size = 12 * 12 * 15 * 8 #TODO 最大可動的羊群數量(8個2) * 最大可移動格數(12*4 - 4) * 最大可移動數量(1~15隻過去)

    process_names = ["blank.exe","Sample_2.exe","Sample_3.exe","Sample_4.exe"]
    env = GameEnvironment()
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load("C:\\講義與作業\\人工智慧總整與實作\\hw\\hw2\\AI_project2_2024_0408\\game_1\\model\\current.weights.h5")
        print("\n\n\nModel weights loaded successfully.\n\n\n")
    except Exception as e:
        print(f"\n\n\nFailed to load model weights: {e}\n\n\n")
    episodes = 10000
    board_size = 12
    m_range = range(1, 16) 
    dir_values = [1, 2, 3, 4, 6, 7, 8, 9]
    all_possible_actions = []
    for x in range(board_size):
        for y in range(board_size):
            for m in m_range:
                for dir in dir_values:  
                    # all_possible_actions.append([(5 , 5), 16, 2])
                    all_possible_actions.append([(x, y), m, dir])
    

    '''
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣, 
            0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    Step(action) : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
            4 X 6
            7 8 9
    '''
    for e in range(episodes):
        proc = subprocess.Popen(['AI_game.exe'], creationflags=subprocess.CREATE_NEW_CONSOLE)

        #while proc.poll() is not None:
            #print("Waiting game open...")
            #time.sleep(1)
        time.sleep(3)
        try:
            id_package, playerID, mapStat, sheepStat= env.reset()
            mapStat = np.reshape(mapStat, [1, state_size])
            sheepStat = np.reshape(sheepStat, [1, state_size])
            count = 0
        except:
            for sub_proc in psutil.process_iter(attrs=['name']):
                    
                    if sub_proc.info['name'] in process_names:
                        try:
                            sub_proc.kill()  
                            
                        except psutil.NoSuchProcess:
                            print(f"proccess {sub_proc.info['name']} 已不存在。")
   
            #381~391
            window_title = "C:\\講義與作業\\人工智慧總整與實作\\hw\\hw2\\AI_project2_2024_0408\\game_1\\AI_game.exe"
            windows = gw.getAllWindows()
            for window in windows:
                if window.title == window_title :
                    window.close()
                    break
 
            STcpClient._StopConnect()     
            proc.kill() 
            continue

        #Example, adjust by real state shape.

        while True:
            
            #(env.done, env.id_package, env.mapStat, env.sheepStat) = STcpClient.GetBoard() 
            

            action = agent.act(mapStat, sheepStat, all_possible_actions, dir_values) 
            try:
                done, id_package, next_mapStat, next_sheepStat, reward = env.step(all_possible_actions[action])    
                #done, id_package, next_mapStat, next_sheepStat, reward = env.step(([0,0],1,1)) #TODO No reward here

            
                next_mapStat = np.reshape(next_mapStat, [1, state_size])
                next_sheepStat = np.reshape(next_sheepStat, [1, state_size])

                agent.remember(mapStat, sheepStat, action, reward, next_mapStat,next_sheepStat, done)
                mapStat = next_mapStat
                sheepStat = next_sheepStat
                #print(f"step: {count}") 
                count+=1   
                print(f"count: {count} reward: {reward} ")

            except:
                print("I'm in except 426")
                print(f"Episode: {e+1}/{episodes} Score: {env.calculate_previous_score(playerID)}")
                score_graph.append(env.calculate_previous_score(playerID))
                agent.replay(count)

                 
                for sub_proc in psutil.process_iter(attrs=['name']):
                   
                    if sub_proc.info['name'] in process_names:
                        try:
                            sub_proc.kill()  
                            
                        except psutil.NoSuchProcess:
                            print(f"proccess {sub_proc.info['name']} 已不存在。")
   
                #419~428
                window_title = "C:\\講義與作業\\人工智慧總整與實作\\hw\\hw2\\AI_project2_2024_0408\\game_1\\AI_game.exe"
                windows = gw.getAllWindows()
                for window in windows:
                    if window.title == window_title :
                        window.close()
                        break
                time.sleep(1)  
                STcpClient._StopConnect()     
                proc.kill()
                break

            if (count > 32) :
                # print(f"Episode: {e+1}/{episodes}") 
                print(f"Episode: {e+1}/{episodes} Score: {env.calculate_player_score(playerID)}")
                score_graph.append(env.calculate_player_score(playerID))
                agent.replay(count)

                 
                for sub_proc in psutil.process_iter(attrs=['name']):
                    if sub_proc.info['name'] in process_names:
                        try:
                            sub_proc.kill() 
                            
                        except psutil.NoSuchProcess:
                            print(f"proccess {sub_proc.info['name']} 已不存在。")
   
                # 458~468
                window_title = "C:\\講義與作業\\人工智慧總整與實作\\hw\\hw2\\AI_project2_2024_0408\\game_1\\AI_game.exe"
                windows = gw.getAllWindows()
                for window in windows:
                    if window.title == window_title :
                        window.close()
                        break
                time.sleep(1)  
                STcpClient._StopConnect()     
                proc.kill()            
                break
            
        try:
            agent.save(f"C:\\講義與作業\\人工智慧總整與實作\\hw\\hw2\\AI_project2_2024_0408\\game_1\\model\\episode{e + 1}.weights.h5")
            print("\n\n\nModel weights save successfully.\n\n\n")
        except Exception as e:
            print(f"\n\n\nFailed to save model weights: {e}\n\n\n")

        time.sleep(2)

    plt.figure(figsize=(10, 5))
    plt.plot(score_graph, marker='o', linestyle='-', color='b')
    plt.title('Scores over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()