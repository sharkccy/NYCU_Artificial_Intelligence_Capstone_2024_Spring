import numpy as np
from scipy.ndimage import label
global findBarrier_dir, all_move_dir
findBarrier_dir = [(-1, 0), (0, -1), (0,1), (1,0)]
all_move_dir = [(-1, 0), (0, -1), (0,1), (1,0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

def calculate_player_score(mapStat, playerID):
        """
        Calculate the score for a given player based on the map state.
    
        Args:
        - mapStat: A 12x12 numpy array representing the map state.
        - playerID: The player's ID for whom to calculate the score.
    
        Returns:
            - The calculated score for the player.
        """
        # Create a binary map: 1 where the playerID matches, 0 elsewhere
        player_map = (mapStat == playerID).astype(int)
    
        # Find connected components (regions) in the binary map
        labeled_array, num_features = label(player_map)
        score = 0
        # Iterate through each connected region to calculate its contribution to the score
        for region_id in range(1, num_features + 1):
            region_size = np.sum(labeled_array == region_id)
            score += pow(region_size, 1.25)
    
        # Round the score according to the game rules
        return round(score)


def check_game_over(mapStat, sheepStat):
        rows, cols = mapStat.shape
        for row in range(rows):
            for col in range(cols):
                if mapStat[row][col] == 0 or mapStat[row][col] == -1:
                    continue
                else:
                    # for playerID in range(1, 5):
                    #     if mapStat[row][col] == playerID:
                            for dx, dy in all_move_dir:
                                nx, ny = row + dx, col + dy
                                if 0 <= nx < rows and 0 <= ny < cols and mapStat[nx][ny] == 0 and sheepStat[row][col] > 1:
                                    return False
        return True

mapStat = np.array([[-1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1,  1,  1,  1,  3, -1,  3, -1, -1],
                    [-1, -1, -1,  1,  1,  1,  1,  1,  3,  2,  2,  3],
                    [-1, -1,  1,  1,  1,  1, -1, -1,  3,  2,  2,  2],
                    [-1, -1,  4,  4,  4,  4,  2,  3,  2,  1,  2, -1],
                    [-1, -1, -1,  2,  4,  2,  3,  2,  3,  3,  3, -1],
                    [-1, -1, -1,  4,  4, -1, -1,  3,  3, -1, -1, -1],
                    [-1, -1, -1, -1,  4,  4,  0,  3,  3,  0, -1, -1],
                    [-1,  0,  4,  0,  4,  4, -1,  0,  0,  3, -1, -1],
                    [-1, -1, -1,  0,  4,  4, -1,  0, -1, -1, -1, -1],
                    [-1, -1, -1, -1,  4, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

sheepStat = np.array([[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                      [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 2, 3, 1, 1, 2, 1, 0],
                      [0, 0, 0, 2, 1, 2, 1, 2, 1, 1, 1, 0],
                      [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                      [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# print(calculate_player_score(mapStat, 1))
# print(check_game_over(mapStat, sheepStat))
print(16.0 > 7)    

# print(calculate_player_score(mapStat, 4))