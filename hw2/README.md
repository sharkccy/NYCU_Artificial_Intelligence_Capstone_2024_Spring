# NYCU Artificial Intelligence Capstone 2024 Spring

StudentID: 110612117  
Name: Chung-Yu Chang (張仲瑜), 110550128 蔡耀霆, 110652032 許元瑞

## Introduction
- In this project, we aim to design and implement an agent that can play a board game called Battle Sheep (see `battle_sheep_rules.png` for the rules).

- We implement an agent that uses the Monte Carlo Tree Search (MCTS) algorithm to make decisions during the game, and evaluate the initialization point with 
  special L1_DOF and L2_DOF for the game to improve the agent's performance.

- There are four modes for the game, including:
  - Mode 1: The basic form (4 players, 16 sheep per player, 64-cell playing field)  
  - Mode 2: Overcrowded bigger board (4 players, 32 sheep per player, 100-cell playing field within a 15x15 square)
  - Mode 3: Imperfect information: A player does not know the numbers of sheep in the cells occupied by the other players. 
  - Mode 4: Cooperative play: The four players form two pairs: Players 1 & 3, and Players 2 & 4. Each player is scored individually first, and then averaged
    with the score of its partner.

- See `team_8.pdf` for more details and analysis for the project.

## File Structure
- `AI_project2_2024_0407/team8_agent`: The main code for the project, including the implementation of the MCTS agent and the game environment.
- `AI_project2_2024_0407/L1DOF_and_L2_DOF.pdf`: The description of L1_DOF and L2_DOF for the Battle Sheep game.
- `AI_project2_2024_0407/team_8.pdf`: The project report, including the design and analysis of the MCTS agent.
- `battle_sheep_rules.png`: The rules of the Battle Sheep game.


