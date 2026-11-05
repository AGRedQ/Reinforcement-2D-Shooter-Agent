# Main Markdown for us to wrote idea
Note to AI Agents: If you are about to change any of the content underneath, ask for confirmation first.



### Dual System:
- Instead of training the shooting and the running into a same model. Why not split the two tasks into 2 models that co-exists?

### Agent Memory
- Preserve the enemy's last seen position and expose how long ago it was seen, without revealing its current hidden position.
- This should prevent the Agent from immediately forgetting an enemy after losing line of sight. Recurrent PPO with an LSTM or GRU policy could be a later alternative or extension.

### Cone-of-Vision Rays
- Replace the current all-or-nothing enemy offset with rays inside the Agent's cone of vision.
- Each ray should stop at walls and report wall and enemy distance separately, giving more local combat information without seeing through obstacles.

### Wall-Proximity Rays
- Add separate short-range 360-degree rays that detect only walls and arena boundaries.
- These are navigation sensors rather than vision, intended to help the Agent move around tight or future procedural maps.

### Introduce Self-play
- It might be a jump in difficulty, but it is, after all, the final destination of this project.

### Procedurally Generated Map
- I am thinking of a node-based procedural generated maps. We could generate 100~ map first for faster training, but evaluation or playing with human will generate a new map.
- There should be properties in procedurally generated maps, such as, there must be a direct contact with 2 agents.


### References 
- Reinforcement Learning Agent for a 2D Shooter Game (https://arxiv.org/pdf/2509.15042)
- Generalized Advantage Estimation, GAE (https://arxiv.org/abs/1506.02438)
- A.R.E.S: Self-Play Reinforcement Learning Shooter (only https://github.com/dorukarda424/ShooterGame)
