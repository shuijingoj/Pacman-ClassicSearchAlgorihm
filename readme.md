# 基于Pacman(吃豆人)的经典搜索算法实现

### 本项目为斯坦福大学CS221课程作业的一部分，项目详细可见：http://stanford.edu/~cpiech/cs221/homework/prog/pacman/pacman.html

### 本项目仅修改/补充了search.py及searchAgents.py部分代码以实现智能体搜索算法，其余代码均为官方源码。修改部分包括:

- search.py 
  
  1. depthFirstSearch(problem) 单目标深度优先搜索算法；
  2. breadthFirstSearch(problem) 单目标广度优先搜索算法；
  3. uniformCostSearch(problem)  单目标代价一致搜索算法；
  4. aStarSearch(problem, heuristic=nullHeuristic) A*搜索算法；
  5. greedySearch(problem, heuristic=nullHeuristic) 贪婪搜索算法。


- searchAgent.py

  1. foodHeuristic(state, problem) A*搜索算法的启发函数。
  2. greedyfoodHeuristic(state, problem) 贪婪搜索算法的启发函数。
