# -*- coding: utf-8 -*-
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()
    stack = util.Stack()  # 用以存储搜索树结点的栈
    stack.push(state)  # 以state为结点，即(x, y)坐标二元组
    actions = []  # 按序存储行动的方向，即东南西北
    visited = []  # 存储已访问结点
    parent = {state:((0, 0), 0)}     # 路径字典，以坐标为key，以(父节点(x,y), action)为value，start结点值如左
    while (not problem.isGoalState(state)):
        if stack.isEmpty():
            print "Can't find solution!"
            return []
        state = stack.pop()                     # 父结点退栈，并在下面将其后继全部入栈，但只选某一后继进行深度优先，这样回溯时可直接访问其兄弟，不必再访问父结点
        visited.append(state)
        for i in problem.getSuccessors(state):  # successor为((x,y),action,cost)三元组
            if i[0] not in visited:
                stack.push(i[0])
                parent[i[0]] = (state, i[1])    # 以当前坐标为key，保存父结点的坐标信息及从父结点到自身的action
                if problem.isGoalState(i[0]):
                    state = i[0]                # 将goal赋值给state，下次循环时结束
    while parent[state][1]:                     # 从goal开始追溯至起点，将行动插入actions
        actions.insert(0, parent[state][1])     # 从goal开始的action是逆序的，故每次都插入到最前面
        state = parent[state][0]

    # while(not problem.isGoalState(state)):
    #     if stack.isEmpty():
    #         print "Can't find solution!"
    #         return []
    #     has_successor = False                   # 标记是否有未曾访问的后继结点
    #     for i in problem.getSuccessors(state)[::-1]:  # successor为((x,y),action,cost)三元组
    #         if i[0] not in visited:
    #             stack.push(i[0])
    #             has_successor = True
    #             actions.append(i[1])
    #             break                           # 搜索到一个合法后继即中断循环，转向访问这个后继
    #     if not has_successor:
    #         stack.pop()                         # 没有合法后继，回溯，退栈并删除最后一步行动
    #         actions.pop()
    #     visited.append(state)
    #     state = stack.list[-1]                  # 获取栈顶元素（但不出栈）
    return actions
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()
    queue = util.Queue()  # 用以存储搜索树结点的队列
    queue.push(state)
    actions = []
    visited = []
    parent = {state:((0,0),0)}               # 路径字典，以坐标为key，以(父节点(x,y), action)为value，start结点值如左
    while (not problem.isGoalState(state)):
        if queue.isEmpty():
            print "Can't find solution!"
            return []
        state = queue.pop()
        visited.append(state)
        for i in problem.getSuccessors(state):
            if i[0] not in visited:
                queue.push(i[0])
                parent[i[0]] = (state, i[1])  # 以当前坐标为key，保存父结点的坐标信息及从父结点到自身的action
                if problem.isGoalState(i[0]):
                    state = i[0]                # 将goal赋值给state，下次循环时结束
    while parent[state][1]:                     # 从goal开始追溯至起点，将行动插入actions
        actions.insert(0, parent[state][1])     # 从goal开始的action是逆序的，故每次都插入到最前面
        state = parent[state][0]
    return actions
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = util.PriorityQueue()    #优先队列存储搜索序列
    item = [problem.getStartState(), problem.getStartState(), 0, None]  #item存储信息包括：当前节点，下一个扩展节点，当前节点到下一节点的耗散，采取的行动
    priority = 0    #优先级，耗散小的优先级高
    priorityQueue.push( item, priority )
    path = []       #路径
    closed = []     #已扩展的节点
    actions = []    #从起始状态至目标状态采取的行动
    while (not priorityQueue.isEmpty()):    #优先队列不为空
        state = priorityQueue.pop()         #当前节点是队列中优先级最高的
        #print state
        path.append([state[0], state[1], state[3]])     #扩展路径，路径信息包括：当前节点坐标，扩展节点坐标，采取的行动
        currentState = state[1]             #选择下一个节点作为当前节点，进行扩展
        if (currentState not in closed):    #如果这个节点未被扩展过，则加入closed表
            closed.append(currentState)
            if(problem.isGoalState(currentState)):
                break
            for i in problem.getSuccessors(currentState):       #对该节点的每个后继节点都存储下列信息
                cost = state[2] + i[2]      #从起始状态到该后继的耗散（起始状态到当前节点的耗散+扩展当前节点所需的耗散）
                action = i[1]               #到后继节点的行动
                nextState = i[0]            #后继节点的坐标
                priorityQueue.push([currentState, nextState, cost, action], cost)       #将当前节点扩展至其后继的信息加入优先队列
    currentState = state[1]     #结束循环时，当前节点是终止状态
    while (currentState != problem.getStartState()):    #从终止状态往回寻找路径，找到初始状态就结束
        for j in path:      #路径合法时
            if (j[1] == currentState):      #如果当前节点与合法路径信息中“扩展节点坐标”对应
                currentState = j[0]         #将该路径信息中“被扩展节点”即父节点坐标记为当前节点
                actions.insert(0, j[2])     #记录寻找路径的行动
                break
    return actions
    util.raiseNotDefined()
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    Use f(n) = cost + heuristic
    """
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()  # 当前状态
    queue = util.PriorityQueue()  # 待检测结点信息队列
    actions = []  # 行动序列
    openlist = {state: (0, heuristic(state, problem))}  # 待检测结点
    # key为结点，value为三元组（起点到结点实际耗散g，结点到终点预测耗散h），起点父节点为(-100,-100)
    queue.push(state, openlist[state][0] + openlist[state][1])  # 根据f=g+h越低，优先值越高的顺序加入队列
    closelist = []  # 已访问结点
    parent = {state: ((0, 0), 0)}  # 父节点信息((x,y),action)，其中action为从父节点到当前结点的操作，起点父节点信息设为((0,0),0)
    goal = 0    #用来存放目标状态
    isgoal = False  #是否是目标结点
    while (not isgoal):  # 如果没找到目标结点，则继续
        if openlist == {}:
            print "Can't find solution!"
            return []
        state = queue.pop()  # 选择fn最小的结点，提取出信息
        while openlist.get(state) is None:  # 如果不在待检测队列，说明已经检测过了，队列中存了非最优的结果，抛弃它
            state = queue.pop()
        cost = openlist[state][0]  # 从起点到当前位置的耗散
        openlist.pop(state)  # 从openlist删除该结点
        closelist.append(state)  # 将它加入closelist
        for i in problem.getSuccessors(state):
            if i[0] not in closelist:  # 选择未访问后继
                if not openlist.has_key(i[0]):  # 若该后继不在待检测列表中
                    if problem.isGoalState(i[0]):  # 并且若它是目标结点
                        openlist[i[0]] = (cost + i[2], 0)  # 将它加入待检测列表，并设置好相应的g、h
                        parent[i[0]] = (state, i[1])  # 设置它的父节点为当前状态state
                        goal = i[0]  # goal设置为目标结点
                        isgoal = True   # goal设置为找到目标结点以跳出循环
                    else:  # 如果不是目标结点，除了不设置goal以外，其余对应if中操作一样
                        openlist[i[0]] = (cost + i[2], heuristic(i[0], problem))
                        parent[i[0]] = (state, i[1])
                        queue.push(i[0], openlist[i[0]][0] + openlist[i[0]][1])
                else:  # 若该后继在待测列表中
                    if openlist[i[0]][0] > cost + i[2]:  # 如果当前路径比已存在的路径有更小的g，则更新当前路径的后继的相关信息
                        openlist[i[0]] = (cost + i[2], heuristic(i[0], problem))
                        parent[i[0]] = (state, i[1])
                        queue.push(i[0], openlist[i[0]][0] + openlist[i[0]][1])

    while parent[goal]!= ((0,0), 0):  # 从终点开始根据父节点找到前往起点的路
        actions.insert(0, parent[goal][1])  # 因为是从终点开始，所以将每次寻找到的位置信息应该放在actions的最前面
        goal = parent[goal][0]  # 选择下一个父节点
    return actions
    util.raiseNotDefined()

def greedySearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest heuristic=h(n)，witch is the predict distance
    between current state and goal.
    Use f(n)=heuristic
    """
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()     #当前状态
    stack = util.Stack()    #保存已访问但仍可能有未访问后继结点的结点
    stack.push(state)
    actions = []    #行动序列
    visited = []    #已访问结点
    nextOk = False

    while(not problem.isGoalState(state)):
        excost = util.PriorityQueue()  # k三元组(state,acition,cost)，按照action对应的后继结点与目标结点的启发式值的优先顺序
        if stack.isEmpty():
            print "Can't find solution!"
            return []
        for i in problem.getSuccessors(state):
            if i[0] not in visited:
                if type(state[0]) == type((0,0)):
                    if state[1][i[0][0][0]][i[0][0][1]]:
                        stack.push(i[0])  # 将选中的后继结点入栈
                        actions.append(i[1])  # 添加对应的行动
                        nextOk = True
                        break
                excost.push(i,heuristic(state,problem))
                #excost[i] = heuristic(state,problem) #计算未访问过的后继结点与目标结点的启发式值

        if not nextOk:
            if not excost.isEmpty():  # state有后继结点就找h(n)最小的那个
                gnext = excost.pop()  # 选择启发式值最小的未访问后继结点对应的action
                stack.push(gnext[0])  # 将选中的后继结点入栈
                actions.append(gnext[1])  # 添加对应的行动
            else:
                stack.pop()  # 若无后继结点可访问，当前状态出栈
                actions.pop()  # 对应出栈结点的行动也要从行动队列中删除
        nextOk = False
        visited.append(state)   #当前结点标记为已访问
        state = stack.list[-1]  #取栈顶元素未新的当前结点
    return actions
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
greedy = greedySearch
