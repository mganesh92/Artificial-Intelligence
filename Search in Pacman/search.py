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
    start=problem.getStartState();
   # print start;
    stack=util.Stack();
    visited=[];
    stack.push((start,[]));
   # print "Added to stack: ", start;
    while not stack.isEmpty():
        currentNode,path=stack.pop();
       # print "Removed from stack ",currentNode;
       # print "path followng: ", path;
        #print currentNode,path
        #print currentNode,path,cost;
        if(problem.isGoalState(currentNode)):
            return path;
        if(currentNode not in visited):
            visited.append(currentNode);
           # print "visited ",visited;
            for nextNode, nextPath, cost in problem.getSuccessors(currentNode):
                #cost not needed for dfs...ignore for now
               # print "nextNode", nextNode;
                if (nextNode not in visited):
                    #print "path, nextpath",path,nextPath;
                    stack.push((nextNode,path+[nextPath]));
                  #  print "Added to Stack: ",nextNode;

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start=problem.getStartState();   
    stack=util.Queue();
    visited=[];
    stack.push((start,[]));   
    while not stack.isEmpty():
        currentNode,path=stack.pop();
        if(problem.isGoalState(currentNode)):
            return path;
        if(currentNode not in visited):
            visited.append(currentNode);           
            for nextNode, nextPath, cost in problem.getSuccessors(currentNode):                
                if (nextNode not in visited):                    
                    stack.push((nextNode,path+[nextPath]));                  

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    visited = [];
    queue = util.PriorityQueue();
    queue.push((start, [], 0), 0);
    while not queue.isEmpty():
        currentNode, path, cost = queue.pop();
        #print "cost popped out ", cost;
        if (problem.isGoalState(currentNode)):
            return path;
        if (currentNode not in visited):
           visited.append(currentNode);
           for nextNode, nextPath, nextCost in problem.getSuccessors(currentNode):
                if nextNode not in visited:
                   queue.push((nextNode, path + [nextPath], nextCost+cost), nextCost+cost);
                   #print "cost sent in q ",(nextCost+cost);         
    
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """Search the node that has the lowest combined cost and heuristic first."""
   
    from util import PriorityQueue
    fringe = PriorityQueue()  # Fringe to manage which states to expand
    ss=problem.getStartState()
    fringe.push(ss, 0)
    currentState = fringe.pop()
    allvisited = []  # List to check whether state has already been visited
    temp = []  # Temp variable to get intermediate paths
    finalpath = []  # List to store final sequence of directions
    presentpath = PriorityQueue()  # Queue to store direction to children (currState and pathToCurrent go hand in hand)
    while not problem.isGoalState(currentState):
        if currentState not in allvisited:
            allvisited.append(currentState)
            successors = problem.getSuccessors(currentState)
            for child, direction, cost in successors:
                temp = finalpath + [direction]
                costToGo = problem.getCostOfActions(temp) + heuristic(child, problem)
                if child not in allvisited:
                    fringe.push(child, costToGo)
                    presentpath.push(temp, costToGo)
        currentState = fringe.pop()
        finalpath = presentpath.pop()
    return finalpath
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
