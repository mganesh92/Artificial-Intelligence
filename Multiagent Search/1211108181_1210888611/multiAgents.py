# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        score = -(sys.maxint)
        currentFood = currentGameState.getFood()                      
        foodList=currentFood.asList()
        for food in foodList:
            distance = -(manhattanDistance(food, successorGameState.getPacmanPosition()))
            if (distance > score):
                score = distance              
        for state in newGhostStates:
            if state.getPosition()== successorGameState.getPacmanPosition():
            	if state.scaredTimer==0:
            		return -100
        if action=='Stop':
            return -sys.maxint
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        bestScore, bestMove = self.maxFunction(gameState, self.depth)
        return bestMove
    def maxFunction(self, gameState, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "noMove"

        themoves = gameState.getLegalActions()
        score = [self.minFunction(gameState.generateSuccessor(self.index, move), 1, depth) for move in themoves]
        best_score = max(score)
        bestindice = [index for index in range(len(score)) if score[index] == best_score]
        chosenIndex = bestindice[0]
        return best_score, themoves[chosenIndex]
        util.raiseNotDefined()

    def minFunction(self, gameState, agent, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "noMove"
        themoves = gameState.getLegalActions(agent)  # get legal actions.
        score = []
        if (agent != gameState.getNumAgents() - 1):
            score = [self.minFunction(gameState.generateSuccessor(agent, move), agent + 1, depth) for move in themoves]
        else:
            score = [self.maxFunction(gameState.generateSuccessor(agent, move), (depth - 1))[0] for move in themoves]
        minScore = min(score)
        worstIndices = [index for index in range(len(score)) if score[index] == minScore]
        chosenIndex = worstIndices[0]
        return minScore, themoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """	
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        global agent
        agent=gameState.getNumAgents()
        legalActions=gameState.getLegalActions(0)
        alpha=-sys.maxint
        beta=sys.maxint
        remaining=self.depth
        for action in legalActions:
        	currentState=gameState.generateSuccessor(0,action)
        	currentValue=self.value(currentState,remaining,1,alpha,beta)
        	if(currentValue>alpha):
        		next=action
        	alpha=max(alpha,currentValue)
        return next

    def value(self, currentState, remaining,agentIndex,alpha,beta):  
    	global agent
        if remaining == 0 or currentState.isWin() or currentState.isLose():
            return self.evaluationFunction(currentState)
        if currentState.isWin():
            return self.evaluationFunction(currentState)
        if currentState.isLose():
            return self.evaluationFunction(currentState)        
        if agentIndex == 0:
            return self.alphamax(currentState,remaining,agentIndex,alpha,beta)        
        return self.alphamin(currentState,remaining,agentIndex,alpha,beta)

    def alphamax(self, gameState, remaining,agentIndex, alpha, beta):        
        v = -10000000  
        global agent;
        legalActions = gameState.getLegalActions(agentIndex)
        if len(legalActions) == 0:
            return self.evaluationFunction(gameState)
        for action in legalActions:
            state = gameState.generateSuccessor(agentIndex, action)
            index=agentIndex+1
            v = max(v, self.value(state,remaining,index,alpha,beta))
            if v > beta:
                return v
            alpha = max(alpha,v)
        return v

    def alphamin(self,gameState, remaining,agentIndex,alpha,beta):        
        v = 1000000
        global agent
        legalActions = gameState.getLegalActions(agentIndex)
        temp=agentIndex+1
        if temp == agent:    
        #reset next        
            index = 0            
            remaining=remaining-1
        else: 
            index = agentIndex + 1
        for action in legalActions:
            state = gameState.generateSuccessor(agentIndex,action)
            v = min(v, self.value(state, remaining,index,alpha,beta))
            if v < alpha:
                return v
            beta = min(beta,v)
        return v	    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Maxima

        def max_Phase(state, depth):
            depth = depth + 1
            if state.isWin() or state.isLose() or depth == self.depth:  # Check the pacman's state
                return self.evaluationFunction(state)
            maximum = -999999.00
            for action in state.getLegalActions(0):
                maximum = max(maximum,
                              expPhase(state.generateSuccessor(0, action), depth, 1))  # the exp phse for ghosts
            return maximum

        # Exp

        def expPhase(state, depth, ghostRank):
            if state.isWin() or state.isLose():  # check whether pacman is alsive or dead
                return self.evaluationFunction(state)
            avg = 0
            for action in state.getLegalActions(ghostRank):
                if ghostRank == gameState.getNumAgents() - 1:
                    avg = avg + (max_Phase(state.generateSuccessor(ghostRank, action), depth)) / len(
                        state.getLegalActions(ghostRank))
                else:
                    avg = avg + (expPhase(state.generateSuccessor(ghostRank, action), depth,
                                                  ghostRank + 1)) / len(state.getLegalActions(ghostRank))
            return avg

        # expectimax  #


        possibleaction = gameState.getLegalActions(0)  # possible action of PACMAN
        maximum = -999999.00
        maxAction = ''
        for action in possibleaction:
            depth = 0
            temp = expPhase(gameState.generateSuccessor(0, action), depth,
                               1)
            if temp > maximum or (temp == maximum and random.random() > .3):
                maximum = temp
                maxAction = action
        return maxAction  
        

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"          
    # state = currentGameState
    pacmanPos = currentGameState.getPacmanPosition()    
    score = 0    
    foodList = currentGameState.getFood().asList()    
    distance =[manhattanDistance(i,pacmanPos) for i in foodList]
    if not distance:
        return sys.maxint
    score =score-len(foodList)*15 -sum(distance)    
    ghost_Distances = [manhattanDistance(i,pacmanPos) for i in currentGameState.getGhostPositions()]
    if not ghost_Distances:
    	return 1
    if ghost_Distances:
        if 0 in ghost_Distances:            
            return (-sys.maxint)                    
        score = score-sum(ghost_Distances)
    for ghostState in currentGameState.getGhostStates():    	
    	score = score + ghostState.scaredTimer  
    
    return score


# Abbreviation
better = betterEvaluationFunction

