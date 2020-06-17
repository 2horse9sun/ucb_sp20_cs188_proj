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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()


        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        print(bestIndices)
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
        foodList = newFood.asList()
        distances = []
        for food in foodList:
            distances.append(manhattanDistance(food,newPos))
        if len(distances) != 0:
            minFoodDistance = min(distances)
        else:
            minFoodDistance = -1
        foodScore = 1.0/minFoodDistance

        ghostDistances = 1
        warning = 0
        for i in range(0,len(newGhostStates)):
            if newScaredTimes[i] == 0:
                distance = manhattanDistance(newPos,newGhostStates[i].getPosition())
                ghostDistances += distance
                if distance <= 1:
                    warning += 1
            else:
                foodScore += 1
        ghostScore = -1.0/ghostDistances - warning
        
        return successorGameState.getScore() + foodScore + ghostScore

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        maxValue = float("-inf")
        action = "Stop"

        for state in gameState.getLegalActions(0):
            utility = self.helper(gameState.generateSuccessor(0, state),1,0)
            if utility > maxValue or maxValue == float("-inf"):
                maxValue = utility
                action = state
        return action
        
        util.raiseNotDefined()

    def helper(self,gameState,agentIndex,depth):

        
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:
            return self.maxValue(gameState,agentIndex,depth)
        else:
            if gameState.getNumAgents() == agentIndex:
                agentIndex = 0
                depth += 1
                return self.helper(gameState,agentIndex,depth)
            return self.minValue(gameState,agentIndex,depth)
        

    def maxValue(self,gameState,agentIndex,depth):
        v = float("-inf")
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex,action)
            v = max(v,self.helper(successor,agentIndex+1,depth))
        return v

    def minValue(self,gameState,agentIndex,depth):
        v = float("+inf")
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex,action)
            v = min(v,self.helper(successor,agentIndex+1,depth))
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxValue = float("-inf")
        action = "Stop"
        a = float("-inf")
        b = float("inf")

        for state in gameState.getLegalActions(0):
            utility = self.helper(gameState.generateSuccessor(0, state),1,0,a,b)
            if utility > maxValue or maxValue == float("-inf"):
                maxValue = utility
                action = state
            a = max(a,maxValue)
        return action

        util.raiseNotDefined()


    def helper(self,gameState,agentIndex,depth,a,b):

        
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:
            return self.maxValue(gameState,agentIndex,depth,a,b)
        else:
            if gameState.getNumAgents() == agentIndex:
                agentIndex = 0
                depth += 1
                return self.helper(gameState,agentIndex,depth,a,b)
            return self.minValue(gameState,agentIndex,depth,a,b)
        

    def maxValue(self,gameState,agentIndex,depth,a,b):
        v = float("-inf")
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex,action)
            v = max(v,self.helper(successor,agentIndex+1,depth,a,b))
            if v > b:
                return v
            a = max(a,v)
        return v

    def minValue(self,gameState,agentIndex,depth,a,b):
        v = float("+inf")
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex,action)
            v = min(v,self.helper(successor,agentIndex+1,depth,a,b))
            if v < a:
                return v
            b = min(b,v)
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
        maxValue = float("-inf")
        action = "Stop"

        for state in gameState.getLegalActions(0):
            utility = self.helper(gameState.generateSuccessor(0, state),1,0)
            if utility > maxValue or maxValue == float("-inf"):
                maxValue = utility
                action = state
        return action
        
        util.raiseNotDefined()

    def helper(self,gameState,agentIndex,depth):

        
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:
            return self.maxValue(gameState,agentIndex,depth)
        else:
            if gameState.getNumAgents() == agentIndex:
                agentIndex = 0
                depth += 1
                return self.helper(gameState,agentIndex,depth)
            return self.expectValue(gameState,agentIndex,depth)
        

    def maxValue(self,gameState,agentIndex,depth):
        v = float("-inf")
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex,action)
            v = max(v,self.helper(successor,agentIndex+1,depth))
        return v

    def expectValue(self,gameState,agentIndex,depth):
        sum = 0
        length = len(gameState.getLegalActions(agentIndex))
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex,action)
            sum += self.helper(successor,agentIndex+1,depth)
        return sum/length

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    evaluate = currentScore + foodScore + ghostScore + capsuleScore

    foodScore: Find the min distance of the food around.Assuming that eating each food increases the score by 10, the shorter the distance is, the lager the evaluation will be. Therefore, foodScore = 10.0/minFoodDistance.

    ghostScore: In order not to make Pacman stuck in one place, use 20.0/ghostDistances instead of 10.0. If distance <= 1, Pacman must get away from the ghost, thus warning must >= 10.0. Finally, ghostScore = -20.0/ghostDistances-warning serving as a penalty. If the ghost is scared right now, Pacman had better eat the ghost first because it will gain 200 score.

    capsuleScore: Because Pacman gains no score for eating a capsule, it's not right to let capsuleScore simply be 10.0/minDistance just like food. Instead, fewer capsules mean more chances to eat the ghost. Finally, capsuleScore = -num*30. Why 30? It must <= the ghostScore penalty in case both a ghost and a capsule are adjacent to Pacman.

    """
    "*** YOUR CODE HERE ***"

    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentCapsules = currentGameState.getCapsules()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

    foodList = currentFood.asList()
    distances = []
    for food in foodList:
        distances.append(manhattanDistance(food,currentPos))
    if len(distances) != 0:
        minFoodDistance = min(distances)
    else:
        minFoodDistance = -1
    foodScore = 10.0/minFoodDistance


    ghostDistances = 1
    warning = 0
    for i in range(0,len(currentGhostStates)):
        distance = manhattanDistance(currentPos,currentGhostStates[i].getPosition())
        if currentScaredTimes[i] == 0:
            ghostDistances += distance
            if distance <= 1:
                warning += 10
        else:
            foodScore += 200.0/distance
    ghostScore = -20.0/ghostDistances - warning

    capsuleScore = -len(currentGameState.getCapsules())*30
    
    return currentGameState.getScore() + foodScore + ghostScore + capsuleScore

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
