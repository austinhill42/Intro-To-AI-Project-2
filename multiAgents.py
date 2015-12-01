# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import layout

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
    """
    import math
    from collections import namedtuple

    Node = namedtuple("Node", "agent, action, child")

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          IMPORTANT: Your code must also print the value of the action that
            getAction returns (i.e., the value of the minimax decision)
        """
        "*** YOUR CODE HERE ***"

        minimax = self.minimax(0, self.depth, gameState)

        print "Action: ", minimax[1]
        print "Minimax: ", minimax[0]

        return minimax[1]

    def minimax(self, agent, depth, state):

        if depth == 0:
            ghostPos = state.getGhostPosition(agent)
            ghostDir = state.getGhostState(agent).getDirection()
            pacPos = state.getPacmanPosition()
            pacDir = state.getPacmanState().getDirection()
            food = self.nearestFood(state)
            foodDir =  self.relativeDir(food, pacPos)

            #if ghostDir == pacDir:


             #   if foodDir != ghostDir:
            #        return [manhattanDistance(food, pacPos), foodDir]
             #   else:
             #       return [manhattanDistance(ghostPos, pacPos), ghostDir]
            #else:
            return [manhattanDistance(ghostPos, pacPos), foodDir]

        if agent == 0:
            best = [float('-inf'), 'Stop']

            for action in state.getLegalActions(agent):
                for ghostAgent in range(1, state.getNumAgents()):
                    val = self.minimax(ghostAgent, depth - 1, state.generateSuccessor(agent, action))
                    best = [max(best[0], val[0]), action]

            return best

        else:
            best = [float('inf'), 'Stop']

            for action in state.getLegalActions(agent):
                val = self.minimax(0, depth, state.generateSuccessor(agent, action))
                best = [min(best[0], val[0]), action]

            return best

    def heuristic(self, agent):
        state = self.gamestate

        pacmanPosition = state.getPacmanPosition()
        ghostDistance = manhattanDistance(state.getPacmanPosition(), state.getGhostPosition(agent))
        food = state.getFood()
        nearestFood = (float('inf'), float('inf'))
        foodDir = []

        for x in range(len(food)):
            for y in range(len(food[x])):
                if food[x][y] == True:
                    if manhattanDistance((x, y), pacmanPosition) < manhattanDistance(pacmanPosition, nearestFood):
                        nearestFood = (x, y)

        foodDir = self.relativeDir(nearestFood, pacmanPosition)

    def nearestFood(self, state):
        pacmanPosition = state.getPacmanPosition()
        food = state.getFood()
        nearestFood = (float('inf'), float('inf'))

        for x in range(food.width):
            for y in range(food.height):
                if food[x][y] == True:
                    if manhattanDistance((x, y), pacmanPosition) < manhattanDistance(pacmanPosition, nearestFood):
                        nearestFood = (x, y)

        return nearestFood

    def relativeDir(self, xy1, xy2):
        """
            Returns the direction of xy1 relative to xy2
        """

        dir = []

        if xy1[0] < xy2[0]:
            dir.append('West')
        elif xy1[0] > xy2[0]:
            dir.append('East')
        if xy1[1] < xy2[1]:
            dir.append('South')
        elif xy1[1] > xy2[1]:
            dir.append('North')

        return dir


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction

          IMPORTANT: Your code must also print the value of the action that
            getAction returns (i.e., the value of the minimax decision)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.

          IMPORTANT: Your code must also print the value of the action that
            getAction returns (i.e., the value of the expectimax decision)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
