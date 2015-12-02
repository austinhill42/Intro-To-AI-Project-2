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

        agent = 0  # Start with Pacman
        actions = gameState.getLegalActions(agent)
        actions.remove(Directions.STOP)
        minimaxactions = {}

        for action in actions:
            mm = self.minimax(agent, self.depth, gameState.generateSuccessor(agent, action))
            minimaxactions[mm] = action  # Use a dictionary to associate the action with the minimax value

        action = minimaxactions[max(minimaxactions)]  # Do final max layer to get action

        print "Action: ", action
        print "Minimax: ", max(minimaxactions)

        return action

    def minimax(self, agent, depth, state):

        if agent == state.getNumAgents():  # Start again with Pacman when ghosts are done
            agent = 0

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)  # Recursive base case, return score

        if agent == 0:
            ret = float('-inf')  # Low number for initial max
            actions = state.getLegalActions(agent)
            actions.remove(Directions.STOP)  # Increase achievable depth

            for action in actions:  # For each legal action recursively call minimax and get max of the result
                ret = max(ret, self.minimax(agent + 1, depth - 1, state.generateSuccessor(agent, action)))

        else:
            ret = float('inf')  # High number for initial min
            actions = state.getLegalActions(agent)
            # Ghosts already can't stop
            for action in actions:  # For each legal action recursively call minimax and get min of the result
                ret = min(ret, self.minimax(agent + 1, depth, state.generateSuccessor(agent, action)))

        return ret


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """
    from collections import namedtuple

    abval = namedtuple("abval", "value, action")

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction

          IMPORTANT: Your code must also print the value of the action that
            getAction returns (i.e., the value of the minimax decision)
        """
        "*** YOUR CODE HERE ***"

        alpha = float('-inf')
        beta = float('inf')
        ab = self.alphabeta(0, self.depth, gameState, alpha, beta)

        print "Action: ", ab.action
        print "AlphaBeta: ", ab.value

        return ab.action

    def alphabeta(self, agent, depth, state, alpha, beta):

        if agent == state.getNumAgents():  # Start again with Pacman when ghosts are done
            agent = 0

        if depth == 0 or state.isWin() or state.isLose():  # Recursive base case
            if agent == 0:  # Get action for current state
                action = state.getPacmanState().getDirection()
            else:
                action = state.getGhostState(agent).getDirection()

            return self.abval(self.evaluationFunction(state), action)

        if agent == 0:
            ret = self.abval(float('-inf'), Directions.STOP)  # Low number for initial max, stop is placeholder
            actions = state.getLegalActions(agent)
            actions.remove(Directions.STOP)  # Increase achievable depth

            for action in actions:
                ab = self.alphabeta(agent + 1, depth - 1, state.generateSuccessor(agent, action), alpha, beta)
                ab = self.abval(ab.value, action)

                if max(ret.value, ab.value) == ab.value:  # Max the alphabeta value
                    ret = ab

                if ret.value >= beta:  # Skip branches worse than the current one
                    return ret

                alpha = max(alpha, ret.value)  # Set alpha

            return ret

        else:
            ret = self.abval(float('inf'), Directions.STOP)  # Low number for initial min, stop is placeholder
            actions = state.getLegalActions(agent)
            # Ghosts already can't stop
            for action in actions:
                ab = self.alphabeta(agent + 1, depth, state.generateSuccessor(agent, action), alpha, beta)
                ab = self.abval(ab.value, action)

                if min(ret.value, ab.value) == ab.value:  # Min the alphabeta value
                    ret = ab

                if ret.value <= alpha:  # Skip branches better than the current one
                    return ret

                beta = max(beta, ret.value)  # Set beta

            return ret


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
