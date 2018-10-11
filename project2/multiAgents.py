# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #xGhost = [newGhostStates[i].getPosition()[0] for i in range(len(newGhostStates))]
    #[yGhost = newGhostStates[i].getPosition()[1] for i in range(len(newGhostStates))]
    for gTuple in newGhostStates:
        d = manhattanDistance(gTuple.getPosition(), newPos)
        if d < 2:
            return -10
        if d < 3:
            return -20

    if len(currentGameState.getFood().asList()) > len(newFood.asList()):
        return 500
    minDist = float("infinity")
    for food in newFood.asList():

        md = manhattanDistance(food, newPos)
        if md < minDist:
            minDist = md
    return 1/minDist*100

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
  ## Value function, adapted from slides
  def value(self, state, d, agentIndex):
    #check if we are on pacman again
    if agentIndex == state.getNumAgents():
      #if so, increase depth and change index
      d += 1
      agentIndex = 0
    #if we are at depth now
    if d > self.depth:
      #return what we think the state is worth
      return self.evaluationFunction(state)
    #if it's about to return to the main function
    if agentIndex==0 and d==1:
      #return the direction to go, not the value  
      return self.minimax(state, d, agentIndex)[1]
    # otherwise return the value to the minimax agent
    return self.minimax(state, d, agentIndex)[0]
  
  #minimax agent minimizes or maximizes the value
  def minimax(self, state, d, agentIndex):
    #initialize list of actions and values
    actions = state.getLegalActions(agentIndex)
    if Directions.STOP in actions:
        actions.remove(Directions.STOP)
    values = []
    #what's the next agent's index?
    nextAgent = agentIndex+1
    #if there are no more actions to take
    if len(actions)==0:
        # return the value of the state
        return (self.evaluationFunction(state), Directions.STOP)
    #for each possible action
    for action in actions:
        # add the value of that action to values[]
        values.append(self.value(state.generateSuccessor(agentIndex, action), d, nextAgent))
    # if the agent is pacman, maximize the value
    if agentIndex==0:
        best = max(values)
    else:
        best = min(values) #otherwise, minimize the value
    #index of the best value, according to the agent
    i = values.index(best)
    #return the best action and its value
    return (values[i], actions[i])

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
    """
    return self.value(gameState, 1, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  ## Value function, adapted from slides
  def value(self, state, d, agentIndex, alpha, beta):
    #check if we are on pacman again
    if agentIndex == state.getNumAgents():
      #if so, increase depth and change index
      d += 1
      agentIndex = 0
    #if we are at depth now
    if d > self.depth:
      #return what we think the state is worth
      return self.evaluationFunction(state)
    #if it's about to return to the main function
    if agentIndex==0 and d==1:
      #return the direction to go, not the value  
      return self.minimax(state, d, agentIndex, alpha, beta)[1]
    # otherwise return the value to the minimax agent
    return self.minimax(state, d, agentIndex, alpha, beta)[0]
  
  #minimax agent minimizes or maximizes the value
  def minimax(self, state, d, agentIndex, alpha, beta):
    #initialize list of actions and values
    actions = state.getLegalActions(agentIndex)
    if Directions.STOP in actions:
        actions.remove(Directions.STOP)
    values = []
    #what's the next agent's index?
    nextAgent = agentIndex+1
    #if there are no more actions to take
    if len(actions)==0:
        # return the value of the state
        return (self.evaluationFunction(state), Directions.STOP)
    #for each possible action
    for action in actions:
        # add the value of that action to values[]
        v = self.value(state.generateSuccessor(agentIndex, action), d, nextAgent, alpha, beta)
        if agentIndex==0:
            if v >= beta:
                return (v, action)
            alpha = max(v, alpha)
        else:
            if v <= alpha:
                return (v, action)
            beta = min(v, beta)
        values.append(v)
    # if the agent is pacman, maximize the value
    if agentIndex==0:
        best = max(values)
    else:
        best = min(values) #otherwise, minimize the value
    #index of the best value, according to the agent
    i = values.index(best)
    #return the best action and its value
    return (values[i], actions[i])

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    alpha = 0
    beta = float('infinity')
    return self.value(gameState, 1, 0, alpha, beta)

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  ## Value function, adapted from slides
  def value(self, state, d, agentIndex):
    #check if we are on pacman again
    if agentIndex == state.getNumAgents():
      #if so, increase depth and change index
      d += 1
      agentIndex = 0
    #if we are at depth now
    if d > self.depth:
      #return what we think the state is worth
      return self.evaluationFunction(state)
    #if it's about to return to the main function
    if agentIndex==0 and d==1:
      #return the direction to go, not the value  
      return self.expectimax(state, d, agentIndex)[1]
    # otherwise return the value to the minimax agent
    return self.expectimax(state, d, agentIndex)[0]
  
  #minimax agent minimizes or maximizes the value
  def expectimax(self, state, d, agentIndex):
    #initialize list of actions and values
    actions = state.getLegalActions(agentIndex)
    if Directions.STOP in actions:
        actions.remove(Directions.STOP)
    values = []
    #what's the next agent's index?
    nextAgent = agentIndex+1
    #if there are no more actions to take
    if len(actions)==0:
        # return the value of the state
        return (self.evaluationFunction(state), Directions.STOP)
    #for each possible action
    for action in actions:
        # add the value of that action to values[]
        values.append(self.value(state.generateSuccessor(agentIndex, action), d, nextAgent))
    # if the agent is pacman, maximize the value
    if agentIndex==0:
        best = max(values)
        #index of the best value, according to the agent
        i = values.index(best)
        #return the best action and its value
        return (values[i], actions[i])
    else:
        exp = sum(values)/float(len(values)) #otherwise, take the expectation
        return (exp, Directions.STOP)

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    return self.value(gameState, 1, 0)

def closest(pos, others):
    if len(others)==0:
        return (0, True, True)
    d = float("infinity")
    loc = None
    for other in others:
        d2 = manhattanDistance(pos, other)
        if d2 < d:
            d = d2
            loc = other
    right = other[0] > pos[0]
    up = other[1] > pos[1]
    return (d, right, up)

def wallBetween(pos1, pos2, gameState):
    if manhattanDistance(pos1, pos2)<=1:
        return False
    x = (pos1[0]+pos2[0])/2.0
    y = (pos1[1]+pos2[1])/2.0
    
    if int(x)!=x:
        return False
    
    walls = gameState.getWalls().asList()
    if (x+1, y) in walls or (x-1, y) in walls or (x, y+1) in walls or (x, y-1) in walls:
        return True
    
    return False

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foodGrid = currentGameState.getFood()
    food = foodGrid.asList()
    pos = currentGameState.getPacmanPosition()
    ghosts = []
    gdists = []
    
    for i in range(1, currentGameState.getNumAgents()):
        g = currentGameState.getGhostPosition(i)
        ghosts.append(g)
        gdists.append(manhattanDistance(pos, g))
    
    #fterm = len(food)
    fterm = foodGrid.width*foodGrid.height-len(food)
    #gterm = closest(pos, ghosts)[0]
    gterm = 0
    gDistTerm = 0
    fdist, right, up = closest(pos, food)
    #fdist = breadthFirstSearch2(currentGameState)
    dterm = 1/float(fdist+1)
    
    #coefficients
    a = 10
    b = 1
    c = 10

    for i in range(len(gdists)):
        if gdists[i]<3 and not wallBetween(pos, ghosts[i], currentGameState):
            gterm-=30
        #else:
        #    gDistTerm+=0.1/d
    
    total = a*fterm + b*gterm + c*dterm #+gDistTerm

    if len(food)==0:
        total *= 10
    
    # break ties
    #if right:
    #    total+=2
    #if up:
    #    total+=1
    return total

def breadthFirstSearch(gameState):
    """
    copied then altered from project one
    """
    originalFoodCount = len(gameState.getFood().asList())
    fringe = util.Queue()
    tup = (gameState, 1)
    setPlease = set()  # set of positions
    fringe.push(tup)
    setPlease.add(gameState.getPacmanPosition())

    while not fringe.isEmpty():
      tuple = fringe.pop()
      gs = tuple[0]
      counter = tuple[1]
      if len(gs.getFood().asList()) < originalFoodCount:
        return counter
      for action in gs.getLegalPacmanActions():
          if action != Directions.STOP:
              gs2 = gs.generatePacmanSuccessor(action)
              tup = (gs2, counter + 1)
              if gs2.getPacmanPosition() not in setPlease:
                fringe.push(tup)
                setPlease.add(gs2.getPacmanPosition)
    #need retrace because should never be here

def breadthFirstSearch2(gameState):
    foodList = gameState.getFood().asList()
    foodNum = len(foodList)
    if foodNum == 0:
        return 0
    fringe = util.Queue()
    visited = set()
    start = (gameState, 0)
    fringe.push(start)
    
    while not fringe.isEmpty():
        gs, depth = fringe.pop()
        pos = gs.getPacmanPosition()
        if pos in visited:
            continue
        visited.add(pos)
        
        foodCount = len(gameState.getFood().asList())
        if foodCount < foodNum:
            return depth
        
        actions = gs.getLegalPacmanActions()
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for action in actions:
            nextState = (gs.generatePacmanSuccessor(action), depth+1)
            fringe.push(nextState)
            
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

