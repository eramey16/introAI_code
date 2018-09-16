# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
import copy
from game import Directions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

# Node class
# Holds a position and a list of steps to get there
class Node:
    
    def __init__(self, position):
        self.pos = position
        self.actions = []
        self.priority = 0
        self.cost = 0

    def move(self, successor, heuristic=None, problem=None):
        #print successor
        newpos = successor[0]
        direction = successor[1]
        newCost = self.cost+successor[2]
        f = newCost
        if heuristic:
            f += heuristic(newpos, problem)
            #print "Heuristic at "+str(newpos)+" "+str(heuristic(newpos, problem))
        newNode = copy.deepcopy(self)
        newNode.actions.append(direction)
        newNode.pos = newpos
        newNode.cost = newCost
        newNode.priority = f
        
        return newNode
    
    def __eq__(self, other):
        return self.pos==other.pos
    
    def __str__(self):
        return "Node at: "+str(self.pos)+", actions: "+str(self.actions)+", bcost: "+str(self.cost)+", priority: "+str(self.priority)
    def __hash__(self):
        return 1

# priority function for priority queue
# priority is kept track of in the nodes
def priorityFunction(node):
    return node.priority

# search function, used for all types of search
# fringe - a data structure for the fringe
# problem - the search problem
# heuristic - the heuristic being used
def search(fringe, problem, heuristic=None):
    # add the start state to the fringe
    startNode = Node(problem.getStartState())
    fringe.push(startNode)
    
    # keep track of visited states
    visited = set()
    
    # while nodes remain on the fringe
    while not fringe.isEmpty():
        # get highest priority leaf node
        leaf = fringe.pop()
        #print "\nLeaf - "+str(leaf)+"\n"
        # check if it is a repeat
        if leaf in visited:
            #print "Visited already"
            continue
        # add it to the visited set
        visited.add(leaf)
        # Goal check
        if problem.isGoalState(leaf.pos):
            #print "Goal state popped"
            return leaf.actions
        
        # get the next possible actions
        kids = problem.getSuccessors(leaf.pos)
        #print "Children: ", kids
        for kid in kids:
            # regiter the cost of each child state
            kidn = leaf.move(kid, heuristic, problem)
            # push the child node onto the fringe
            #print "Child node - ", kidn
            fringe.push(kidn)
    #print "\nreturning None\n"
    return []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()   
    return search(fringe, problem)
    
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    return search(fringe, problem)

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueueWithFunction(priorityFunction)
    return search(fringe, problem)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueueWithFunction(priorityFunction)
    
    return search(fringe, problem, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
