# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections
from util import Counter

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        nextValues = util.Counter()
        for iteration in range(0, self.iterations):
            for state in self.mdp.getStates():
                values = []
                if self.mdp.isTerminal(state):
                    values.append(0)
                for action in self.mdp.getPossibleActions(state):
                    values.append(self.computeQValueFromValues(state, action))
                nextValues[state] = max(values)
            self.values = nextValues.copy()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        values = []
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for transition in transitions:
            reward = self.mdp.getReward(state, action, transition[0])
            values.append(transition[1] * (reward + self.discount * self.values[transition[0]]))

        return sum(values)
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)

        if not actions:
            return None

        QValue = float('-inf')
        bestAction = None
        for action in actions:
            value = self.computeQValueFromValues(state, action)
            if value > QValue:
                QValue = value
                bestAction = action

        return bestAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        nextValues = util.Counter()
        states = self.mdp.getStates()
        length = len(states)
        index = 0
        for iteration in range(0, self.iterations):
            values = []
            if self.mdp.isTerminal(states[index%length]):
                values.append(0)
            for action in self.mdp.getPossibleActions(states[index%length]):
                values.append(self.computeQValueFromValues(states[index%length], action))
            nextValues[states[index%length]] = max(values)
            index += 1
            self.values = nextValues.copy()

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        fringe = util.PriorityQueue()
        states = self.mdp.getStates()
        predecessors = {}
        for tState in states:
            previous = set()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for next, probability in transitions:
                        if probability != 0:
                            if tState == next:
                                previous.add(state)
            predecessors[tState] = previous
        for state in states:
            if self.mdp.isTerminal(state) == False:
                currentValue = self.values[state]
                values = []
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    value = self.computeQValueFromValues(state, action)
                    values.append(value)
                maxQvalue = max(values)
                diff = -abs(currentValue - maxQvalue)
                fringe.push(state, diff)

        for i in range(0, self.iterations):
            if fringe.isEmpty():
                break

            state = fringe.pop()
            if not self.mdp.isTerminal(state):
                values = []
                for action in self.mdp.getPossibleActions(state):
                    value = 0
                    for next, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        reward = self.mdp.getReward(state, action, next)
                        value = value + (prob * (reward + (self.discount * self.values[next])))
                    values.append(value)
                self.values[state] = max(values)
 
            for previous in predecessors[state]:
                currentValue = self.values[previous]
                values = []
                for action in self.mdp.getPossibleActions(previous):
                    values.append(self.computeQValueFromValues(previous, action))
                maxQValue = max(values)
                diff = abs(currentValue - maxQValue)
                if (diff > self.theta):
                    fringe.update(previous, -diff)

