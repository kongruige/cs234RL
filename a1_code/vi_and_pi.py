### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim
import copy 

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = 0.
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################

    backup_val = R[state, action] + gamma * np.dot(T[state, action, :], V)

    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################

    # 1. Loop until the value function converges
    while True:
        # Keep track of the maximum change in the value function on this pass
        delta = 0
        
        # 2. Loop over all states to update their values
        for state in range(num_states):
            # Store the old value of the state to check for convergence
            v_old = value_function[state]
            
            # 3. Get the SINGLE action prescribed by the policy
            action = policy[state]
            
            # 4. Compute the new value using the Bellman backup for that state-action pair
            #    This is the Bellman *expectation* equation, not the optimality equation
            value_function[state] = bellman_backup(state, action, R, T, gamma, value_function)
            
            # Update the maximum change seen so far
            delta = max(delta, abs(v_old - value_function[state]))
            
        # 5. If the value function has stabilized, exit the loop
        if delta < tol:
            break

    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #

    # loop-based implementation
    # for state in range(num_states):
    #     best_action = 0
    #     best_q = float('-inf')
    #     for action in range(num_actions):
    #         q = bellman_backup(state, action, R, T, gamma, V_policy)

    #         if q > best_q:
    #             best_q = q
    #             best_action = action
        
    #     new_policy[state] = best_action

    # vectorized implementation
    # 1. Calculate Q-values for ALL state-action pairs at once.
    expected_future_values = T @ V_policy # Shape: (num_states, num_actions)
    q_values = R + gamma * expected_future_values

    # 2. Find the best action for each state using np.argmax.
    new_policy = np.argmax(q_values, axis=1)

    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #

    while True:
        # 1. Evaluate the current policy
        V_policy = policy_evaluation(policy, R, T, gamma, tol)
        
        # 2. Store a copy of the old policy to check for convergence later
        old_policy = policy.copy()
        
        # 3. Improve the policy by acting greedily with respect to V_policy
        policy = policy_improvement(policy, R, T, V_policy, gamma)

        # 4. Check if the policy has stopped changing
        # We can use np.array_equal here because the policies are INTEGER VALUED and therefore no precision errors
        if np.array_equal(old_policy, policy):
            break

    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        # Keep a copy of the old value function to check for convergence
        old_value_function = value_function.copy()
        
        # 1. Calculate Q-values for all state-action pairs at once
        #    This is the corrected Bellman optimality update
        expected_future_values = T @ value_function # Should be value_function, NOT policy
        q_values = R + gamma * expected_future_values

        # 2. Update the value function by taking the max Q-value for each state
        value_function = np.max(q_values, axis=1)

        # 3. Check for convergence using the tolerance 'tol'
        #    Stop if the largest change in the value function is less than tol
        if np.max(np.abs(value_function - old_value_function)) < tol:
            break
    
    # 5 Calculate Q-values for ALL state-action pairs at once.
    expected_future_values = T @ value_function # Shape: (num_states, num_actions)
    q_values = R + gamma * expected_future_values

    # 6. Find the best action for each state using np.argmax.
    policy = np.argmax(q_values, axis=1)

    ############################
    return value_function, policy


if __name__ == "__main__":
    SEED = 1234
    
    # You will need to run this whole script three times, 
    # changing this value each time.
    RIVER_CURRENT = 'WEAK' # Change to 'MEDIUM', then 'STRONG'
    
    print(f"########### Analyzing RiverSwim with {RIVER_CURRENT} current ###########")

    env = RiverSwim(RIVER_CURRENT, SEED)
    R, T = env.get_model()

    # Start with a high discount factor and decrease it
    discount_factor = 0.99
    
    while discount_factor >= 0:
        # Run value iteration to find the optimal policy for the current gamma
        # Option 1: Value Iteration
        # V_opt, policy_opt = value_iteration(R, T, gamma=discount_factor, tol=1e-3)

        # Option 2: Policy Iteration
        V_opt, policy_opt = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)

        # Get the action at the starting state (state 0)
        action_at_start = policy_opt[0]
        
        print(f"gamma = {discount_factor:.2f} -> Action at start: {['L', 'R'][action_at_start]}")

        # Check if the agent chooses to go LEFT (action 0)
        if action_at_start == 0:
            print("\n" + "="*50)
            print(f"TIPPING POINT FOUND for {RIVER_CURRENT} current!")
            print(f"The largest discount factor for which the agent goes LEFT is {discount_factor:.2f}")
            print(f"The optimal value function at this point is: \n{V_opt}")
            print(f"The optimal policy at this point is: \n{[['L', 'R'][a] for a in policy_opt]}")
            print("="*50 + "\n")
            break # We found our answer, so exit the loop

        # If the action is still RIGHT, decrease gamma and try again
        discount_factor -= 0.01

    # This will be printed if the loop finishes without finding a LEFT action
    if discount_factor < 0:
        print("No tipping point found. The agent always goes RIGHT.")

