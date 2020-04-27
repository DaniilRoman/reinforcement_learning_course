
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """    
    Qi = 0
    for next_state in mdp.get_next_states(state, action):
        prob   = mdp.get_transition_prob(state, action, next_state)
        reward = mdp.get_reward(state, action, next_state)
        Vi = state_values[next_state]
        
        Qi = Qi + (prob * (reward + gamma*Vi))
        
    return Qi
