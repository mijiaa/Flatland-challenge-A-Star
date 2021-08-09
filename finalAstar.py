from flatland.envs.rail_env import RailEnv
from flatland.core.transition_map import GridTransitionMap
import itertools


class AgentNode:
    """
    node representing the current state of a train
    """
    def  __init__(self, agent_index, position,incoming_dir =None, action = None,parent=None):
        self.agent_index = agent_index
        self.parent = parent
        self.position = position
        self.incoming_dir = incoming_dir
        self.action = action

        self.g = 0 # cost
        self.h = 0 # cost to goal node
        self.f = 0 # total cost

    def __eq__(self, other):
        # return self.position == other.position
        return self.position == other.position and self.agent_index == other.agent_index


class StateNode:
    """
    graph node representing the current state of all the trains
    """
    def __init__(self, agent_nodes):
        agents = []
        agents_positions,agents_directions,agents_actions = [],[],[]
        agents_f, agents_h,agents_g= [],[],[]
        for node in agent_nodes:
            agents.append(node.agent_index)
            agents_positions.append(node.position)
            agents_directions.append(node.incoming_dir)
            agents_actions.append(node.action)
            agents_f.append(node.f)
            agents_h.append(node.h)
            agents_g.append(node.g)

        self.agent_index = agents.sort()
        self.position = agents_positions
        self.incoming_dir = agents_directions
        self.action = agents_actions
        self.agent_nodes = agent_nodes

        self.g = sum(agents_g)
        self.h = sum(agents_h)
        self.f = sum(agents_f)

    def __eq__(self, other ):
        return self.position == other.position


def get_possible_directions(env,current_node,incoming):
    dir_cor = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    valid_dir_bool = GridTransitionMap.get_transitions(env.rail,current_node.position[0], current_node.position[1],incoming)
    valid_direction,dir_index = [],[]

    for i in range(len(valid_dir_bool)):
        if valid_dir_bool[i] == 1:
            valid_direction.append(dir_cor[i])
            dir_index.append(i)
    return valid_direction,dir_index


def get_action(dir_index, outgoing):
    dir_cor = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # for k in range(len(dir_cor)):
    #     if (loc[0] + dir_cor[k][0]) == prev_loc[0] and (loc[1] + dir_cor[k][1]) == prev_loc[1]:
    #         outgoing = dir_cor[k]
    # print(dir_index, outgoing)
    if dir_index is 0:
        if outgoing == dir_cor[0] or outgoing == dir_cor[2]:
            return 2  # forward
        if outgoing == dir_cor[1]:
            return 3  # right
        if outgoing == dir_cor[3]:
            return 1 # left

    if dir_index is 1:
        if outgoing == dir_cor[1] or outgoing == dir_cor[3]:
            return 2
        if outgoing == dir_cor[2]:
            return 3
        if outgoing == dir_cor[0]:
            return 1

    if dir_index is 2:
        if outgoing == dir_cor[2] or outgoing == dir_cor[0]:
            return 2
        if outgoing == dir_cor[3]:
            return 3
        if outgoing == dir_cor[1]:
            return 1

    if dir_index is 3:
        if outgoing == dir_cor[3] or outgoing == dir_cor[1]:
            return 2
        if outgoing == dir_cor[0]:
            return 3
        if outgoing == dir_cor[2]:
            return 1


def return_state(current_state : StateNode):
    actions_taken = []
    for node in current_state.agent_nodes:
        curr = node
        actions = []
        while curr is not None:
            if curr.action is not None:
                actions.append(curr.action)
            else:
                actions.append(2)
            curr = curr.parent
        for i in range(node.agent_index):
            actions.insert(4,4)
        actions_taken.append((node.agent_index,actions[::-1]))

    # sort according to order of trains
    actions_taken.sort(key=lambda x:x[0])
    for i in range(len(actions_taken)):
        actions_taken[i] = actions_taken[i][1]
    max_len = max([len(i) for i in actions_taken])

    for i in range(len(actions_taken)):
        while len( actions_taken[i]) < max_len:
            actions_taken[i].append(2)
    return actions_taken


all_states = []


def a_star(env:RailEnv, starting_state : StateNode,goal_state,goals):
    open_state, closed_state= [], []
    open_state.append(starting_state)

    while len(open_state) > 0:
        current_state = open_state[0]
        current_index = 0
        # pick node with lowest f to expand
        for index, state in enumerate(open_state):

            if state.f < current_state.f:
                current_state = state
                current_index = index
            # Tie breaking
            elif state.f == current_state.f:
                if state.h < current_state.h:
                    current_state = state
                    current_index = index
                elif state.h == current_state.h:
                    if state.g < current_state.g:
                        current_state = state
                        current_index = index


        all_states.append(current_state)
        open_state.pop(current_index)
        closed_state.append(current_state)

        # if this node has collision, do not expand
        collision_flag = len(set(current_state.position)) == len(current_state.position)
        if not collision_flag:
            continue

        # if state is goal then backtrack
        if current_state == goal_state:
            return return_state(current_state)
        valid_children = []

        # neighbor function producing all possible next-state graph nodes from a given graph node
        for index,current_node in enumerate(current_state.agent_nodes):

            # after picking a node, find children
            children=[]

            # restrict possible directions using grid information
            possible_directions,direction_index = get_possible_directions(env,current_node,current_node.incoming_dir)

            # safe spots for train to stop to avoid collision by not moving
            new_node = AgentNode(current_node.agent_index, current_node.position, current_node.incoming_dir, 4,current_node)

            children.append(new_node)

            # create new nodes for next possible locations from current state
            for i, dir in enumerate(possible_directions):
                incoming = direction_index[i]
                new_position = (current_node.position[0] + dir[0], current_node.position[1] + dir[1])
                action = get_action(current_node.incoming_dir,dir)
                new_node =  AgentNode(current_node.agent_index,new_position,incoming,action,current_node)
                children.append(new_node)

            valid_child = []

            # calculate child node's g,h,f values
            for child in children:

                child.g = current_node.g + 1
                child.h = ((child.position[0] - goals[current_node.agent_index].position[0]) ** 2) + \
                          ((child.position[1] - goals[current_node.agent_index].position[1]) ** 2)
                child.f = child.g + child.h

                valid_child.append(child)

            valid_children.append(valid_child)

        # check and generate valid new state
        if len(valid_children) == 1:
            if StateNode(valid_children[0]) in open_state or StateNode(valid_children[0]) in closed_state:
                continue
            open_state.append(StateNode(valid_children[0]))
        else:
            combinations = list(itertools.product(*valid_children))
            for i in range(len(combinations)):
                if StateNode(combinations[i]) in open_state or StateNode(combinations[i]) in closed_state  :
                    continue
                open_state.append(StateNode(combinations[i]))
