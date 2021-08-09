from finalAstar import a_star, AgentNode, StateNode
from flatland.envs.rail_env import RailEnv


def search(env: RailEnv):
    open_nodes, end_nodes, goal_state = [], [], []
    num_agents = 0
    for i in env.get_agent_handles():
        agent = env.agents[i]
        start = agent.initial_position
        end = agent.target
        direction = agent.direction
        start_node = AgentNode(i, start, direction)
        end_node = AgentNode(i, end, direction)
        open_nodes.append(start_node)
        end_nodes.append(end_node)
        num_agents += 1

    # create node for initial and goal state
    starting_state = StateNode(open_nodes)
    goal_state = StateNode(end_nodes)

    agents_actions = a_star(env, starting_state, goal_state,end_nodes)

    # add actions of each train to the schedule
    schedule = []
    for m in range(len(agents_actions[0])):
        _actions = {}
        for n in range(num_agents):
            _actions[n] = agents_actions[n][m]
        schedule.append(_actions)
    return schedule
