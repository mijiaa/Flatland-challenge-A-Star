'''
@author: Frits de Nijs
'''

from math import floor

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_manual_specifications_generator, complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator

def two_agent_collision_map(): 

    # Example generate a rail given a manual specification,
    # a map of tuples (cell_type, rotation)
    specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (7, 0), (0, 0)],
             [(7, 270), (1, 90), (8, 90), (8, 0), (8, 180), (0, 0)],
             [(7, 270), (1, 90), (10, 270), (2, 90), (7, 90), (0, 0)],
             [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]
    
    env = RailEnv(width=6, height=4, rail_generator=rail_from_manual_specifications_generator(specs), number_of_agents=2)
    
    # Position the agents at their initial positions. 
    env.reset()
    env.agents[0].initial_position = (2,4)
    env.agents[0].initial_direction = 1
    env.agents[0].target = (1,0)
    env.agents[1].initial_position = (2,1)
    env.agents[1].initial_direction = 1
    env.agents[1].target = (0,4)
    
    """
    This second reset is necessary to ensure the internal bookkeeping of the RailEnv is
    consistent with the initial positions and targets of the agents. This second call
    takes care of updating env.agent_positions
    """ 
    env.reset(regenerate_rail=False,regenerate_schedule=False)

    return env

def two_agent_head_on_map(): 

    # Example generate a rail given a manual specification,
    # a map of tuples (cell_type, rotation)
    specs = [[  (0, 0),   (0, 0),  (0, 0),  (0, 0),  (0, 0), (0, 0)],
             [  (0, 0),   (0, 0),  (0, 0),  (0, 0),  (0, 0), (0, 0)],
             [(7, 270),  (1, 90), (1, 90), (1, 90), (1, 90), (7, 90)],
             [  (0, 0),   (0, 0),  (0, 0),  (0, 0),  (0, 0), (0, 0)]]
    
    env = RailEnv(width=6, height=4, rail_generator=rail_from_manual_specifications_generator(specs), number_of_agents=2)
    
    # Position the agents at their initial positions. 
    env.reset()
    env.agents[0].initial_position = (2,5)
    env.agents[0].initial_direction = 1
    env.agents[0].target = (2,0)
    env.agents[1].initial_position = (2,0)
    env.agents[1].initial_direction = 3
    env.agents[1].target = (2,5)
    
    """
    This second reset is necessary to ensure the internal bookkeeping of the RailEnv is
    consistent with the initial positions and targets of the agents. This second call
    takes care of updating env.agent_positions
    """ 
    env.reset(regenerate_rail=False,regenerate_schedule=False)

    return env

def two_agent_parallel_map(): 

    # Example generate a rail given a manual specification,
    # a map of tuples (cell_type, rotation)
    specs = [[  (0, 0),   (0, 0),  (0, 0),  (0, 0),  (0, 0),  (0, 0)],
             [  (0, 0),   (8, 0), (1, 90), (1, 90),  (8, 90), (0, 0)],
             [(7, 270),  (2, 90), (1, 90), (1, 90), (10, 270),(7, 90)],
             [  (0, 0),   (0, 0),  (0, 0),  (0, 0),  (0, 0),  (0, 0)]]
    
    env = RailEnv(width=6, height=4, rail_generator=rail_from_manual_specifications_generator(specs), number_of_agents=2)
    
    # Position the agents at their initial positions. 
    env.reset()
    env.agents[0].initial_position = (2,5)
    env.agents[0].initial_direction = 1
    env.agents[0].target = (2,0)
    env.agents[1].initial_position = (2,0)
    env.agents[1].initial_direction = 3
    env.agents[1].target = (2,5)
    
    """
    This second reset is necessary to ensure the internal bookkeeping of the RailEnv is
    consistent with the initial positions and targets of the agents. This second call
    takes care of updating env.agent_positions
    """ 
    env.reset(regenerate_rail=False,regenerate_schedule=False)

    return env

def two_agent_turnout_map(): 

    # Example generate a rail given a manual specification,
    # a map of tuples (cell_type, rotation)
    specs = [[  (0, 0),  (0, 0),  (0, 0),    (0, 0),  (0, 0),  (0, 0),  (0, 0)],
             [  (0, 0),  (0, 0),  (0, 0),    (7, 0),  (0, 0),  (0, 0),  (0, 0)],
             [(7, 270), (1, 90), (1, 90), (10, 270), (1, 90), (1, 90),  (7, 90)],
             [  (0, 0),  (0, 0),  (0, 0),    (0, 0),  (0, 0),  (0, 0),  (0, 0)]]
    
    env = RailEnv(width=6, height=4, rail_generator=rail_from_manual_specifications_generator(specs), number_of_agents=2)
    
    # Position the agents at their initial positions. 
    env.reset()
    env.agents[0].initial_position = (2,5)
    env.agents[0].initial_direction = 3
    env.agents[0].target = (2,6)
    env.agents[1].initial_position = (2,0)
    env.agents[1].initial_direction = 3
    env.agents[1].target = (2,6)
    
    """
    This second reset is necessary to ensure the internal bookkeeping of the RailEnv is
    consistent with the initial positions and targets of the agents. This second call
    takes care of updating env.agent_positions
    """ 
    env.reset(regenerate_rail=False,regenerate_schedule=False)

    return env

def two_agent_overlap_map(): 

    # Example generate a rail given a manual specification,
    # a map of tuples (cell_type, rotation)
    specs = [[  (0, 0),   (0, 0),  (0, 0),  (0, 0),  (0, 0),  (0, 0)],
             [  (0, 0),   (8, 0), (1, 90), (1, 90),  (8, 90), (0, 0)],
             [(7, 270),  (2, 90), (1, 90), (1, 90), (10, 270),(7, 90)],
             [  (0, 0),   (0, 0),  (0, 0),  (0, 0),  (0, 0),  (0, 0)]]
    
    env = RailEnv(width=6, height=4, rail_generator=rail_from_manual_specifications_generator(specs), number_of_agents=2)
    
    # Position the agents at their initial positions. 
    env.reset()
    env.agents[0].initial_position = (2,0)
    env.agents[0].initial_direction = 3
    env.agents[0].target = (2,5)
    env.agents[1].initial_position = (2,0)
    env.agents[1].initial_direction = 3
    env.agents[1].target = (2,5)
    
    """
    This second reset is necessary to ensure the internal bookkeeping of the RailEnv is
    consistent with the initial positions and targets of the agents. This second call
    takes care of updating env.agent_positions
    """ 
    env.reset(regenerate_rail=False,regenerate_schedule=False)

    return env

def random_square_map(dimension, agents, seed):
    
    # Create new environment.
    env = RailEnv(
                width=dimension,
                height=dimension,
                rail_generator=complex_rail_generator(
                                        nr_start_goal=int(1.5 * agents),
                                        nr_extra=int(1.2 * agents),
                                        min_dist=int(floor(dimension / 2)),
                                        max_dist=99999,
                                        seed=0),
                schedule_generator=complex_schedule_generator(),
                malfunction_generator_and_process_data=None,
                number_of_agents=agents)

    # Initialize positions.
    env.reset(random_seed=seed)

    return env
