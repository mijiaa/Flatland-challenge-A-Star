import time
import threading

import pyglet

from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.utils.graphics_pgl import RailViewWindow
from math import floor

import finalplanpath

def evalfun(debug = False, refresh = 0.1):
    # A list of (mapsize, agent count) tuples, change or extend this to test different sizes.
    # problemsizes = [(5, 1), (7, 2), (40, 20)]

    #red pink purple dark pur
    # problemsizes = [(9,6),(10,7)]
    # problemsizes =  [(5, 2), (6, 3), (7, 4), (8, 5)]
    # problemsizes = [(9,2),(9,3),(9,4),(6,2),(7,2),(7,3),(8,2),(8,3),(8,4)]
    problemsizes = [(5,3),(6,3),(7,3),(8,3),(9,3),(10,3),(11,3),(12,3),(13,3),(14,3),(15,3),(16,3),(17,3),(18,3),(19,3),(20,3),(30,3)]
    # problemsizes = [(10,2),(10,3),(10,4),(10,5),(10,6),(10,7),(10,8)]
    # problemsizes = [(6,3),(10,3),(15,3),(20,3),(30,3),(35,3),(45,3),(50,3),(60,3),(65,3)]
    # problemsizes = [(100,50)]
    _seed = 2984379
    
    print("%10s\t%8s\t%9s" % ("Dimensions", "Success", "Runtime"))
    for problemsize in problemsizes:
    
        dimension = problemsize[0]
        NUMBER_OF_AGENTS = problemsize[1];
    
        # Create new environment.
        env = RailEnv(
                    width=dimension,
                    height=dimension,
                    rail_generator=complex_rail_generator(
                                            nr_start_goal=int(1.5 * NUMBER_OF_AGENTS),
                                            nr_extra=int(1.2 * NUMBER_OF_AGENTS),
                                            min_dist=int(floor(dimension / 2)),
                                            max_dist=99999,
                                            seed=0),
                    schedule_generator=complex_schedule_generator(),
                    malfunction_generator_and_process_data=None,
                    number_of_agents=NUMBER_OF_AGENTS)
    
        env_renderer = RenderTool(env, screen_width=1920, screen_height=1080)
    
        # Initialize positions.
        env.reset(random_seed=_seed)
    
        # Time the search.
        start = time.time()
        schedule = finalplanpath.search(env)
        duration = time.time() - start;
    
        if debug:
            env_renderer.render_env(show=True, frames=False, show_observations=False)
            time.sleep(refresh)
    
        # Validate that environment state is unchanged.
        assert env.num_resets == 1 and env._elapsed_steps == 0
    
        # Run the schedule
        success = False;
        for action in schedule:
            _, _, _done, _ = env.step(action)
            success = _done['__all__']
            if debug:
                print(action)
                env_renderer.render_env(show=True, frames=False, show_observations=False)
                time.sleep(refresh)
    
        # Print the performance of the algorithm
        print("%10s\t%8s\t%9.6f" % (str(problemsize), str(success), duration))


if __name__ == "__main__":

    _debug = True
    _refresh = 0.3

    if (_debug):
        window = RailViewWindow()

    evalthread = threading.Thread(target=evalfun, args=(_debug,_refresh,))
    evalthread.start()

    if (_debug):
        pyglet.clock.schedule_interval(window.update_texture, 1/120.0)
        pyglet.app.run()

    evalthread.join()
