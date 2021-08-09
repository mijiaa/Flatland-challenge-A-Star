'''
@author: Frits de Nijs
'''
import sys
import time
import threading

import pyglet

from flatland.utils.rendertools import RenderTool
from flatland.utils.graphics_pgl import RailViewWindow

import planpath
import Solution.src.solution as solution
import Solution.src.testmaps as testmaps


def runmap(env, schedule, debug = False, env_renderer = None, refresh = 0.1):

    env.reset(regenerate_rail=False,regenerate_schedule=False)
    if debug:
        env_renderer.reset()
        env_renderer.render_env(show=True, frames=False, show_observations=False)

    success = False;
    for action in schedule:
        _, _, _done, _ = env.step(action)
        success = all(_done.values())

        if debug:
            env_renderer.render_env(show=True, frames=False, show_observations=False)
            time.sleep(refresh)

    return success

def evalmap(env1, env2, debug = False, refresh = 0.1):

    # Create renderer.
    env_renderer = None
    if debug:
        env_renderer = RenderTool(env1, screen_width=1920, screen_height=1080)

    # Compute the reference schedule.
    expected = solution.search(env1)

    # In case of debugging, show the expected behavior.
    if debug:
        print("[INFO] Expected behavior")
        runmap(env1, expected, debug, env_renderer, refresh)

    # Compute the solution schedule.
    result = {'answer': None}
    searchthread = threading.Thread(target=lambda res, env: res.update({'answer': planpath.search(env)}), args=(result, env2,))
    searchthread.start()
    searchthread.join(timeout=60)
    if searchthread.is_alive():
        print("[ERROR] Test timed out; stop program manually.", file=sys.stderr)
        # Return False to signal we have to stop the evaluation here, and do not create more long-running threads.
        return False 

    schedule = result['answer']

    # No schedule returned means we have to stop here.
    if schedule is None:
        print("[ERROR] No schedule returned.", file=sys.stderr)
        return True

    if len(schedule) != len(expected):
        print("[ERROR] Schedule length not equal to optimal schedule length (Expected %d, actual %d)" % (len(expected), len(schedule)), file=sys.stderr)

    # In case of debugging, show the expected behavior.
    if debug:
        print("[INFO] Actual behavior")

    # Run the schedule
    success = runmap(env1, schedule, debug, env_renderer, refresh)

    if not success:
        print("[ERROR] Search implementation did not solve the environment.", file=sys.stderr)
    else:
        print("[INFO] Search implementation solved the environment")

    return True


def evaltask(title, genfun, genargs, debug, refresh):

    print("=== TESTING %s ===" % (title))

    env1 = genfun(*genargs)
    env1.reset(regenerate_rail=False,regenerate_schedule=False)

    env2 = genfun(*genargs)
    env2.reset(regenerate_rail=False,regenerate_schedule=False)

    return evalmap(env1, env2, debug, refresh)


def eval_all(debug, refresh):

    if not evaltask("Turn-out map", testmaps.two_agent_turnout_map, (), debug, refresh):
        return

    if not evaltask("Parallel map", testmaps.two_agent_parallel_map, (), debug, refresh):
        return

    if not evaltask("Collision map", testmaps.two_agent_collision_map, (), debug, refresh):
        return

    if not evaltask("Head-on map", testmaps.two_agent_head_on_map, (), debug, refresh):
        return

    if not evaltask("Overlapping map", testmaps.two_agent_overlap_map, (), debug, refresh):
        return

    seeds = [1, 2, 3, 4, 5]
    for seed in seeds:
        if not evaltask("Random 3-agent map %d" % (seed), testmaps.random_square_map, (5, 3, seed), debug, refresh):
            return

if __name__ == "__main__":

    debug = True
    refresh = 0.1

    window = None
    if debug:
        window = RailViewWindow()

    evalthread = threading.Thread(target=eval_all, args=(debug, refresh))
    evalthread.start()

    if debug:
        pyglet.clock.schedule_interval(window.update_texture, 1/120.0)
        pyglet.app.run()

    evalthread.join()
