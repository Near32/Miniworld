#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import argparse
import math

import gymnasium as gym
import pyglet
from pyglet.window import key

import miniworld

# import sys

def str2bool(arg):
    if arg is None:
        return False
    if isinstance(arg, bool):
        return arg
    if isinstance(arg, str):
        if 'true' in arg.lower():
            return True
        else:
            return False
    else:
        raise NotImplementedError 


parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=miniworld.envs.env_ids[0])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--domain-rand", action="store_true", help="enable domain randomization"
)
parser.add_argument(
    "--no-time-limit", action="store_true", help="ignore time step limits"
)
parser.add_argument(
    "--entity_visibility_oracle", action="store_true", help="enable entity visibility oracle"
)
parser.add_argument("--too_far_threshold", type=float, default=-1)
parser.add_argument("--collision", type=str2bool, default=True)
parser.add_argument(
    "--include_discrete_depth", action="store_true", help="adds discrete depth to entity visibility oracle"
)
parser.add_argument(
    "--top_view",
    action="store_true",
    help="show the top view instead of the agent view",
)
args = parser.parse_args()
view_mode = "top" if args.top_view else "agent"

env = gym.make(
    args.env_name, 
    view=view_mode, 
    render_mode="human",
    num_objs=15,
    #room_size=3,
    collision=args.collision,
)

if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True
if args.entity_visibility_oracle:
    from miniworld.wrappers import EntityVisibilityOracleWrapper
    env = EntityVisibilityOracleWrapper(
        env, 
        verbose=True, 
        with_top_view=True,
        include_discrete_depth=args.include_discrete_depth,
        too_far_threshold=args.too_far_threshold,
    )

print("============")
print("Instructions")
print("============")
print("move: arrow keys\npickup: P\ndrop: D\ndone: ENTER\nquit: ESC")
print("============")

env.reset(seed=args.seed)

# Create the display window
env.render()


def step(action):
    global args 
    print(
        "step {}/{}: {}".format(
            env.step_count + 1, env.max_episode_steps, env.actions(action).name
        )
    )

    obs, reward, termination, truncation, info = env.step(action)

    if reward > 0:
        print(f"reward={reward:.2f}")

    if termination or truncation:
        print("done!")
        env.reset(seed=args.seed)
    
    if "visible_entities" in info:
        print(f"Visible entities: {info['visible_entities']}")
    
    env.render()

    if isinstance(obs, dict):
        if "mission" in obs:
            print(f"Mission: {obs['mission']}")
        if "agent_pos_in_top_view" in obs:
            print(f"The agent is at: {obs['agent_pos_in_top_view']}")


    

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
        return

    if symbol == key.ESCAPE:
        env.close()
        # sys.exit(0)

    if symbol == key.UP:
        step(env.actions.move_forward)
    elif symbol == key.DOWN:
        step(env.actions.move_back)

    elif symbol == key.LEFT:
        step(env.actions.turn_left)
    elif symbol == key.RIGHT:
        step(env.actions.turn_right)

    elif symbol == key.PAGEUP or symbol == key.P:
        step(env.actions.pickup)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(env.actions.drop)

    elif symbol == key.ENTER:
        step(env.actions.done)


@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass


@env.unwrapped.window.event
def on_draw():
    env.render()


@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()


# Enter main event loop
pyglet.app.run()

env.close()
