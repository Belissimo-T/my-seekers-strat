import time
from collections import defaultdict

from belissimolib import *

from seekers import Seeker
from seekers.debug_drawing import draw_circle, draw_line, draw_text

logger = logging.getLogger("belissimo_reborn")

agents: list[Agent] = []

angle = 0


def circle_pos():
    global angle
    angle += 2 * math.pi / 7

    return Vector.from_polar(angle, 250)


def decide(seekers: list[Seeker], other_seekers, all_seekers, goals, other_players, own_camp, camps, world,
           current_time):
    while len(agents) < len(seekers):
        agents.append(Agent(plan_limit=5))

    want_solves = defaultdict(list)

    def solve_mgr(id_: str, solve):
        want_solves[id_].append(solve)

    for i, (seeker, agent) in enumerate(zip(seekers, agents)):
        if len(agent.future.future_targets) < 1:
            agent.future.future_targets.append(Target(circle_pos() + world.middle()))

        agent.update_navigation(seeker, world, current_time, solve_mgr)
        agent.debug_draw(seeker, current_time, world)

    if want_solves:
        key = random.choice(list(want_solves.keys()))
        want_solves[key][0]()

    for seeker, agent in zip(seekers, agents):
        agent.update_seeker_target(seeker, current_time)

    draw_text(f"{current_time} {sum(len(val) for val in want_solves.values())}",
              Vector(5, world.height / 2), color=(255, 255, 255), center=False)

    return seekers
