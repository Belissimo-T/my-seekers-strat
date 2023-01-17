import time
from collections import defaultdict

from belissimolib import *

from seekers import Seeker
from seekers.debug_drawing import draw_circle, draw_line, draw_text

logger = logging.getLogger("belissimo_reborn")

agents: list[Agent] = []
goal_futures: dict[str, GoalFuture] = {}


def decide(seekers: list[Seeker], other_seekers, all_seekers, goals, other_players, own_camp, camps, world,
           current_time):
    while len(agents) < len(seekers):
        agents.append(Agent())

    want_solves = defaultdict(list)

    def solve_mgr(id_: str, solve):
        want_solves[id_].append(solve)

    for i, (seeker, agent) in enumerate(zip(seekers, agents)):
        if len(agent.future.future_targets) < 1:
            agent.future.future_targets.append(Target(world.random_position()))

        agent.update_navigation(seeker, world, current_time, solve_mgr)
        agent.debug_draw(seeker, current_time, world)

    if want_solves:
        key = random.choice(list(want_solves.keys()))
        want_solves[key][0]()

    for seeker, agent in zip(seekers, agents):
        agent.update_seeker_target(seeker, current_time)

    for goal in goals:
        if goal.id not in goal_futures:
            goal_futures[goal.id] = GoalFuture()

        goal_futures[goal.id].update(goal, current_time)
        goal_futures[goal.id].debug_draw(current_time, world, goal)

    t1 = time.perf_counter()
    # noinspection PyTypeChecker
    entity_fut: list[EntityFuture] = [agent.future for agent in agents] + [f for f in goal_futures.values()]
    collisions = get_collisions(entity_fut, current_time, world, seekers[0].config, max_time=200)
    t2 = time.perf_counter()

    for (fut1_i, fut2_i), col_time in collisions.items():
        pos1 = entity_fut[fut1_i].pos_at(current_time)
        pos2 = entity_fut[fut2_i].pos_at(current_time)

        world.normalize_position(pos1)
        world.normalize_position(pos2)

        col_pos = entity_fut[fut1_i].pos_at(col_time)
        world.normalize_position(col_pos)

        draw_circle(col_pos, 10, (255, 0, 0), 0)
        draw_line(col_pos, pos1, (255, 0, 0), 1)
        draw_line(col_pos, pos2, (255, 0, 0), 1)
        draw_text(str(int((col_time - current_time) / 10)), col_pos, (255, 255, 255))

    draw_text(f"{current_time} {sum(len(val) for val in want_solves.values())} {t2 - t1:.3f}",
              Vector(5, world.height / 2), color=(255, 255, 255), center=False)

    return seekers
