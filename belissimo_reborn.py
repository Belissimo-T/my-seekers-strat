import time

from belissimolib import *

from seekers import Seeker
from seekers.debug_drawing import draw_circle, draw_line, draw_text

logger = logging.getLogger("belissimo_reborn")

# noinspection PyTypeChecker
col_env = CollisionEnv(
    world=None,
    physical_friction=None,
    seeker_radius=None,
    seeker_base_thrust=None
)
col_mgr = CollisionManager(col_env)
strat = GameStrategy(plan_limit=4, col_mgr=col_mgr)


def decide(seekers: list[Seeker], other_seekers, all_seekers, goals, other_players, own_camp, camps, world,
           current_time):
    col_env.world = world
    col_env.physical_friction = seekers[0].config.physical_friction
    col_env.seeker_radius = seekers[0].config.seeker_radius
    col_env.seeker_base_thrust = get_thrust(seekers[0], 0)

    strat.update(seekers, current_time)

    for agent in strat.agents:
        if len(agent.future.future_targets) < 1:
            agent.future.future_targets.append(Target(world.random_position()))

    strat.update_navigation(seekers, world, current_time)
    strat.update_goals(goals, current_time)

    cllsns = strat.get_collisions(current_time, 500)
    for col in cllsns:
        pos1, pos2 = col.pos1, col.pos2
        world.normalize_position(pos1)
        world.normalize_position(pos2)
        # pos = (pos1 + pos2) / 2

        # tone = min(255, 50 / (col.time - current_time + 0.5) * 255)
        # draw_circle(pos, get_target_accuracy(col.time - current_time, col.future1.radius + col.future2.radius),
        #             (tone, 0, 0), 5)

        draw_circle(pos1, col.future1.radius + 1, (255, 0, 0), 2)
        draw_circle(pos2, col.future2.radius + 1, (255, 0, 0), 2)
        # draw_circle(pos, col.future1.radius*2, (255, 0, 0), 1)
        draw_line(pos1, world.normalized_position(col.future1.pos_at(current_time)), (255, 0, 0), 1)
        draw_line(pos2, world.normalized_position(col.future2.pos_at(current_time)), (255, 0, 0), 1)
        # draw_text(str(int((col.time - current_time) / 10 + 1)), pos + Vector(0, -25), (255, 255, 255))

    t1 = time.perf_counter()
    strat.debug_draw(seekers, goals, world, current_time)
    debug_time = time.perf_counter() - t1

    draw_text(
        f"{current_time} {strat.last_want_solves} C: {strat.last_collision_time:.4f} C:{strat.collision_mgr.info} "
        f"D: {debug_time:.4f}",
        Vector(5, world.height / 2), color=(255, 255, 255), center=False)

    return seekers
