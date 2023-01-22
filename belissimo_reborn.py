from belissimolib import *

from seekers import Seeker
from seekers.debug_drawing import draw_circle, draw_line, draw_text

logger = logging.getLogger("belissimo_reborn")

strat = GameStrategy()


def decide(seekers: list[Seeker], other_seekers, all_seekers, goals, other_players, own_camp, camps, world,
           current_time):
    strat.update(seekers)

    for agent in strat.agents:
        if len(agent.future.future_targets) < 1:
            agent.future.future_targets.append(Target(world.random_position()))

    strat.update_navigation(seekers, world, current_time)
    strat.update_goals(goals, current_time)

    entity_futures, collisions = strat.get_collisions(world, current_time, config=seekers[0].config)

    for (fut1_i, fut2_i), col_time in collisions.items():
        pos1 = entity_futures[fut1_i].pos_at(current_time)
        pos2 = entity_futures[fut2_i].pos_at(current_time)

        world.normalize_position(pos1)
        world.normalize_position(pos2)

        col_pos = entity_futures[fut1_i].pos_at(col_time)
        world.normalize_position(col_pos)

        draw_circle(col_pos, 10, (255, 0, 0), 0)
        draw_line(col_pos, pos1, (255, 0, 0), 1)
        draw_line(col_pos, pos2, (255, 0, 0), 1)
        draw_text(str(int((col_time - current_time) / 10)), col_pos, (255, 255, 255))

    strat.debug_draw(seekers, goals, world, current_time)
    draw_text(f"{current_time} {strat.last_want_solves} {strat.last_collision_time:.3f}",
              Vector(5, world.height / 2), color=(255, 255, 255), center=False)

    return seekers
