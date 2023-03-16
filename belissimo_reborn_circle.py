from belissimolib import *

from seekers import Seeker
from seekers.debug_drawing import draw_text

logger = logging.getLogger("belissimo_reborn")

game_strat = GameStrategy(plan_limit=5)
angle = math.pi


def circle_pos():
    global angle
    angle += 2 * math.pi / 7

    return Vector.from_polar(angle, 250)


def decide(seekers: list[Seeker], other_seekers, all_seekers, goals, other_players, own_camp, camps, world,
           current_time):
    game_strat.update(seekers, current_time)

    for seeker, agent in zip(seekers, game_strat.agents):
        if len(agent.future.future_targets) < 1:
            agent.future.future_targets.append(Target(circle_pos() + world.middle()))

    game_strat.update_navigation(seekers, world, current_time)
    game_strat.debug_draw(seekers, goals, world, current_time)

    draw_text(f"{current_time} {game_strat.last_want_solves}",
              Vector(5, world.height / 2), color=(255, 255, 255), center=False)

    return seekers
