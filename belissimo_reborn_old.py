import time

from belissimolib import *
from seekers import Seeker

from seekers.debug_drawing import draw_circle, draw_line, draw_text

friction_movement_model = FrictionMovementModel()
seekers_targets = {}
seekers_a_vectors = {}
seekers_last_frames = {}
seekers_errors = defaultdict(Vector)
j = 1

logger = logging.getLogger("belissimo_reborn")


def decide(seekers: list[Seeker], other_seekers, all_seekers, goals, other_players, own_camp, camps, world,
           passed_playtime=None):
    global j

    if passed_playtime:
        j = passed_playtime
    else:
        j += 1
    # print("=" * 20 + f" {j} START")

    updates_performed_this_frame = 0

    for i, seeker in enumerate(seekers):
        if seeker.is_disabled:
            continue

        base_thrust = seeker.base_thrust

        upd = False
        if i not in seekers_targets or world.torus_distance(seeker.position, seekers_targets[i]) < seeker.radius:
            logger.debug(f"[{i}] seeker reached target")
            seekers_targets[i] = world.random_position()

        midp = world.middle() + Vector(0, i * 35 + 30)
        draw_circle(midp, 3, width=0)

        try:
            draw_line(midp, midp + seekers_a_vectors[i] * 15)

            seeker_pos_at_target = future_position2(seekers_last_frames[i] - j, seeker,
                                                    seekers_a_vectors[i] * base_thrust)
            world.normalize_position(seeker_pos_at_target)
            difference = seeker_pos_at_target - seekers_targets[i]

            def resolve(eta: float, seeker: Seeker, a: Vector, target_accuracy: float) -> None | float:
                for dt in range(-3, 4):
                    seeker_pos_at_target = future_position2(eta + dt, seeker, a * base_thrust)
                    world.normalize_position(seeker_pos_at_target)
                    diff_sq = (seeker_pos_at_target - seekers_targets[i]).squared_length()
                    # logger.debug(f"RESOLVE: {dt:.2f} {diff_sq:.2f}")
                    if diff_sq < target_accuracy ** 2:
                        return dt + eta

            def target_accuracy(eta: float, r: float):
                a = r + eta * 0.5 - 20
                return max(r, a)

            ta = target_accuracy(seekers_last_frames[i] - j, seeker.radius)
            diff_sq = difference.squared_length()
            if diff_sq > ta ** 2:
                upd = True

                if diff_sq < (diff_sq ** .5 + seeker.radius / 2) ** 2:
                    if (new_eta := resolve(seekers_last_frames[i] - j, seeker, seekers_a_vectors[i], ta)) is not None:
                        seekers_last_frames[i] = j + new_eta
                        # logger.debug(f"[{i}] RESOLVE SUCCESSFUL")
                        upd = False

                draw_circle(midp, 5, color=(255, 0, 0), width=0)
                draw_circle(seeker_pos_at_target, ta, (255, 0, 0), width=1)
            else:
                draw_circle(seeker_pos_at_target, ta, (0, 255, 0), width=1)

            last_pos = seeker.position
            eta = seekers_last_frames[i] - j
            steps = 20
            for t in range(steps + 1):
                t_ = (t / steps) ** 2
                timestep = t_ * eta

                next_pos = future_position2(timestep, seeker, seekers_a_vectors[i] * base_thrust)
                draw_line(last_pos, next_pos, width=1, color=(0, 100, 0))
                # draw_circle(position=next_pos, radius=seeker.radius, width=1, color=(0, 70, 0))
                last_pos = next_pos


        except KeyError:
            upd = True

        draw_circle(position=seekers_targets[i], radius=seeker.radius, width=1)
        draw_text(f"{i}", seeker.position, color=(255, 255, 255))

        if upd and (updates_performed_this_frame < 1):  # or (j + i * (50 // len(seekers))) % 50 == 0:

            updates_performed_this_frame += 1
            t1 = time.perf_counter()
            a, eta = Solvers.solve_const_acc(
                friction_movement_model,
                seeker.velocity,
                seekers_targets[i] - seeker.position,
                world,
                a=base_thrust * 1
            )
            a = a / (base_thrust * 1)
            a_vector = a * 1

            dt = time.perf_counter() - t1
            # logger.debug(f"[{i}] solve_for_const_acceleration_torus took {dt:.3f}s ({1 / dt:.0f} fps)")

            seekers_a_vectors[i] = a_vector
            seekers_last_frames[i] = eta + j
            if i in seekers_errors: del seekers_errors[i]
            logger.debug(f"[{i}] Reroute, eta={eta:.2f}")
        try:
            a, seekers_errors[i] = balance_a_with_error2(seekers_a_vectors[i], seekers_errors[i])
            seeker.target = seeker.position + a * 20
        except KeyError:
            pass

    for i, goal in enumerate(goals):
        if goal.velocity.squared_length() < 0.001:
            continue

        t_pos = terminal_position2(goal)
        draw_line(goal.position, t_pos, width=1, color=(0, 50, 0))
        draw_circle(t_pos, goal.radius, width=1, color=(60, 60, 60))

    # print("=" * 20 + f" {j}")
    draw_text(f"{j}", world.middle(), color=(255, 255, 255))

    return seekers


def main():
    ...


if __name__ == "__main__":
    main()
