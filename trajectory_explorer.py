import pygame
import pygame.gfxdraw
from belissimolib import *


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
    pygame.display.set_caption("Trajectory Explorer")
    clock = pygame.time.Clock()

    tot_methods = ["std", "v0y0", "solve-t-v0y0", "solve-t"]
    method = 0
    targets: list[Target] = [Target(Vector(100, 100))]
    target_radius = 10
    drag = False
    world = World(800, 600)
    click_target = None
    seekers_t = 0
    upd = True

    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if event.key == pygame.K_RIGHT:
                    method = (method + 1) % len(tot_methods)
                    upd = True
                    print(f"Method: {tot_methods[method]}")
            if event.type == pygame.VIDEORESIZE:
                # screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                world.width = event.w
                world.height = event.h

            if not drag:
                click_target = None
                if hasattr(event, "pos"):
                    for i, target in enumerate(targets):
                        if (target.pos - Vector(event.pos[0], event.pos[1])).length() < target_radius:
                            click_target = i
                            break

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if not drag:
                        targets.append(Target(Vector(event.pos[0], event.pos[1])))
                        upd = True
                    drag = False
                if event.button == 3:
                    if None is not click_target != 0:
                        targets.pop(click_target)
                        upd = True

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if click_target is not None:
                        drag = True
                        break

            if event.type == pygame.MOUSEMOTION:
                if event.buttons[0] and drag and click_target is not None:
                    targets[click_target].pos = Vector(event.pos[0], event.pos[1])
                    upd = True
        screen.fill((255, 255, 255))

        # draw targets
        for target in targets:
            pygame.gfxdraw.aacircle(screen, int(target.pos.x), int(target.pos.y), target_radius, (0, 0, 0))

        if upd:
            cur_pos = targets[0].pos
            cur_vel = Vector(0, 0)
            cur_t = 0
            segmented_nav = SegmentedFuture()
            for target in targets[1:]:
                nav = ConstAccelerationNavigation.solve_from_target(
                    target=target.pos,
                    start_pos=cur_pos,
                    start_vel=cur_vel,
                    friction=0.02,
                    thrust=0.1,
                    world=None,
                    start_time=cur_t,
                    method=tot_methods[method]
                )
                segmented_nav.segments.append(nav)
                cur_pos = nav.planned_end_pos()
                cur_vel = nav.planned_end_vel()
                cur_t = nav.arrival_time
            upd = False

        # draw trajectory
        cur_pos = targets[0].pos
        for nav in segmented_nav.segments:
            pygame.draw.aaline(screen, (0, 0, 0), tuple(cur_pos), tuple(cur_pos + nav.thrust_vector * 200))
            _prev_pos = cur_pos
            it = 40
            for i in range(0, it + 1):
                _cur_t = nav.start_time + nav.duration * (i / it) ** 2
                _nav_pos = nav.planned_pos_at(_cur_t)

                pygame.draw.aaline(screen, (0, 128, 0), tuple(_prev_pos), tuple(_nav_pos))
                _prev_pos = _nav_pos
            cur_pos = nav.planned_end_pos()

        seeker_disp_t = 20
        t = seekers_t % seeker_disp_t
        while t < segmented_nav.planned_end_time(0):
            pos = segmented_nav.planned_pos_at(t)
            pygame.gfxdraw.aacircle(screen, int(pos.x), int(pos.y), 3, (0, 0, 0))
            t += seeker_disp_t

        seekers_t += 1

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
