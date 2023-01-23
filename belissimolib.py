import abc
import dataclasses
import logging
import math
import random
import time as time_module
from collections import defaultdict
import scipy.optimize
import typing

from seekers import Physical, Vector, World, Seeker, Goal, Config, Color

logger = logging.getLogger("belissilib")


class ConstAccelerationMovementModel(abc.ABC):
    @abc.abstractmethod
    def get_position_of_t(self, t: float, a: float = 0, v_0: float = 0) -> float:
        """Returns the position of a Physical at time t given its acceleration a and initial velocity v_0."""

    @abc.abstractmethod
    def get_velocity_of_t(self, t: float, a: float = 0, v_0: float = 0) -> float:
        """Returns the velocity of a Physical at time t given its acceleration a and initial velocity v_0."""


class NoFrictionMovementModel(ConstAccelerationMovementModel):
    """A movement model that is based on the flawed assumption that friction is zero. It is simple though. Use
    FrictionMovementModel instead."""

    def get_position_of_t(self, t: float, a: float = 0, v_0: float = 0) -> float:
        return 1 / 2 * a * t ** 2 + v_0 * t

    def get_velocity_of_t(self, t: float, a: float = 0, v_0: float = 0) -> float:
        return a * t + v_0


class FrictionMovementModel(ConstAccelerationMovementModel):
    """A movement model that incorporates friction and was derived mathematically."""

    @staticmethod
    def get_velocity_of_t_any_f(t: float, a: float = 0, v_0: float = 0, f: float = .02) -> float:
        c = 1 - f
        return v_0 * c ** t + a * (c ** t - 1) / -f

    @staticmethod
    def get_position_of_t_any_f(t: float, a: float = 0, v_0: float = 0, f: float = .02) -> float:
        c = 1 - f
        return (a * (c ** t - t * math.log(c) - 1) - f * v_0 * (c ** t - 1)) / (-f * math.log(c))

    @staticmethod
    def get_max_velocity_any_f(a: float, f: float = .02) -> float:
        return -a / (f - 1)

    @staticmethod
    def get_velocity_of_t_a0(t: float, v_0: float = 0, f: float = .02) -> float:
        c = 1 - f
        return v_0 * c ** t

    @staticmethod
    def get_position_of_t_a0(t: float, v_0: float = 0, f: float = .02) -> float:
        c = 1 - f
        return v_0 * (c ** t - 1) / math.log(c)

    @classmethod
    def get_terminal_position_any_f(cls, a: float = 0, v_0: float = 0, f: float = .02) -> float:
        if a != 0:
            # haven't found a closed form with a yet, so we have to rely on t=big_value...
            return cls.get_position_of_t_any_f(1_000_000_000, a, v_0, f)

        return -v_0 / math.log(1 - f)

    def __init__(self, friction: float = .02):
        self.friction = friction
        self.log_c = math.log(1 - friction)

        # import sympy as sp
        #
        # v_0, a, t, f = sp.symbols("v_0 a t f")
        #
        # c = 1 - f
        #
        # v_of_t = v_0 * c ** t + a * (c ** t - 1) / (c - 1)
        #
        # pos_of_t = sp.integrate(v_of_t, (t, 0, t))
        #
        # pos_of_t_wolfram = (a * (c ** t - t * sp.log(c) - 1) + (c - 1) * v_0 * (c ** t - 1)) / ((c - 1) * sp.log(c))
        #
        # pos_of_t_a0 = pos_of_t_wolfram.subs(a, 0)
        """
        The goal is to find a function that calculates the position of a
        seeker at time t when a constant acceleration is applied and friction is accounted for.
        Note: Both axes are completely independent. A MovementModel only models one axis.

        From reading the source code we can formulate the following equation:

        t = number of passed frames
        f = friction
        v_0 = initial velocity                  (along the chosen axis)
        a = acceleration of the seeker          (along the chosen axis)
        v(t) = velocity of the seeker at time t (along the chosen axis)

        each frame, the friction is applied, then the acceleration
        v(t) = (((v_0 * (1 - f) + a) * (1 - f) + a) * (1 - f) + a)...
                        - - - - - - - - - - t times - - - - - - - - -

        let c = 1 - f
        
        simplify via wolfram alpha

        v(t) = v_0 * c**t + a *  c**(t-1) + a * c**(t-2) + ... + a * c**1 + a * c**0
             = v_0 * c**t + a * (c**(t-1) +     c**(t-2) + ... +     c**1 +     c**0)  # factor out a
             = v_0 * c**t + a * sum(i=0, t-1, c**i)

        from https://opendsa-server.cs.vt.edu/OpenDSA/Books/CS3/html/Summations.html
                      sum(i=0, n, a**i) = (a ** (n + 1) - 1) / (a - 1)

        v(t) = v_0 * c**t + a * (c ** t - 1) / (c - 1)

        by integrating we get the position of the object at time t

        wolframalpha:
            p(t) = integral_0^t (v_0 c^t + a (c^t - 1)/(c - 1)) dt 
                 = (a (c^t - t log(c) - 1) + (c - 1) v_0 (c^t - 1))/((c - 1) log(c))

        == Terminal Position ==
        we can actually get a much simpler formula for p(t) when a = 0:
            p(t) = v_0*(c**t - 1)/log(c)
        
        plugging that into wolframalpha gives us the following limit:
            lim(t->inf) v_0*(c**t - 1)/log(c) = -v_0/log(c)
        
        == Terminal Velocity ==
        solving this equation:
            v(t) = v(t + 1)
            v = v * f + a
            
            => v = -a / (f - 1) 
        """

    def get_position_of_t(self, t: float, a: float = 0, v_0: float = 0) -> float:
        return self.get_position_of_t_any_f(t, a, v_0, self.friction)

    def get_velocity_of_t(self, t: float, a: float = 0, v_0: float = 0) -> float:
        return self.get_velocity_of_t_any_f(t, a, v_0, self.friction)


def solve_for_const_acceleration(movement_model: ConstAccelerationMovementModel,
                                 v0: tuple[float, float],
                                 target: tuple[float, float] | Vector,
                                 a: float,
                                 normalized: bool = True
                                 ) -> tuple[Vector, float]:
    v0x, v0y = v0

    def angle_to_acceleration(angle: float) -> tuple[float, float]:
        return math.cos(angle) * a, math.sin(angle) * a

    def d(arg: tuple[float, float]):
        #            angle  t
        angle, t = arg

        a_x, a_y = angle_to_acceleration(angle)

        p_x = movement_model.get_position_of_t(t, a_x, v0x)
        p_y = movement_model.get_position_of_t(t, a_y, v0y)

        return (p_x - target[0]) ** 2 + (p_y - target[1]) ** 2 + ((-t * 10_000) if t < 0 else 0)

    for _ in range(10):
        # noinspection PyTypeChecker
        res = scipy.optimize.minimize(d, (random.random() * 2 * math.pi, random.random() * 500))

        if res.fun < 10 and res.x[1] > 0:
            # good enough
            break

    # noinspection PyUnboundLocalVariable
    res_angle, res_t = res.x
    if res_t < 0:
        logger.warning(f"Only invalid solutions found! (t={res_t})")

    res_a_x, res_a_y = angle_to_acceleration(res_angle)

    if normalized:
        return (1 / a) * Vector(res_a_x, res_a_y), res_t
    else:
        return Vector(res_a_x, res_a_y), res_t


def solve_for_const_acceleration_and_velocity(movement_model: ConstAccelerationMovementModel,
                                              v0: tuple[float, float],
                                              target: tuple[float, float] | Vector,
                                              target_velocity: tuple[float, float] | Vector
                                              ) -> tuple[Vector, float]:
    v0x, v0y = v0

    def angle_to_acceleration(angle: float, a: float) -> tuple[float, float]:
        return math.cos(angle) * a, math.sin(angle) * a

    def d(arg: tuple[float, float, float]):
        #            angle  t      a
        angle, t, a = arg

        a_x, a_y = angle_to_acceleration(angle, a)

        p_x = movement_model.get_position_of_t(t, a_x, v0x)
        p_y = movement_model.get_position_of_t(t, a_y, v0y)

        dist = (p_x - target[0]) ** 2 + (p_y - target[1]) ** 2

        vel_dist = (
                (movement_model.get_velocity_of_t(t, a_x, v0x) - target_velocity[0]) ** 2
                + (movement_model.get_velocity_of_t(t, a_y, v0y) - target_velocity[1]) ** 2
        )

        return dist * vel_dist

    for _ in range(10):
        # noinspection PyTypeChecker
        res = scipy.optimize.minimize(d, (random.random() * 2 * math.pi, random.random() * 500, 0.1))

        if res.fun < 10 and res.x[1] > 0:
            # good enough
            break

    # noinspection PyUnboundLocalVariable
    res_angle, res_t, res_a = res.x
    if res_t < 0:
        logger.warning(f"Only invalid solutions found! (t={res_t})")

    res_a_x, res_a_y = angle_to_acceleration(res_angle, res_a)

    return Vector(res_a_x, res_a_y), res_t


def solve_for_const_acceleration_torus(movement_model: ConstAccelerationMovementModel,
                                       v0: tuple[float, float],
                                       target: Vector,
                                       world: World,
                                       a: float,
                                       normalized: bool = True,
                                       *,
                                       only_consider_nearest_n: int = 3
                                       ) -> tuple[Vector, float]:
    possible_positions = [
        Vector(i * world.width + target.x, j * world.width + target.y) for i in [-1, 0, 1] for j in [-1, 0, 1]
    ]

    viable_postions = sorted(possible_positions, key=lambda p: p.squared_length())[:only_consider_nearest_n]

    record_res = None

    for pos in viable_postions:
        res = solve_for_const_acceleration(movement_model, v0, pos, a, normalized)
        if record_res is None or record_res[1] > res[1] > 0:
            record_res = res

    return record_res


def resolve_new_eta(eta: float, seeker: Seeker, target: Vector, world: World, a: Vector, target_accuracy: float,
                    min_dt=-3, max_dt=6) -> None | float:
    """Try to find a new eta that is closer to the target."""
    for dt in range(min_dt, max_dt):
        seeker_pos_at_target = future_position2(eta + dt, seeker, a)

        world.normalize_position(seeker_pos_at_target)
        diff_sq = (seeker_pos_at_target - target).squared_length()

        if diff_sq < target_accuracy ** 2:
            return dt + eta


def future_position(time: float, acceleration: Vector = Vector(), velocity: Vector = Vector(),
                    position: Vector = Vector(), friction: float = .02) -> Vector:
    return Vector(
        FrictionMovementModel.get_position_of_t_any_f(time, acceleration.x, velocity.x, friction) + position.x,
        FrictionMovementModel.get_position_of_t_any_f(time, acceleration.y, velocity.y, friction) + position.y
    )


def future_position2(time: float, physical: Physical, acceleration: Vector = Vector()) -> Vector:
    return future_position(time, acceleration, physical.velocity, physical.position, physical.config.physical_friction)


def future_velocity(time: float, acceleration: Vector = Vector(), velocity: Vector = Vector(),
                    friction: float = .02) -> Vector:
    return Vector(
        FrictionMovementModel.get_velocity_of_t_any_f(time, acceleration.x, velocity.x, friction),
        FrictionMovementModel.get_velocity_of_t_any_f(time, acceleration.y, velocity.y, friction)
    )


def future_velocity2(time: float, physical: Physical, acceleration: Vector = Vector()) -> Vector:
    return future_velocity(time, acceleration, physical.velocity, physical.config.physical_friction)


def terminal_position(acceleration: Vector = Vector(), velocity: Vector = Vector(), position: Vector = Vector(),
                      friction: float = .02) -> Vector:
    return Vector(
        FrictionMovementModel.get_terminal_position_any_f(acceleration.x, velocity.x, friction) + position.x,
        FrictionMovementModel.get_terminal_position_any_f(acceleration.y, velocity.y, friction) + position.y
    )


def terminal_position2(physical: Physical, acceleration: Vector = Vector(0, 0)) -> Vector:
    return terminal_position(acceleration, physical.velocity, physical.position, physical.config.physical_friction)


def max_velocity(max_acceleration: float, friction: float) -> float:
    return FrictionMovementModel.get_max_velocity_any_f(max_acceleration, friction)


def t_of_max_angle_velocity(a_x: float, a_y: float, v_0_x: float, v_0_y: float, f: float) -> float:
    return math.log(
        (a_x ** 2 + a_y ** 2) / (
                a_x ** 2 - 2 * a_x * f * v_0_x +
                a_y ** 2 - 2 * a_y * f * v_0_y +

                f ** 2 * v_0_x ** 2 +
                f ** 2 * v_0_y ** 2
        )
    ) / (2 * math.log(1 - f))


def balance_a_random(a: Vector) -> Vector:
    return a if random.random() < a.length() else -a


def balance_a_with_error(a: Vector, error: float = 0) -> tuple[Vector, float]:
    if error < a.length():
        return a.normalized(), error + (1 - a.length())

    return Vector(0, 0), error - a.length()


def balance_a_with_error2(a: Vector, error: Vector = Vector()) -> tuple[Vector, Vector]:
    target = -error + a

    out = target.normalized()

    error += out - a

    return out, error


def normalize_line(world: World, a: Vector, b: Vector) -> tuple[Vector, Vector]:
    d = b - a

    an = world.normalized_position(a)

    return an, an + d


def get_target_accuracy(eta: float, r: float):
    a = r + eta * 0.3 - 5
    return min(max(r, a), r * 3)


@dataclasses.dataclass
class Target:
    pos: Vector | None
    vel: Vector | None = None


class Navigation(abc.ABC):
    start_pos: Vector
    start_vel: Vector
    target: Target
    start_time: float
    arrival_time: float
    helper_navigation: bool = False

    last_seeker_info: "SeekerInfo | None" = None

    _path_color: tuple[int, int, int]

    @abc.abstractmethod
    def planned_pos_at(self, time: float) -> Vector:
        ...

    @abc.abstractmethod
    def pos_at(self, time: float) -> Vector:
        ...

    @abc.abstractmethod
    def planned_vel_at(self, time: float) -> Vector:
        ...

    @abc.abstractmethod
    def vel_at(self, time: float) -> Vector:
        ...

    @abc.abstractmethod
    def planned_end_pos(self) -> Vector:
        ...

    @abc.abstractmethod
    def end_pos(self) -> Vector:
        ...

    @abc.abstractmethod
    def planned_end_vel(self) -> Vector:
        ...

    @abc.abstractmethod
    def end_vel(self) -> Vector:
        ...

    @abc.abstractmethod
    def get_acceleration(self, time: float) -> Vector:
        ...

    @abc.abstractmethod
    def pos_error2(self, physical: Physical, current_time: float, world: World) -> float:
        ...

    def __contains__(self, item: float):
        return self.start_time <= item <= self.arrival_time

    def debug_draw(self, current_time: float, world: World, physical: Physical, index: int, steps: int
                   ) -> Color:
        from seekers.debug_drawing import draw_line, draw_circle

        if index == 0:
            # noinspection PyTypeChecker
            path_color: tuple[int, int, int] = tuple(map(lambda x: min(255, x + 100), self._path_color))
            start_time = current_time
        else:
            path_color = self._path_color
            start_time = self.start_time

        eta = self.arrival_time - start_time
        last_pos = self.pos_at(start_time)
        for t in range(1, steps + 1):
            t_ = (t / steps) ** 2
            time = start_time + t_ * eta

            next_pos = self.pos_at(time)

            a, b = normalize_line(world, last_pos, next_pos)

            draw_line(a, b, width=1, color=path_color)

            # draw_circle(position=next_pos, radius=seeker.radius, width=1, color=(0, 70, 0))
            last_pos = next_pos.copy()

        if self.target.pos is not None:
            draw_circle(self.target.pos, physical.radius, width=1, color=(255, 255, 255))

        if self.target.pos is not None:
            color = (255, 255, 0)

            if index != 0:
                color = (70, 70, 0)

            world.normalize_position(last_pos)
            draw_circle(last_pos,
                        get_target_accuracy(
                            self.arrival_time - current_time, physical.radius
                        ), color=color, width=1)

        return path_color


class ConstAccelerationNavigation(Navigation):
    _path_color = (0, 100, 0)

    def __init__(self,
                 thrust_vector: Vector,
                 start_time: float, arrival_time: float,
                 target: Vector | None,
                 start_pos: Vector, start_vel: Vector,
                 friction: float):
        self.thrust_vector = thrust_vector
        self.start_time = start_time
        self.arrival_time = arrival_time
        self.target = Target(target)
        self.start_pos = start_pos
        self.start_vel = start_vel
        self.friction = friction

        self.last_seeker_info: SeekerInfo | None = None

    def planned_pos_at(self, time: float) -> Vector:
        dt = time - self.start_time
        return future_position(
            dt,
            self.thrust_vector,
            self.start_vel,
            self.start_pos,
            self.friction
        )

    def pos_at(self, time: float) -> Vector:
        if self.last_seeker_info is None:
            return self.planned_pos_at(time)

        return future_position(
            time - self.last_seeker_info.time,
            self.get_acceleration(time),
            self.last_seeker_info.velocity,
            self.last_seeker_info.position,
            self.last_seeker_info.friction,
        )

    def planned_vel_at(self, time: float) -> Vector:
        dt = time - self.start_time
        return future_velocity(
            dt,
            self.thrust_vector,
            self.start_vel,
            self.friction
        )

    def vel_at(self, time: float) -> Vector:
        if self.last_seeker_info is None:
            return self.planned_vel_at(time)

        return future_velocity(
            time - self.last_seeker_info.time,
            self.get_acceleration(time),
            self.last_seeker_info.velocity,
            self.last_seeker_info.friction,
        )

    def planned_end_pos(self) -> Vector:
        return self.planned_pos_at(self.arrival_time)

    def end_pos(self) -> Vector:
        return self.pos_at(self.arrival_time)

    def planned_end_vel(self) -> Vector:
        return self.planned_vel_at(self.arrival_time)

    def end_vel(self) -> Vector:
        return self.vel_at(self.arrival_time)

    def get_acceleration(self, time: float) -> Vector:
        # constant acceleration
        return self.thrust_vector

    def pos_error2(self, physical: Seeker | None, current_time: float, world: World) -> float:
        if self.target.pos is None:
            # target None means navigation is always correct
            return 0

        if physical is not None and get_thrust(physical) ** 2 - self.thrust_vector.squared_length() > 1e-6:
            # thrust changed
            return float('inf')

        if physical is not None:
            fut_pos = future_position2(self.arrival_time - current_time, physical, self.thrust_vector)
        else:
            fut_pos = self.end_pos()
        fut_planned_pos = self.planned_end_pos()

        world.normalize_position(fut_pos)
        world.normalize_position(fut_planned_pos)

        return world.torus_difference(fut_pos, fut_planned_pos).squared_length()

    @classmethod
    def solve_from_target(cls,
                          target: Vector,
                          start_pos: Vector,
                          start_vel: Vector,
                          friction: float,
                          thrust: float,
                          world: World,
                          start_time: float
                          ) -> "ConstAccelerationNavigation":
        # noinspection PyTypeChecker
        thrust_vec, eta = solve_for_const_acceleration_torus(
            FrictionMovementModel(friction),
            tuple(start_vel),
            target - start_pos,
            world,
            thrust,

            normalized=False
        )

        return cls(thrust_vec, start_time, start_time + eta, target, start_pos, start_vel, friction)

    def max_angle_velocity(self) -> float:
        return self.start_time + t_of_max_angle_velocity(
            *self.thrust_vector,
            *self.start_vel,
            f=self.friction
        )

    def debug_draw(self, current_time: float, world: World, physical: Physical, index: int, steps: int
                   ) -> Color:
        path_color = super().debug_draw(current_time, world, physical, index, steps)

        try:
            max_angle_vel_t = self.max_angle_velocity()
        except ValueError:
            return path_color

        if max_angle_vel_t not in self or current_time > max_angle_vel_t:
            return path_color

        pos = self.pos_at(max_angle_vel_t)
        world.normalize_position(pos)
        vel = self.vel_at(max_angle_vel_t)

        from seekers.debug_drawing import draw_line
        draw_line(pos, pos + vel.rotated(math.pi / 2).normalized() * 10,
                  width=1, color=path_color)


class DisabledNavigation(ConstAccelerationNavigation):
    _path_color = (50, 50, 255)
    helper_navigation = True

    def __init__(self, start_time: float, arrival_time: float, start_pos: Vector, start_vel: Vector, friction: float):
        super().__init__(
            Vector(0, 0),
            start_time, arrival_time,
            target=None,
            start_pos=start_pos, start_vel=start_vel,
            friction=friction
        )

    def debug_draw(self, current_time: float, world: World, physical: Goal | Seeker, index: int, steps: int
                   ) -> Color:
        path_color = super().debug_draw(current_time, world, physical, index, steps)

        from seekers.debug_drawing import draw_text, draw_circle

        if index == 0 and isinstance(physical, Seeker):
            end_pos = world.normalized_position(self.end_pos())
            counter = int((physical.disabled_counter / physical.config.seeker_disabled_time) * 9) + 1

            draw_circle(end_pos, physical.radius, color=self._path_color, width=1)
            draw_text(str(counter), end_pos, color=(255, 255, 255))

        return path_color


SolveManager = typing.Callable[[str, typing.Callable[[], None]], None]


class EntityFuture(abc.ABC):
    radius: float

    @abc.abstractmethod
    def planned_pos_at(self, time: float) -> Vector:
        ...

    @abc.abstractmethod
    def pos_at(self, time: float) -> Vector:
        ...

    @abc.abstractmethod
    def planned_vel_at(self, time: float) -> Vector:
        ...

    @abc.abstractmethod
    def vel_at(self, time: float) -> Vector:
        ...

    @abc.abstractmethod
    def planned_end_pos(self, default: Vector) -> Vector:
        ...

    @abc.abstractmethod
    def end_pos(self, default: Vector) -> Vector:
        ...

    @abc.abstractmethod
    def planned_end_vel(self, default: Vector) -> Vector:
        ...

    @abc.abstractmethod
    def end_vel(self, default: Vector) -> Vector:
        ...

    def debug_draw(self, current_time: float, world: World, physical: Physical, steps: int):
        ...


class PhysicalFuture(EntityFuture):
    radius: float

    def __init__(self, segments: list[Navigation] = None):
        self.segments: list[Navigation] = [] if segments is None else segments

    def get_segment(self, time: float) -> Navigation:
        for segment in self.segments:
            if time in segment:
                return segment

        raise FutureUncertainError(f"No segment found for time {time}.")

    def planned_pos_at(self, time: float) -> Vector:
        return self.get_segment(time).planned_pos_at(time)

    def pos_at(self, time: float) -> Vector:
        return self.get_segment(time).pos_at(time)

    def planned_vel_at(self, time: float) -> Vector:
        return self.get_segment(time).planned_vel_at(time)

    def vel_at(self, time: float) -> Vector:
        return self.get_segment(time).vel_at(time)

    def planned_end_pos(self, default: Vector) -> Vector:
        if not self.segments:
            return default

        return self.segments[-1].planned_end_pos()

    def end_pos(self, default: Vector) -> Vector:
        if not self.segments:
            return default

        return self.segments[-1].end_pos()

    def planned_end_vel(self, default: Vector) -> Vector:
        if not self.segments:
            return default

        return self.segments[-1].planned_end_vel()

    def end_vel(self, default: Vector) -> Vector:
        if not self.segments:
            return default

        return self.segments[-1].end_vel()

    def planned_end_time(self, default: float) -> float:
        if not self.segments:
            return default

        return self.segments[-1].arrival_time

    def debug_draw(self, current_time, world, physical, steps):
        for i, segment in enumerate(self.segments):
            segment.debug_draw(current_time, world, physical, i, steps)

    def get_acceleration(self, time: float) -> Vector:
        return self.get_segment(time).get_acceleration(time)


class GoalFuture(EntityFuture):
    def __init__(self):
        self.pos: Vector = Vector()
        self.vel: Vector = Vector()
        self.time: float = 0
        self.friction: float = 0.02
        self.radius: float = 6

    def update(self, goal: Goal, time: float):
        self.pos = goal.position
        self.vel = goal.velocity
        self.time = time
        self.radius = goal.radius

    def pos_at(self, time: float) -> Vector:
        return future_position(
            time=time - self.time,
            acceleration=Vector(0, 0),
            velocity=self.vel,
            position=self.pos,
            friction=self.friction
        )

    def planned_pos_at(self, time: float) -> Vector:
        return self.pos_at(time)

    def vel_at(self, time: float) -> Vector:
        return future_velocity(
            time=time - self.time,
            acceleration=Vector(0, 0),
            velocity=self.vel,
            friction=self.friction
        )

    def planned_vel_at(self, time: float) -> Vector:
        return self.vel_at(time)

    def end_pos(self, default: Vector = None) -> Vector:
        return terminal_position(
            acceleration=Vector(0, 0),
            velocity=self.vel,
            position=self.pos,
            friction=self.friction
        )

    def planned_end_pos(self, default: Vector) -> Vector:
        return self.end_pos(default)

    def end_vel(self, default: Vector) -> Vector:
        return Vector(0, 0)

    def planned_end_vel(self, default: Vector) -> Vector:
        return self.end_vel(default)

    def debug_draw(self, current_time: float, world: World, physical: Goal, steps: int = 15):
        from seekers.debug_drawing import draw_circle, draw_line

        end_pos = self.end_pos()
        draw_circle(world.normalized_position(end_pos), physical.radius, color=(100, 100, 100), width=1)
        draw_line(physical.position, end_pos, color=(50, 50, 50), width=1)


@dataclasses.dataclass
class SeekerInfo:
    position: Vector
    velocity: Vector
    friction: float
    time: float


class SeekerFuture(PhysicalFuture):
    def __init__(self, targets: list[Target] = None, plan_limit: int = 4):
        super().__init__()

        self.future_targets: list[Target] = [] if targets is None else targets
        self.plan_limit = plan_limit
        self._last_disabled: int = 0
        self.radius: float = 10

    def planned_targets(self, start_i: int = 0) -> typing.Iterator[Target]:
        for nav in self.segments[start_i:]:
            if nav.helper_navigation:
                continue

            yield nav.target

    def plan_target(self, seeker: Seeker, current_time: float, world: World, solve_manager: SolveManager):
        if not self.future_targets:
            return

        def solve():
            if len(self.segments) >= self.plan_limit:
                return

            target = self.future_targets.pop(0)
            assert target.vel is None, "Velocity targets not supported yet."

            nav = ConstAccelerationNavigation.solve_from_target(
                target.pos,
                world.normalized_position(self.end_pos(default=seeker.position)),
                self.end_vel(default=seeker.velocity),
                seeker.config.physical_friction,
                get_thrust(seeker),
                world,
                int(self.planned_end_time(default=current_time))
            )

            self.segments.append(nav)

        if not self.segments:
            solve()
        else:
            solve_manager(seeker.id, solve)

    def invalidate_segments(self, i: int = 0):
        self.future_targets = [*self.planned_targets(i), *self.future_targets]
        del self.segments[i:]

    def check_segments(self, seeker: Seeker, current_time: float, world: World):
        for i, nav in enumerate(self.segments):
            err2 = nav.pos_error2(seeker if i == 0 else None, current_time, world)
            max_err = get_target_accuracy(nav.arrival_time - current_time, seeker.radius)

            if err2 > max_err ** 2:
                self.invalidate_segments(i)
                break

    def update_seeker_info(self, seeker: Seeker, current_time: float):
        pos = seeker.position
        vel = seeker.velocity
        time = current_time

        for segment in self.segments:
            segment.last_seeker_info = SeekerInfo(
                position=pos,
                velocity=vel,
                friction=seeker.config.physical_friction,
                time=time
            )

            pos = segment.end_pos()
            vel = segment.end_vel()
            time = segment.arrival_time

    def update(self, seeker: Seeker, current_time: float, world: World, solve_manager: SolveManager):

        if seeker.disabled_counter > self._last_disabled:
            self.invalidate_segments()
            self.segments: list[Navigation] = [
                DisabledNavigation(current_time, current_time + seeker.disabled_counter,
                                   seeker.position, seeker.velocity, seeker.config.physical_friction)
            ]

        # remove segment if it is finished
        if self.segments and self.segments[0].arrival_time < current_time:
            self.segments.pop(0)

        self.check_segments(seeker, current_time, world)

        self.update_seeker_info(seeker, current_time)
        for _ in range(self.plan_limit - len(self.segments)):
            self.plan_target(seeker, current_time, world, solve_manager)
            self.update_seeker_info(seeker, current_time)

        self._last_disabled = seeker.disabled_counter
        self.radius = seeker.radius


def get_thrust(seeker: Seeker, magnet: float | None = None) -> float:
    a = seeker.config.physical_friction * seeker.config.physical_max_speed

    if None is not magnet is True:
        a *= seeker.config.seeker_magnet_slowdown
    else:
        if seeker.magnet.is_on():
            a *= seeker.config.seeker_magnet_slowdown

    return a


class FutureUncertainError(Exception): ...


class Agent:
    def __init__(self, plan_limit: int = 4):
        self.future = SeekerFuture(plan_limit=plan_limit)
        self.acceleration_error = Vector()

    def update_navigation(self, seeker: Seeker, world: World, current_time: float, solve_manager: SolveManager):
        # update navigation
        self.future.update(seeker, current_time, world, solve_manager)

    def update_seeker_target(self, seeker: Seeker, current_time: float):
        # set seeker target
        # a, self.acceleration_error = balance_a_with_error2(self.navigation.direction, self.acceleration_error)
        if not self.future.segments:
            if current_time > 20:
                logger.error("No navigation!")
            return

        a = self.future.get_acceleration(current_time)
        seeker.target = seeker.position + a * 20
        # seeker.target = self.target

    def debug_draw(self, seeker: Seeker, current_time: float, world: World, steps=20):
        self.future.debug_draw(current_time, world, seeker, steps)


T = typing.TypeVar("T")


def _get_close_indices(a: list[T],
                       threshold: float,
                       value_key: typing.Callable[[T], float] = lambda x: x[0],
                       index_key: typing.Callable[[T], int] = lambda x: x[1]
                       ) -> list[tuple[int, int]]:
    out = []

    indices = list(range(len(a)))
    indices.sort(key=lambda i: value_key(a[i]))

    for i in range(len(indices) - 1):
        j = indices[i]
        j1 = indices[i + 1]

        cur = value_key(a[j])
        nxt = value_key(a[j1])

        if nxt - cur < threshold:
            index1 = index_key(a[j])
            index2 = index_key(a[j1])

            if index1 > index2:
                out.append((index2, index1))
            else:
                out.append((index1, index2))

    return out


def get_raw_collisions(futures: typing.Iterable[EntityFuture], current_time: float, world: World, config: Config,
                       max_time) -> dict[tuple[int, int], float]:
    out = {}

    max_r = max(future.radius for future in futures)

    max_speed = max_velocity(config.physical_max_speed + config.physical_friction, config.physical_friction)
    diameter_time = 2 * config.seeker_radius / max_speed
    timestep = int(diameter_time)

    disabled_entities: set[int] = set()
    dt = 0

    while dt < max_time:
        positions_x: list[tuple[float, int]] = []
        positions_y: list[tuple[float, int]] = []

        for i, future in enumerate(futures):
            if i in disabled_entities:
                continue

            try:
                pos = future.pos_at(current_time + dt)
                world.normalize_position(pos)
            except FutureUncertainError:
                disabled_entities.add(i)
                continue

            positions_x.append((pos.x, i))
            positions_y.append((pos.y, i))

        x_collisions = set(_get_close_indices(positions_x, max_r * 3.5))
        y_collisions = set(_get_close_indices(positions_y, max_r * 3.5))

        collisions = x_collisions & y_collisions

        for a, b in collisions:
            if {a, b} & disabled_entities:
                continue

            out |= {(a, b): current_time + dt}

        dt += timestep

    return out


def refine_collision(future1: EntityFuture, future2: EntityFuture, col_time: float, world: World,
                     current_time: float, t_range: int = 40) -> float:
    t1 = int(max(current_time, col_time - t_range))
    t2 = int(col_time + t_range)

    for t in range(t1, t2):
        try:
            pos1 = world.normalized_position(future1.pos_at(t))
            pos2 = world.normalized_position(future2.pos_at(t))
        except FutureUncertainError:
            continue

        if (pos2 - pos1).squared_length() < ((future1.radius + future2.radius) * 1.2) ** 2:
            return t


def get_collisions(futures: typing.Sequence[EntityFuture], current_time: float, world: World, config: Config,
                   max_time) -> dict[tuple[int, int], float]:
    raw_collisions = get_raw_collisions(futures, current_time, world, config, max_time)

    out = {}
    for (a, b), time in raw_collisions.items():
        future1, future2 = futures[a], futures[b]

        time = refine_collision(future1, future2, time, world, current_time)

        if time is None:
            continue

        out |= {(a, b): time}

    return out


class GameStrategy:
    def __init__(self, plan_limit: int = 4):
        self.plan_limit = plan_limit
        self.agents: list[Agent] = []
        self.goal_futures: dict[str, GoalFuture] = {}
        self.last_collision_time = -1
        self.last_want_solves = -1

    def update(self, my_seekers: list[Seeker]):
        while len(self.agents) < len(my_seekers):
            self.agents.append(Agent(plan_limit=self.plan_limit))

    def update_navigation(self, my_seekers: list[Seeker], world: World, current_time: float):
        want_solves = defaultdict(list)

        def solve_mgr(id_: str, solve):
            want_solves[id_].append(solve)

        for seeker, agent in zip(my_seekers, self.agents):
            agent.update_navigation(seeker, world, current_time, solve_mgr)

        if want_solves:
            key = random.choice(list(want_solves.keys()))
            want_solves[key][0]()

        for seeker, agent in zip(my_seekers, self.agents):
            agent.update_seeker_target(seeker, current_time)

        self.last_want_solves = sum([len(solves) for solves in want_solves.values()])

    def update_goals(self, goals: list[Goal], current_time: float):
        for goal in goals:
            if goal.id not in self.goal_futures:
                self.goal_futures[goal.id] = GoalFuture()

            self.goal_futures[goal.id].update(goal, current_time)

    def debug_draw(self, my_seekers: list[Seeker], goals: list[Goal], world: World, current_time: float):
        for agent, seeker in zip(self.agents, my_seekers):
            agent.debug_draw(seeker, current_time, world)

        for future, goal in zip(self.goal_futures.values(), goals):
            future.debug_draw(current_time, world, goal)

    def get_collisions(self, world: World, current_time: float, config: Config, max_time: int = 200
                       ) -> tuple[list[EntityFuture], dict[tuple[int, int], float]]:
        t1 = time_module.perf_counter()

        # noinspection PyTypeChecker
        entity_futures: list[EntityFuture] = (
                [agent.future for agent in self.agents]
                + [f for f in self.goal_futures.values()]
        )
        collisions = get_collisions(entity_futures, current_time, world, config, max_time=max_time)

        t2 = time_module.perf_counter()
        self.last_collision_time = t2 - t1

        return entity_futures, collisions
