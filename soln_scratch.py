import matplotlib
import sympy as sp

t = sp.symbols("t", nonnegative=True, real=True)
f = sp.symbols("f", positive=True, real=True)
a, p, v_0 = sp.symbols("a p v_0", real=True)
angle = sp.Symbol("α", real=True)

c = 1 - f

v_of_t: sp.Expr = v_0 * c ** t + a * (c ** t - 1) / (c - 1)
# p_of_t: sp.Expr = (a * (c ** t - t * sp.log(c) - 1) - f * v_0 * (c ** t - 1)) / (-f * sp.log(c))
# p_of_t2: sp.Expr = (a * (c ** (t + 1) - c * (t + 1) + t) + (c - 1) * v_0 * (c ** (t + 1) - 1)) / (c - 1) ** 2
p_of_t2 = sp.summation(v_of_t, (t, 1, t)).simplify()

# wolframalpha: sum 1->t
p_of_t3: sp.Basic = (a * (c ** (t + 1) - c * (t + 1) + t) + (c - 1) * c * v_0 * (c ** t - 1)) / (c - 1) ** 2

print(f"{p_of_t2 = }")
print("for a = 0:", sp.mathematica_code(p_of_t3.subs(a, 0)))

# COMPARISON p_of_t, p_of_t2
# sp.plot(((p_of_t / p_of_t2).subs({f: 0.02, v_0: 0, a: 0.1}), (t, 0, 5)),
#         ((p_of_t / p_of_t2).subs({f: 0.02, v_0: 2, a: 0.1}), (t, 0, 5)),
#         ((p_of_t / p_of_t2).subs({f: 0.02, v_0: -1, a: 0.1}), (t, 0, 5)))
# sp.plot((p_of_t.subs({f: 0.02, v_0: 0, a: 0.1}), (t, 0, 50)),
#         (p_of_t2.subs({f: 0.02, v_0: 0, a: 0.1}), (t, 0, 50)))

# T_OF_P
# t_of_p = sp.solve(sp.Eq(p_of_t3, p), t, dict=True)  # no soln
# print(t_of_p)

# t_of_p = (
#         sp.log(
#             -a * sp.LambertW(-(a - f * v_0) * sp.exp((-a + f * p * sp.log(1 - f) + f * v_0) / a) / a) / (a - f * v_0)
#         ) / sp.log(1 - f)
# )

# sum 0->t
# https://www.wolframalpha.com/input?i2d=true&i=solve+Divide%5B%5C%2840%29a+%5C%2840%29Power%5Bc%2C%5C%2840%29t+%2B+1%5C%2841%29%5D+-+c+%5C%2840%29t+%2B+1%5C%2841%29+%2B+t%5C%2841%29+%2B+%5C%2840%29c+-+1%5C%2841%29+v_0+%5C%2840%29Power%5Bc%2C%5C%2840%29t+%2B+1%5C%2841%29%5D+-+1%5C%2841%29%5C%2841%29%2CPower%5B%5C%2840%29c+-+1%5C%2841%29%2C2%5D%5D+for+t
# t_of_p2 = (
#         (a * sp.LambertW(-(c ** (-(v_0 * c) / (a * (c - 1)) - c / (c - 1) + v_0 / (a * (c - 1)) + 1) * sp.log(c) * (
#                 a + c * v_0 - v_0)) / (a * (c - 1))) - a * c * sp.LambertW(-(
#                 c ** (-(v_0 * c) / (a * (c - 1)) - c / (c - 1) + v_0 / (a * (c - 1)) + 1) * sp.log(c) * (
#                 a + c * v_0 - v_0)) / (a * (c - 1))) - a * c * sp.log(c) - c * v_0 * sp.log(c) + v_0 * sp.log(
#             c)) / (a * (c - 1) * sp.log(c))
# )

# sum 1->t
t_of_p2 = (
        -sp.LambertW(
            (sp.log(1 - f) * (a - f * v_0) * (1 - f) ** ((f * p) / a + ((f - 1) * v_0) / a + 1 / f)) / (a * f)
        ) / sp.log(1 - f) + (f * p) / a + ((f - 1) * v_0) / a + 1 / f - 1
)

# t_of_p2 = (
#     -sp.LambertW(-(sp.log(c) * c ** (1 / (1 - c) - v_0 / a) * (a + (c - 1) * v_0)) / (a * (c - 1))) / sp.log(c) - v_0 / a - c / (c - 1)
# )

# t_of_p_angle = t_of_p2.subs(a, a * sp.cos(angle))
#
# sp.plot((t_of_p_angle.subs({f: 0.02, v_0: 0, a: 0.1, angle: 0}), (p, 0, 100)),
#         (t_of_p_angle.subs({f: 0.02, v_0: 0, a: 0.1, angle: sp.pi / 6}), (p, 0, 100)),
#         (t_of_p_angle.subs({f: 0.02, v_0: 0, a: 0.1, angle: sp.pi / 3}), (p, 0, 100)))
#
# t_of_p_l = sp.lambdify((a, p, v_0, f), t_of_p2)
#
# print(t_of_p_l(0.1, 10, 0, 0.02))

# MAX_V

# max_speed1d = -a / (f - 1)
# max_speed2d = max_speed1d.subs(a, a * sp.cos(angle))
#
# sp.plot(max_speed2d.subs({f: 0.02, a: 0.1}), (angle, 0, sp.pi))

# MAX ANGLE VELOCITY

# direction_change_t = sp.solve(sp.Eq(v_of_t, 0), t, dict=True)[0][t]
# print("When v_0 = 0, t =", direction_change_t)

# acc = v_of_t.diff(t)
# print("Change in velocity:", acc)
#
# start_angle = sp.Symbol("start_angle")
#
# v_0_x, v_0_y = sp.symbols("v_x v_y", real=True)
# a_x, a_y = sp.symbols("a_x a_y", real=True)
#
# v_of_t_x = v_of_t.subs({v_0: v_0_x, a: a_x})
# v_of_t_y = v_of_t.subs({v_0: v_0_y, a: a_y})
#
# angle_of_t = sp.atan2(v_of_t_y, v_of_t_x)
# anglev_of_t = angle_of_t.diff(t)
# angleacc_of_t = anglev_of_t.diff(t)
#
# print("Anglev of t:", anglev_of_t.simplify())
# print("Change in anglev:", angleacc_of_t.simplify())
#
# start_angle_val = sp.pi / 1.5
# v_0_val = 1
# v_0_x_val = v_0_val * sp.cos(start_angle_val)
# v_0_y_val = v_0_val * sp.sin(start_angle_val)
# end_angle_val = 0
# a_val = 0.1
# a_x_val = a_val * sp.cos(end_angle_val)
# a_y_val = a_val * sp.sin(end_angle_val)
#
# sp.plot((angle_of_t.subs({f: 0.02, v_0_x: v_0_x_val, v_0_y: v_0_y_val, a_x: a_x_val, a_y: a_y_val}),
#          (t, 0, 40)),
#         (anglev_of_t.subs({f: 0.02, v_0_x: v_0_x_val, v_0_y: v_0_y_val, a_x: a_x_val, a_y: a_y_val}),
#          (t, 0, 40)),
#         (angleacc_of_t.subs({f: 0.02, v_0_x: v_0_x_val, v_0_y: v_0_y_val, a_x: a_x_val, a_y: a_y_val}),
#          (t, 0, 40)))
#
# # anglev_max = sp.solve(sp.Eq(angleacc_of_t, 0), t, dict=True)[0][t]  # takes a while, solns below
#
# anglev_max = sp.log(
#     -sp.sqrt((a_x ** 2 + a_y ** 2) / (
#             a_x ** 2 - 2 * a_x * f * v_0_x + a_y ** 2 - 2 * a_y * f * v_0_y + f ** 2 * v_0_x ** 2 + f ** 2 * v_0_y ** 2))
# ) / sp.log(1 - f)
#
# anglev_max2 = sp.log(
#     (a_x ** 2 + a_y ** 2) / (
#             a_x ** 2 - 2 * a_x * f * v_0_x + a_y ** 2 - 2 * a_y * f * v_0_y + f ** 2 * v_0_x ** 2 + f ** 2 * v_0_y ** 2)
# ) / (2 * sp.log(1 - f))
#
# print("Max angle change:", anglev_max2)
# print("Max angle change:",
#       anglev_max2.subs({f: 0.02, v_0_x: v_0_x_val, v_0_y: v_0_y_val, a_x: a_x_val, a_y: a_y_val})
#       .evalf()
#       )

#
# p_x, p_y = sp.symbols("p_x p_y", real=True)
#
# t_of_p_x = t_of_p2.subs({v_0: v_0_x, a: sp.cos(angle) * a, p: p_x})
# t_of_p_y = t_of_p2.subs({v_0: v_0_y, a: sp.sin(angle) * a, p: p_y})

# exprs = [
#     t_of_p_x.subs({f: 0.02, a: 0.1, v_0_x: 0.1, v_0_y: 0, p_x: 10, p_y: 5}),
#     t_of_p_y.subs({f: 0.02, a: 0.1, v_0_x: 0.1, v_0_y: 0, p_x: 10, p_y: 5}),
# ]
#
# from sympy.plotting.pygletplot import PygletPlot as Plot
#
# plt = Plot(*exprs)
# plt.show()

#
# q = sp.Eq(t_of_p_x, t_of_p_y)
#
# print(sp.mathematica_code(q))
# sol = sp.solve(q, angle, dict=True)
#
# print(sol)

# WE DONT NEED ONE VELOCITY COMPONENT
# t_of_p2_v0 = t_of_p2.subs({v_0: 0}).simplify()
# print("t_of_p with v=0:", t_of_p2_v0.subs({f: 0.02, a: 0.1, p: 1}))
# sp.plot(t_of_p2_v0.subs({f: 0.02, a: 0.1}), (p, 0, 100))

# A OF P, T
a_of_p_t = -(
        f * (-f * v_0 * (1 - f) ** t + v_0 * (1 - f) ** t + f * v_0 + f * p - v_0)
) / (f * (1 - f) ** t - (1 - f) ** t - f * t - f + 1)
print("a_of_p_t", a_of_p_t)
print("a_of_p_t", a_of_p_t.simplify())
print("a_of_p_t, v=0:", a_of_p_t.subs({v_0: 0}))
print("a_of_p_t, v=0:", a_of_p_t.subs({v_0: 0}).simplify())

# POS OF T (WITH V_0 = 0)
print("Pos of t with v_0 = 0:", p_of_t2.subs({v_0: 0}).simplify())
print("Pos of t with v_0 = 0:", p_of_t3.subs({v_0: 0}).simplify())
