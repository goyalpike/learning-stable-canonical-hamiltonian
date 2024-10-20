import numpy.linalg as la
import numpy as np

"""symplectic implicit mid-point integrator
   https://na.uni-tuebingen.de/~lubich/chap6.pdf"""


def Mid_point(rhs, jac, initial_cond, t):
    tol = 1e-12
    dt = t[1] - t[0]
    z0 = initial_cond.reshape(-1, 1)
    ldim = z0.shape[0]

    res = lambda z1, zm, z0: z1 - z0 - dt * rhs([], zm).reshape(-1, 1)
    Jac = lambda zm: np.eye(ldim) - 0.5 * dt * jac(zm)

    z = np.zeros((ldim, t.size))

    z[:, 0] = z0.reshape(
        -1,
    )

    for i in range(t.size - 1):
        z1 = z0
        zm = 0.5 * (z1 + z0)
        err = la.norm(res(z1, zm, z0))
        c = 0
        while err > tol:
            z1 = z1 - la.solve(Jac(zm).squeeze(), res(z1, zm, z0))
            zm = 0.5 * (z1 + z0)
            err = la.norm(res(z1, zm, z0))
            c = c + 1
            if c > 10:
                print(
                    "err =",
                    err,
                    ",Nr. of Newton iter=",
                    c,
                    ", Finished ",
                    round(100 * (i + 1) / (t.size - 1), 2),
                    "percent",
                )
                break
        z0 = z1
        z[:, i + 1] = z1.reshape(
            -1,
        )
    return z
