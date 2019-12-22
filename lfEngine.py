import numpy as np
import matplotlib.pyplot as plt

# global vars
G = 6.674e-11


class Body:
    def __init__(self, r, v, m):
        self.r = np.array(r, dtype=np.complex64)
        self.v = np.array(v, dtype=np.complex64)
        self.m = m


def gen_force_mat(_arr_bodies, epsilon):
    global G

    vec_f = np.zeros([len(_arr_bodies), len(_arr_bodies), len(_arr_bodies[0].r)], dtype=np.float64)
    for i in range(len(_arr_bodies)):
        for j in range(i+1, len(_arr_bodies)):
            double_mass = _arr_bodies[i].m * _arr_bodies[j].m
            vec_sep = np.array(_arr_bodies[i].r - _arr_bodies[j].r)
            norm = np.linalg.norm(vec_sep)
            vec_f[i, j, :] = np.real(double_mass * vec_sep / (norm**2 + epsilon**2)**(3/2))
            vec_f[i, j, :] = - G * vec_f[i, j, :]

    vec_f_trans = np.transpose(vec_f, (1, 0, 2))
    return vec_f - vec_f_trans


def get_potential(_arr_bodies):
    global G

    u = 0
    for i in range(len(_arr_bodies)-1):
        for j in range(i+1, len(_arr_bodies)):
            vec_sep = np.array(_arr_bodies[i].r - _arr_bodies[j].r)
            norm = np.linalg.norm(vec_sep)
            u += G*_arr_bodies[i].m*_arr_bodies[j].m / norm
    return -u


def half_step(arr_bodies_, dt_, softening=0):
    dt_2_ = dt_/2
    mat_force_ = gen_force_mat(arr_bodies_, softening)
    for j in range(len(arr_bodies_)):
        vec_sum_f_ = np.sum(mat_force_[j, :, :], axis=0)  # Net force on ith particle
        arr_bodies_[j].v -= vec_sum_f_ * dt_2_ / arr_bodies_[j].m
    return arr_bodies_


def execute(arr_bodies, dt, n_iter, softening=0):
    dt_2 = dt/2

    v_t = -dt_2
    r_t = 0

    v_ts = np.empty([int(n_iter)])
    r_ts = np.empty([int(n_iter)])
    e_ts = np.empty([int(n_iter)])
    rs = np.empty([len(arr_bodies), int(n_iter), 3])
    vs = np.empty([len(arr_bodies), int(n_iter), 3])
    Eks = np.empty([int(n_iter)])
    pot = np.empty([int(n_iter)])

    for i in range(int(n_iter)):
        v_t += dt
        r_t += dt

        mat_force = gen_force_mat(arr_bodies, softening)

        Ek = 0

        for j in range(len(arr_bodies)):
            vec_sum_f = np.sum(mat_force[j, :, :], axis=0) # Net force on ith particle

            arr_bodies[j].v += vec_sum_f * dt / arr_bodies[j].m
            arr_bodies[j].r += arr_bodies[j].v * dt

            tmp_v = arr_bodies[j].v + vec_sum_f * dt_2 / arr_bodies[j].m
            Ek += (1/2) * arr_bodies[j].m * np.linalg.norm(tmp_v)**2

            rs[j, i, :] = arr_bodies[j].r
            vs[j, i, :] = arr_bodies[j].v

        Eks[i] = Ek
        pot[i] = get_potential(arr_bodies)

        r_ts[i] = r_t
        v_ts[i] = v_t

    #for i in range(len(arr_bodies)):
    #    plt.plot(rs[i, :, 1], rs[i, :, 0])
    out = {"v_ts": v_ts, "r_ts": r_ts, "vs": vs, "rs": rs, "Eks": Eks, "pot": pot}
    return out
