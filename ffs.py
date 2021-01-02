import pandas as pd
import numpy as np

'''
Solving the MDP problem of New York Yellow taxi data using feature-based RL 
LSTD and Fast Feature Selection as features
'''

"""
class state containing information of each state coordinates
in the map, states that it goes to, and rewards it gains going
to those states
"""


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.to_states = dict()
        self.state_r = dict()

    def add_state(self, state):
        """
        Sum the total number of transitions from
        current state to this state
        """
        if state in self.to_states.keys():
            self.to_states[state] += 1
        else:
            self.to_states[state] = 1

    def add_r(self, rew, state):
        """
        Save the reward gaining from the transition
        """
        if state not in self.state_r.keys():
            self.state_r[state] = rew


# Remove the outliers to better discretize the map
def remove_outlier(df, n_col):
    q_low_pl = df[n_col].quantile(0.01)
    q_hi_pl = df[n_col].quantile(0.99)
    df = df[(df[n_col] < q_hi_pl) & (df[n_col] > q_low_pl)]
    return df


def create_transition_reward():
    taxi_data = pd.read_csv("yellow_tripdata_2016-01.csv")
    taxi_data_sub = taxi_data.loc[taxi_data['VendorID'] == 2]

    # Removing outliers

    taxi_data_sub = remove_outlier(taxi_data_sub, "pickup_longitude")
    taxi_data_sub = remove_outlier(taxi_data_sub, "pickup_latitude")
    taxi_data_sub = remove_outlier(taxi_data_sub, "dropoff_longitude")
    taxi_data_sub = remove_outlier(taxi_data_sub, "dropoff_latitude")

    taxi_matrix = np.array(taxi_data_sub)

    # discretize based on longtitude and latitude

    long = np.hstack((taxi_matrix[:, 5], taxi_matrix[:, 9]))
    lat = np.hstack((taxi_matrix[:, 6], taxi_matrix[:, 10]))

    min_long = np.min(long)
    max_long = np.max(long)

    min_lat = np.min(lat)
    max_lat = np.max(lat)

    # discretize the map of transitions into a 50 x 50 cells
    n = 50

    # Number of actions indicates the available directions
    n_actions = 4

    diff_long = (max_long - min_long) / n
    diff_lat = (max_lat - min_lat) / n

    # create a grid consisting of None values
    ls2d = None

    for i in range(n):
        ls = list()
        for j in range(n):
            ls.append(None)
        if ls2d is None:
            ls2d = ls
        else:
            ls2d = np.vstack((ls2d, ls))

    grid = ls2d

    # check that state coordinates does not fall out of bound
    assign = lambda x: n - 1 if x > n - 1 else x

    for row in taxi_matrix:

        # p_i and p_j are the coordinates of the start (from) state
        p_i = assign(int((row[5] - min_long) / diff_long))
        p_j = assign(int((row[6] - min_lat) / diff_lat))

        # assign the start state, if there is no state assigned to this cell
        if grid[p_i][p_j] is None:
            grid[p_i][p_j] = State(p_j, p_j)

        # d_i and d_j are the coordinates of the end (to) state
        d_i = assign(int((row[9] - min_long) / diff_long))
        d_j = assign(int((row[10] - min_lat) / diff_lat))

        # assign the start state, if there is no state assigned to this cell
        if grid[d_i][d_j] is None:
            grid[d_i][d_j] = State(d_i, d_j)

        # add end state to the states that start state transit to
        grid[p_i][p_j].add_state(grid[d_i][d_j])

        # add the reward gaining from this transition
        grid[p_i][p_j].add_r(row[18], grid[d_i][d_j])

    grid_final = []
    n_states = 0

    # remove None cells
    for row in grid:
        rem_none = [elem for elem in row if elem is not None]
        ln = len(rem_none)
        n_states += ln
        if ln:
            grid_final.append(rem_none)

    state_list = list()

    # create a final list of all states
    for row in grid_final:
        for elem in row:
            state_list.append(elem)

    # constructing transition and reward matrices from the states
    trn_mtx_a = np.zeros((n_actions, n_states, n_states))
    reward = np.zeros((n_states, n_actions))

    for i, s in enumerate(state_list):

        # states to the right of the current state
        state_to_ls_r = dict()
        # states to the left of the current state
        state_to_ls_l = dict()
        # states above of the current state
        state_to_ls_u = dict()
        # states down of the current state
        state_to_ls_d = dict()
        p_i, p_j = s.x, s.y

        for sub_s, value in s.to_states.items():

            d_i, d_j = sub_s.x, sub_s.y

            ind = state_list.index(sub_s)
            r = s.state_r[sub_s]

            # if the end state can be reached by both right and down actions, randomly add to one
            if d_i > p_i and d_j > p_j:

                num = np.random.randint(0, 2)
                if num == 0:
                    state_to_ls_r[ind] = (value, r)

                else:
                    state_to_ls_d[ind] = (value, r)

            # if the end state can be reached by both right and up actions, randomly add to one
            elif d_i > p_i and d_j < p_j:

                num = np.random.randint(0, 2)
                if num == 0:
                    state_to_ls_r[ind] = (value, r)
                else:
                    state_to_ls_u[ind] = (value, r)

            elif d_i > p_i and d_j == p_j:

                state_to_ls_r[ind] = (value, r)

            # if the end state can be reached by both left and up actions, randomly add to one
            elif d_i < p_i and d_j < p_j:

                num = np.random.randint(0, 2)
                if num == 0:
                    state_to_ls_l[ind] = (value, r)
                else:
                    state_to_ls_u[ind] = (value, r)

            # if the end state can be reached by both left and down actions, randomly add to one
            elif d_i < p_i and d_j > p_j:

                num = np.random.randint(0, 2)
                if num == 0:
                    state_to_ls_l[ind] = (value, r)
                else:
                    state_to_ls_u[ind] = (value, r)

            elif d_i < p_i and d_j == p_j:

                state_to_ls_l[ind] = (value, r)

            elif d_i == p_i and d_j > p_j:

                state_to_ls_d[ind] = (value, r)

            elif d_i == p_i and d_j < p_j:

                state_to_ls_u[ind] = (value, r)

        total_transits = sum([item[0] for item in state_to_ls_u.values()])

        for key, value in state_to_ls_u.items():
            trn_mtx_a[0, i, key] = value[0] / total_transits
            reward[i, 0] += trn_mtx_a[0, i, key] * value[1]

        total_transits = sum([item[0] for item in state_to_ls_d.values()])

        for key, value in state_to_ls_d.items():
            trn_mtx_a[1, i, key] = value[0] / total_transits
            reward[i, 1] += trn_mtx_a[1, i, key] * value[1]

        total_transits = sum([item[0] for item in state_to_ls_r.values()])

        for key, value in state_to_ls_r.items():
            trn_mtx_a[2, i, key] = value[0] / total_transits
            reward[i, 2] += trn_mtx_a[2, i, key] * value[1]

        total_transits = sum([item[0] for item in state_to_ls_l.values()])

        for key, value in state_to_ls_l.items():
            trn_mtx_a[3, i, key] = value[0] / total_transits
            reward[i, 3] += trn_mtx_a[3, i, key] * value[1]

    return trn_mtx_a, reward, n_states, n_actions


trn_mtx_a, reward, n_states, n_actions = create_transition_reward()


# extracting features using Fast Feature Selection (FFS)
def TFFS(k):
    phi = np.zeros((n_states, n_actions, k))
    for a in range(n_actions):
        u, s, vt = np.linalg.svd(trn_mtx_a[a, :, :])
        sigma = np.zeros((n_states, n_states))
        sigma[:, :] = np.diag(s)
        u1, s1, vh1 = u[:, :k], sigma[:k, :], vt
        phi[:, a, :] = u1
    return phi


# Evaluate policy
def eval_policy_lstd(phi, w):
    V = np.zeros(n_states)

    for s in range(n_states):

        v = 0

        for a in range(n_actions):
            v += w.T @ phi[s][a]

        V[s] = v

    return np.array(V)


# Implementing LSTD to solve MDP for feature based RL
def lstd(phi, p, r, k):
    gamma = 1
    B = np.eye(k) * 0.1
    b = np.zeros(k)
    w = np.ones(k) / k
    threshold = 0.001
    prev_w = np.array([1000 for _ in range(k)])
    max_iter = 0

    while True:

        V = eval_policy_lstd(phi, w)

        for s in range(n_states):

            q_hat = np.zeros(n_actions)

            for a in range(n_actions):

                inds = [i for i, elem in enumerate(p[s, a, :]) if elem != 0]

                for s_p in inds:

                    for a_p in range(n_actions):
                        q_hat[a_p] = w.T.dot(phi[s_p][a_p])

                    best_a = np.argmax(q_hat)

                    for i in range(k):
                        for j in range(k):
                            B[i][j] += phi[s][a][i] * (phi[s][a][j] - gamma * phi[s_p][best_a][j])

                b += r[s][a] * phi[s, a]

        max_iter += 1

        w = np.linalg.pinv(B).dot(b)
        print(max_iter)

        if np.sum(prev_w - w) >= threshold and max_iter < 10:
            prev_w = w
        else:
            return V


# considering 27 features
k = 27
Phi = TFFS(k)
V = lstd(Phi, trn_mtx_a.reshape((trn_mtx_a.shape[1], trn_mtx_a.shape[0], trn_mtx_a.shape[2])), reward, k)
print(sum(V) / len(V))
