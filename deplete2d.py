# Main simulation script. Let's just whack it all in and see what happens

# New and improved version, using KDTree to speedup the search process
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

class sim2d:

    def __init__(self, mu=100, r=0.01, v=1, plot=False):
        self.mu = mu
        self.r = r
        self.v = v
        self.plot = plot

        # Generate prey
        self.n = np.random.poisson(self.mu)
        self.x_prey = np.random.rand(self.n, 2)
        self.x_prey_orig = self.x_prey.copy()

        # Initialise arrays
        self.t = 0
        self.n_eaten = 0
        self.popn = self.n
        self.t_eat = np.zeros(self.n + 1)
        self.u = np.zeros(self.n + 1)
        self.u[0] = self.n
        self.i_eat = np.zeros(self.n)
        self.alive = np.array([True for i in range(self.n)])
        self.pointer = 1

        self.x_pred = np.array([0.5, 0.5])
        self.tree = scipy.spatial.cKDTree(self.x_prey)
        if self.plot:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.markers_alive = self.ax.plot(self.x_prey[self.alive,0], 
                    self.x_prey[self.alive,1], 'bo')
            self.marker_pred = self.ax.plot(self.x_pred[0], self.x_pred[1], 'ko')
            self.fig.canvas.draw()

    def step(self, u):
        # If step would take us outside of the boundary, recenter and
        # recompute KDTree. We recenter so that the predator ends up at (0.5, 0.5) at the end of his step
        if ((self.x_pred + u < self.r) | (self.x_pred + u > 1 - self.r)).any():
            dx = np.array([0.5, 0.5]) - self.x_pred - u
            self.x_prey = np.mod(self.x_prey + dx, 1)
            self.x_pred = self.x_pred + dx
            self.tree = scipy.spatial.cKDTree(self.x_prey)
            
            
        # find candidate points for closeness from KDTree - using radius of l + r from start point. is_candidate will be a Boolean array of points
        # that are in the neighbourhood and are alive
        l = np.linalg.norm(u)
        is_candidate = np.zeros(self.n, dtype=bool)
        is_candidate[self.tree.query_ball_point(self.x_pred, l+self.r)] = True
        is_candidate = (is_candidate & self.alive).nonzero()[0]
        
        z = self.x_prey[is_candidate]
        idx, d = self.find_prey(z, u)
        n_eaten = idx[0].size
        self.popn = self.popn - n_eaten
        self.alive[is_candidate[idx]] = False
        self.x_pred = self.x_pred + u
        
        # Update the arrays
        # print d
        self.t_eat[self.pointer:(self.pointer + n_eaten)] = d / self.v + self.t
        self.u[self.pointer:(self.pointer + n_eaten)] = self.u[self.pointer-1] + np.arange(-1,-n_eaten-1,-1)
        self.pointer = self.pointer + n_eaten
        self.t = self.t + l / self.v
        
        
        if self.plot:        
            plt.setp(self.markers_alive, data=(self.x_prey[self.alive,0], self.x_prey[self.alive, 1]))
            plt.setp(self.marker_pred, data=(self.x_pred[0], self.x_pred[1]))
            self.fig.canvas.draw()

        # print self.popn, np.sum(self.alive), self.x_pred
         

    def find_prey(self, z, u):
        # Form projections onto u and u_perp by rotating candidate points. The predator position used is before the step
        l = np.linalg.norm(u)
        Q = 1/l * np.array([[u[0], -u[1]], [u[1], u[0]]])
        z = np.dot(z - self.x_pred, Q)
        #plt.figure()
        #plt.plot(z[:,0], z[:,1], 'ro')
        #plt.show()

        # check to see which ones satisfy constraints
        idx = (np.linalg.norm(z, axis=1) < self.r) | (np.linalg.norm(z - np.array([l, 0.]), axis=1) < self.r) | \
        ((abs(z[:,1]) < self.r) & (0 <= z[:,0]) & (z[:,0] <= l))
        #plt.figure()
        #plt.plot(z[idx,0], z[idx,1], 'ko')
        #plt.show()
        d = z[idx,0]
        d[d < 0] = 0.
        return idx.nonzero(), d
        
    def brownianwalk(self, T=100, sigma=0.1):
        # Choose a number of steps that guarantees we'll go too far
        n = 2*T / (sigma / self.v)

        # Generate steps from this distribution
        steps = sigma / np.sqrt(2)*np.random.randn(n, 2)

        # Compute the cumulative time taken
        t = np.cumsum(np.hypot(steps[:,0], steps[:,1]) / self.v)

        # Take those steps, and truncate the last one
        i_last = np.where(t > T)[0][0]
        steps = steps[:i_last+1,:]
        steps[-1] = (T - t[i_last-1]) / (t[i_last] - t[i_last-1]) * steps[-1]
        
        # Perform the walk
        for u in steps:
            self.step(u)




if __name__ == "__main__":
    import matplotlib.pylab as plt

    s = sim2d(mu=10000, r=0.05, plot=True)
    sigma = 0.05;
    T = 200

    s.brownianwalk(T=T, sigma=sigma)
    #s.step(array([0.2, 0.2]))    
    
    plt.plot(s.t_eat[0:s.pointer], s.u[0:s.pointer])
    plt.show()




