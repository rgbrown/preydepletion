# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial

# Prey Depletion numerical experiments
# ====================================
# 
# We want to investigate how accumulated prey depletion slows the rate of population depletion when compared with a well-mixed model.

# deplete2d class
# ---------------
# 
# Below is a class that sets up an environment with a random number of prey items of a specified Poisson density, and a single predator. The predator has a given perceptive radius, and constant speed, and individual vector steps can be specified.

def init_prey(n):
    """
    Initialize prey using a structured array
    """
    
    prey = np.zeros(n, dtype='uint32, 2float64, uint8')
    prey.dtype.names = ('index', 'position', 'alive')
    prey['index'] = range(n)
    prey['position'] = np.random.rand(n, 2)
    prey['alive'] = np.ones(n)
    return prey
    
def init_output_array(n):
    output = np.zeros(n, dtype='float64, uint32')
    output.dtype.names = ('time', 'index')
    output['time'] = np.inf
    return output

class deplete2d:
    def __init__(self, density=100, speed=1., radius=0.01):
        # The actual number of prey is Poisson distributed
        self.n_prey = scipy.random.poisson(density)
        self.population = self.n_prey
        self.output = init_output_array(self.n_prey)
        self.outp = 0
        self.time = 0.
        
        # Initialise prey, and make a copy to work on, and compute KDTree
        self.prey_original = init_prey(self.n_prey)
        self.prey = self.prey_original[self.prey_original['alive'].nonzero()]
        self.tree = scipy.spatial.cKDTree(self.prey['position'])
        
        self.speed = speed
        self.radius = radius
        self.x_pred = np.array([0.5, 0.5])
        
    def update_domain(self, centre=[0.5,0.5]):
        """
        translate the domain (mod 1) so that centre is the new centre,
        copy dead prey information to original block,
        create new working prey set,
        update KD Tree
        """
        centre = np.array(centre)
        assert(centre.size == 2)
        
        # First, copy dead dude information back to original prey block
        i_dead = self.prey['alive'] == 0
        self.prey_original['alive'][self.prey[i_dead]['index']] = 0
        
        # Remove dead prey items
        self.prey = self.prey[~i_dead]
        
        # Recenter and recompute tree
        self.prey['position'] = np.remainder(self.prey['position'] + 
                                          np.array([0.5, 0.5]) - centre, 1.)
               
        if self.prey.size != 0:
            self.tree = scipy.spatial.cKDTree(self.prey['position'])
        
    def step(self, u):
        if self.population == 0:
            return
        u = np.array(u)
        assert(u.size == 2)
        
        # Check whether domain update is required
        if ((self.x_pred + u < self.radius) |
            (self.x_pred + u > 1 - self.radius)).any():
            self.update_domain(self.x_pred + u)
            self.x_pred = np.array([0.5, 0.5]) - u
            
        # We may now assume that our step will stay in bounds
        step_length = np.hypot(u[0], u[1])
        
        # Query KDTree for nearby points
        i_nearby = np.array(self.tree.query_ball_point(self.x_pred, step_length + self.radius))
        
        # Now, create a smaller copy to work on, with only nearby (and alive) dudes
        if i_nearby.size != 0:
            prey_local = self.prey[i_nearby]
            i_keep = prey_local['alive'] > 0
            i_nearby = i_nearby[i_keep]
            prey_local = prey_local[i_keep]
      
            # Find the indices (into prey_local) of the prey eaten on the step, and the
            # distance travelled when each one was eaten
            i_eaten, d_travelled = self.find_prey(prey_local, u)
            n_eaten = i_eaten[0].size
        
            #print self.prey['alive'][i_nearby[i_eaten]]
            assert(self.prey['alive'][i_nearby[i_eaten]].all())
            self.prey['alive'][i_nearby[i_eaten]] = 0
        
            self.prey_original['alive'][prey_local[i_eaten]['index']] = 0
            self.output[self.outp:self.outp + n_eaten]['index'] = prey_local[i_eaten]['index']
            self.output[self.outp:self.outp + n_eaten]['time'] = self.time + d_travelled / self.speed
            self.outp += n_eaten
            self.population -= n_eaten
        
        # Update time and predator position
        self.time += step_length / self.speed
        self.x_pred += u
        
    def find_prey(self, prey_local, u):
        # Project onto u and u_perp by rotating candidate points. 
        # Predator position before step is used as centre of rotation
        step_length = np.hypot(u[0], u[1])
        Q = 1 / step_length * np.array([[u[0], -u[1]], [u[1], u[0]]])
    
        Z = prey_local['position']
        Z = np.dot(Z - self.x_pred, Q)
        
        # The step is now along the first axes. Look for pieces within radius r of 
        # the line segment between (0,0) and (step_length, 0)
        idx = ((np.linalg.norm(Z, axis=1) < self.radius) |
               (np.linalg.norm(Z - np.array([step_length, 0]), axis=1) < self.radius) |
               ((abs(Z[:,1]) < self.radius) & 
                (0 <= Z[:, 0]) & (Z[:, 0] <= step_length)))
        
        # Define d_travelled to be when the predator is closest to the prey item on its 
        # path through
        d_travelled = Z[idx, 0]
        
        # If any values are negative, then assume they were eaten instantaneously.
        # This should only happen on the first time step
        d_travelled[d_travelled < 0] = 0.
        
        return idx.nonzero(), d_travelled
    
    def get_output(self):
        t = np.sort(self.output[0:self.outp]['time'])
        return (t, self.n_prey - np.arange(self.outp))

    def get_prey(self, t):
        """get prey at time t
        """
        i_dead = self.output[self.output['time'] <= t]['index']
        i_alive = np.setdiff1d(self.prey_original['index'], i_dead,
                assume_unique=True)
        return self.prey_original[i_alive]
        

        
        
def brownianwalk(sim, T=100, sigma=0.1):
    nsteps = 2*T / (sigma / sim.speed)

    # The mess below creates too many steps, then truncates them to the right amount
    # so that the simulation runs exactly to time, truncating the last step 
    steps = sigma / np.sqrt(2) * np.random.randn(nsteps, 2)
    t = np.cumsum(np.hypot(steps[:,0], steps[:,1]) / sim.speed)
    i_last = np.where(t > T)[0][0]
    steps = steps[:i_last+1,:]
    steps[-1] = (T - t[i_last-1]) / (t[i_last] - t[i_last-1]) * steps[-1]

    for u in steps:
        sim.step(u)




if __name__ == "__main__":
    sim = deplete2d(density=1000, radius=0.01)
    brownianwalk(sim, T=200, sigma=0.01)
    t, pop = sim.get_output()
    plt.plot(t, pop)
    plt.show()

