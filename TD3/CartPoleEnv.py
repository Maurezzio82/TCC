import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from random import uniform
import pygame, math

def wrap_to_pi(theta):
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

def deg2rad(theta):
    return theta*np.pi/180

def boundary_potential(x, x_max, A = 50, a = 0.2 ):
    return  A/(1+((x-x_max)/a)**2) + A/(1+((x+x_max)/a)**2)

def upright_potential(theta ):#, A = 150, a = 0.1
    return -np.abs(theta)/np.pi #A / (1 + (theta / a)**2)
    

class CartPole(gym.Env):
    metadata = {'render.modes': ['console']}
    
    def __init__(self, gamma, simul_time = 15):
        super(CartPole, self).__init__()
        
        self.gamma = gamma          #temporal discount factor (necessary for the reward system)
              
        # System parameters
        self.m = 1.0    # mass at the tip of the pendulum
        self.M = 2.0    # mass of the cart
        self.L = 1.0    # pendulum length
        #self.b = 0.01   # viscous friction coefficient
        self.g = 9.81    # gravitational acceleration

        # Time step
        self.dt = 0.025
        
        # Action Constraint
        MaxForce = 70.0

        
        # Action and observation space
        # Control input u is bounded
        
        self.action_space = spaces.Box(
            low=np.array([-MaxForce], dtype=np.float32),
            high=np.array([MaxForce], dtype=np.float32),
            dtype=np.float32)
        
        self.x_max = 5.0
        
        # Observation: x, x', sin(θ), cos(θ), θ'
        self.observation_space = spaces.Box(
            low=np.array([-self.x_max, -np.inf, -1, -1, -np.inf], dtype=np.float32),
            high=np.array([self.x_max, np.inf, 1, 1, np.inf], dtype=np.float32),
            dtype=np.float32)
        
        
        self.reached_upright = False                              
        self.state = None
        self.current_it = 0                         #current iteration updated in the reset and step functions
        self.max_it = round(simul_time/self.dt)                    #max 25s of simulation time (for dt = 0.025, max_it = 800)



    def reset(self, seed=None, options=None, start_upright = False):
        super().reset(seed=seed)
        self.current_it = 0
        
        x_start = 0#uniform(-self.x_max/4, self.x_max/4)
        
        if start_upright:
            angle = uniform(-deg2rad(40), deg2rad(40))
            self.reached_upright = True
            self.started_upright = True
        else:
            angle = np.pi + uniform(-0.4, 0.4)                  # the pendulum is started close to the stable stationary point
            self.reached_upright = False
            self.started_upright = False
            
        self.state = np.array([x_start, 0.0, angle, 0.0], dtype=np.float32)  # initial position and velocity
        
        sin_theta = np.sin(angle)
        cos_theta = np.cos(angle)
        self.observation = np.array([x_start, 0.0, sin_theta, cos_theta, 0.0], dtype=np.float32)
        
        return self.observation, {}

    def dynamics(self, t, y, u):
        m = self.m
        M = self.M
        l = self.L
        g = self.g
        
        x, x_dot, theta, theta_dot = y
        
        # Mass matrix
        mass_matrix = np.array([
            [M + m, (m*l/2)*np.cos(theta)],
            [(m*l/2)*np.cos(theta), (m*l**2)/3]
        ])

        # Right-hand side of the equation of motion
        rhs = np.array([
            u + (m*l/2) * (theta_dot**2) * np.sin(theta),
            (m*g*l/2) * np.sin(theta)
        ])

        # Solve M * qdd = rhs
        qdd = np.linalg.solve(mass_matrix, rhs)

        x_ddot, theta_ddot = qdd
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def step(self, action):
        
        terminated = False
        truncated = False
        self.current_it += 1
    
        u = float(action)
            
        # Integrate ODE
        sol = solve_ivp(self.dynamics, [0, self.dt], self.state, args=(u,), t_eval=[self.dt])

        # Update state
        #prev_state = self.state
        self.state = sol.y[:, -1] # x, xd, θ, θd
        
        # WrapToPi
        self.state[2] = wrap_to_pi(self.state[2])
        
        x, x_dot, theta, theta_dot = self.state
        #_x, _x_dot, _theta, _theta_dot = prev_state


        upright_now = abs(theta) < 0.1 and abs(theta_dot) < 0.5 and abs(x_dot) < 1.0     
        
        if upright_now:
            self.reached_upright = True

        #=============== Reward System ===============#

        # Smooth reward shaping toward upright
        # shaping_reward =  self.gamma*upright_potential(theta) - upright_potential(_theta)
        
        # Cost terms
        cost = 3.0*x**2 + 0.1*x_dot**2 + 2.0*theta**2 + 0.2*theta_dot**2 + 0.01*u**2 #1.0*x**2 + 0.1*x_dot**2 + 2*theta**2 + 0.2*theta_dot**2 + 0.01*u**2 
                # the max value of this cost should be around 70 in worst case cenario

        # Punishment for coming close to the boundary
        # if the current position is closer to either boundary than the previous one, there should be a cost to it
        #shaping_cost = self.gamma*boundary_potential(x, self.x_max) - boundary_potential(_x, self.x_max)


        # shaping_cost and shaping reward are omited for preliminary testing
        # a +1 is added to ensure a positive reward. Since the episode ends when the 
        # cart goes out of bound, the agent was practicing some reward hacking by going
        # straight to the boundary to end the episode, thus ending the negative reward
        reward = 1 - 0.02*cost
        #reward = np.clip(reward, -1.0, 1.0) # np.tanh(reward)

        #=============================================#
        
        if x>self.x_max or x<-self.x_max:
            truncated = True

        if abs(theta) > np.pi/2 and self.started_upright:
            truncated = True
        
        if self.current_it >= self.max_it:
            truncated = True
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        self.observation = np.array([x, x_dot, sin_theta, cos_theta, theta_dot], dtype=np.float32)
        return self.observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if not hasattr(self, 'screen'):
            pygame.init()
            pygame.display.init()
            self.screen_size = (1500,400)#(600, 400)
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Cart-Pole Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))  # white background

        width, height = self.screen_size
        ground_y = height // 2  # ground baseline

        # Cart dimensions (in pixels)
        cart_width, cart_height = 60, 30
        pole_length = 150 * self.L/1.0  # pixels (visual scaling)

        # Extract state variables
        x, x_dot, theta, theta_dot = self.state

        # Convert cart position (x) to screen coordinates
        # Scaling factor: how many pixels per unit x
        scale = 100  # px per meter (adjust as needed)
        cart_x = width // 2 + int(x * scale)
        cart_y = ground_y - cart_height // 2

        # Draw cart as a rectangle
        cart_rect = pygame.Rect(0, 0, cart_width, cart_height)
        cart_rect.center = (cart_x, cart_y)
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect)

        # Pole pivot point (top-center of cart)
        pivot = (cart_x, cart_y - cart_height // 2)

        # Pole tip position
        pole_x = pivot[0] + pole_length * math.sin(theta)
        pole_y = pivot[1] - pole_length * math.cos(theta)

        # Draw pole
        pygame.draw.line(self.screen, (0, 0, 255), pivot, (pole_x, pole_y), 4)

        # Draw bob (red circle at the tip)
        pygame.draw.circle(self.screen, (255, 0, 0), (int(pole_x), int(pole_y)), 10)

        # Draw ground line
        pygame.draw.line(self.screen, (0, 0, 0), (0, ground_y), (width, ground_y), 2)

        pygame.display.flip()
        self.clock.tick(60)


    def close(self):
        if hasattr(self, 'screen'):
            pygame.display.quit()
            pygame.quit()
            del self.screen
