from scipy.spatial import distance
from scipy.interpolate import interp1d
from collections import deque
from numpy import arctan2, unwrap, linspace
from abc import ABC, abstractmethod
from math import sqrt
from scipy.integrate import quad

class Segment(ABC):
    def __init__(self, points):
        """
        Initializes a Segment object with given points and sets up vehicles deque and functions.
        ## Args:
            points (list): A list of points that define the segment.
        ## Example:
            >>> points = [(0, 0), (1, 1), (2, 2)]
            >>> segment = Segment(points)
        """

        self.points = points
        self.vehicles = deque()

        self.set_functions()
        

    def set_functions(self):
        """
        Sets up interpolation functions for points and headings along a segment.
        This method initializes two interpolation functions:
        1. `self.get_point`: Interpolates points along the segment.
        2. `self.get_heading`: Interpolates headings (angles) along the segment.
        The `self.get_point` function uses linear interpolation to estimate the 
        position of a point along the segment based on a parameter ranging from 0 to 1.
        The `self.get_heading` function calculates the heading (angle) between 
        consecutive points along the segment. If there is only one heading, it 
        returns a constant function. Otherwise, it uses linear interpolation to 
        estimate the heading based on a parameter ranging from 0 to 1.
        ## Example:
            >>> segment = Segment(points=[(0, 0), (1, 1), (2, 0)])
            >>> segment.set_functions()
            >>> point = segment.get_point(0.5)
            >>> heading = segment.get_heading(0.5)
        """

        # Point
        self.get_point = interp1d(linspace(0, 1, len(self.points)), self.points, axis=0)
        
        # Heading
        headings = unwrap([arctan2(
            self.points[i+1][1] - self.points[i][1],
            self.points[i+1][0] - self.points[i][0]
        ) for i in range(len(self.points)-1)])
        if len(headings) == 1:
            self.get_heading = lambda x: headings[0]
        else:
            self.get_heading = interp1d(linspace(0, 1, len(self.points)-1), headings, axis=0)

    def get_length(self):
        """
        Calculate the total length of the segment by summing the Euclidean distances 
        between consecutive points.
        ## Returns:
            float: The total length of the segment.
        ## Example:
            >>> segment = Segment(points=[(0, 0), (3, 4), (6, 8)])
            >>> segment.get_length()
            10.0
        """
        length = 0
        for i in range(len(self.points) -1):
            length += distance.euclidean(self.points[i], self.points[i+1])
        return length

    def add_vehicle(self, veh):
        """
        Adds a vehicle to the segment.
        ## Args:
            veh (Vehicle): The vehicle to be added. It is expected that the vehicle object has an 'id' attribute.
        """
        self.vehicles.append(veh.id)

    def remove_vehicle(self, veh):
        """
        Removes a vehicle from the segment.
        ## Parameters:
        veh (Vehicle): The vehicle object to be removed. It is expected that the vehicle object has an 'id' attribute.
        """
        self.vehicles.remove(veh.id)

    def compute_x(self, t:float)->float:
        """
        Compute the x-coordinate of a point on the segment at a given parameter t.
        ## Args:
            t (float): The parameter, typically between 0 and 1, representing the 
                       position along the segment where 0 is the start and 1 is the end.
        ## Returns:
            float: The x-coordinate of the point on the segment at parameter t.
        """

        return self.start[0] + t * self.delta[0]

    def compute_y(self, t:float)->float:
        """
        Compute the y-coordinate of a point on the segment at a given parameter t.
        ## Args:
            t (float): The parameter t, typically between 0 and 1, representing the 
                       position along the segment where 0 is the start and 1 is the end.
        ## Returns:
            float: The y-coordinate of the point on the segment at the given parameter t.
        """

        return self.start[1] + t * self.delta[1]

    def compute_dx(self, t:float)->float:
        """
        Compute the change in the x-coordinate (dx) for a given parameter t.
        ## Parameters:
        t (float): The parameter value at which to compute dx.
        ## Returns:
        float: The change in the x-coordinate (dx).
        """
        return self.delta[0]

    def compute_dy(self, t:float)->float:
        """
        Compute the change in the y-coordinate (dy) for a given parameter t.
        ## Parameters:
        t (float): A parameter that might be used to compute dy (currently unused).
        ## Returns:
        float: The change in the y-coordinate (dy).
        """
        return self.delta[1]

    def abs_f(self, t):
        """
        Calculate the absolute value of the function at a given parameter t.
        This method computes the Euclidean distance (or magnitude) of the vector 
        formed by the derivatives of the function with respect to t.
        ## Args:
            t (float): The parameter at which to evaluate the function.
        ## Returns:
            float: The Euclidean distance at the given parameter t.
        """
        return sqrt(self.compute_dx(t)**2 + self.compute_dy(t)**2)
    
    def find_t(self, a, L, epsilon):
        """
        Finds the parameter `t` such that the integral of `self.abs_f` from `a` to `t` is approximately equal to `L`.
        ## Parameters:
        a (float): The lower bound of the integration.
        L (float): The target length for the integral.
        epsilon (float): The tolerance for the difference between the integral and `L`.
        ## Returns:
        float: The parameter `t` such that the integral of `self.abs_f` from `a` to `t` is approximately equal to `L`.
               If the target length `L` cannot be reached, returns 1.
        ## Example:
            >>> segment = Segment()
            >>> t = segment.find_t(0, 5, 0.01)
            0.75
        """
        def f(t):
            integral_value, _ = quad(self.abs_f, a, t)
            return integral_value
        
        # if we cannot reach the target length, return 1 
        if f(1) < L: return 1
        
        lower_bound = a
        upper_bound = 1
        mid_point = (lower_bound + upper_bound) / 2.0
        integ = f(mid_point)
        while abs(integ-L) > epsilon:
            if integ < L:       lower_bound = mid_point
            else:               upper_bound = mid_point
            mid_point = (lower_bound + upper_bound) / 2.0
            integ = f(mid_point)
        return mid_point
    
    def find_normalized_path(self, CURVE_RESOLUTION=50):
        """
        Generates a normalized path for the segment.

        This method computes a series of points along the segment, spaced evenly
        according to the specified curve resolution. The path starts at the beginning
        of the segment and ends at the end of the segment.

        ## Args:
            CURVE_RESOLUTION (int, optional): The number of points to generate along
                the path. Default is 50.

        ## Returns:
            list of tuple: A list of (x, y) tuples representing the normalized path.
        ## Example:
            >>> segment = Segment(points=[(0, 0), (1, 2), (4, 4)])
            >>> normalized_path = segment.find_normalized_path(CURVE_RESOLUTION=5)
            >>> normalized_path
            [(0, 0), (1.0, 2.0), (2.5, 3.0), (3.5, 3.5), (4, 4)]
        """

        normalized_path = [(self.compute_x(0), self.compute_y(0))]
        l = self.get_length()
        target_l = l/(CURVE_RESOLUTION-1)
        epsilon = 0.01
        a = 0
        for i in range(CURVE_RESOLUTION-1):
            t = self.find_t(a, target_l, epsilon)
            new_point = (self.compute_x(t), self.compute_y(t))
            normalized_path.append(new_point)
            if t == 1: break
            else:      a = t
        return normalized_path