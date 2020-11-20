import numpy as np
import math
import random

from datetime import datetime
from itertools import product


class CabDriverEnvironment:
    def __init__(self, locations=5, cost=5, reward=9):
        """
        Constructor for intializing environment class
        """
        self.hyperparams = self.init_hyperparams(locations, cost, reward)
        self.action = self.init_action()
        self.state= self.init_state()
        self.state_initial = self.set_inital_state()
        self.time_matrix = np.load("TM.npy")
        self.reset_state()

    def init_hyperparams(self, locations, cost, reward):
        """
        Initialize all the hyperparams in the form of dictionary
        """
        hyperparms = {
            "m": locations,  # locations
            "t": 24,  # 24hr in day
            "d": 7,  #days
            "C": cost,  # fuel cost/hr
            "R": reward,  #reward from passenger/hrs
        }
        return hyperparms

    def init_action(self) -> list:
        """
        An action will be represeted by to and from location (src, destination).
        Function is used to initailize the entire action space
        """
        location_count = self.hyperparams["m"]
        action = [
            (src, dest)
            for src in range(1, location_count + 1)
            for dest in range(1, location_count + 1)
            if src != dest
        ]
        action.append((0, 0))
        return action

    def init_state(self) -> list:
        """
        The action space is represented by (driver current location, curr time, day of week)
        This function intializes all the combination of above mentioned values
        """

        locations = [i for i in range(1, self.hyperparams["m"] + 1)]
        hrs = [i for i in range(0, self.hyperparams["t"])]
        week_day = [i for i in range(0, self.hyperparams["d"])]
        state = product(locations, hrs, week_day)
        # convert product object into a list
        return list(state)

    def set_inital_state(self):
        """ Select a random state for a cab to start
        """

        location = [i for i in range(1, self.hyperparams["m"] + 1)]
        hours = [i for i in range(0, self.hyperparams["t"])]
        days = [i for i in range(0, self.hyperparams["d"])]

        return (np.random.choice(location), np.random.choice(hours), np.random.choice(days))

    def reset_state(self):
        return self.action, self.state, self.state_initial


    def state_to_vec(self, state):
        """
        Function to convert state to vector so that it can be fed to NN
        This method converts a given state into a vector format. The vector is of size m + t + d.
        """

        locations = np.zeros(self.hyperparams["m"])
        hours = np.zeros(self.hyperparams["t"])
        days = np.zeros(self.hyperparams["d"])

        curr_loc, curr_hour, curr_day = state

        locations[curr_loc  - 1] = 1
        hours[curr_hour] = 1
        days[curr_day] = 1

        state_vect = np.hstack(
            (locations, hours, days)
        ).reshape(
            1,
            self.hyperparams["m"]
            + self.hyperparams["t"]
            + self.hyperparams["d"],
        )

        return state_vect

    def get_total_possible_requests(self, current_state):
        """
        Use Poisson distribution to return number of locations
        """
        current_location = current_state[0]
        distribution_lambda = [2, 12, 4, 7, 8]
        total_requests = np.random.poisson(
            distribution_lambda[current_location - 1]
        )
        if total_requests >= 10:
            total_requests = 10
        return total_requests

    def get_requests_per_location(self, current_state):
        """
        Get the possible request by a cab depending on the current state of the driver
        This function return the actions a driver can take
        """
        total_requests = self.get_total_possible_requests(current_state)
        # Removing last option of  0,0 for actions as that is not a valid action
        total_actions = self.action[:-1]
        action_index = random.sample(
            range(len(total_actions)), total_requests
        )
        allowed_actions = [total_actions[i] for i in action_index]

        allowed_actions.append((0, 0))

        if len(action_index) == 0:
            action_index = [0]
        else:
            action_index.append(len(action_index))

        return action_index, allowed_actions

    def get_next_state(self, state, action):
        """
        Get the next state of cab driver
        1. next state
        2. reward
        3. ride time
        """
        current_location, current_hour, current_day = state
        next_state = None
        location_src, location_dst = action

        total_rewards = self.get_rewards_per_ride(state, action)
        total_ride_time = 0

        if action == (0, 0):
            # For specific state increase hour by1
            current_hour = int(current_hour + 1)
            current_hour, current_day = self.update_revised_time(
                current_hour, current_day
            )

            total_ride_time = 1
            next_state = (current_location, current_hour, current_day)
        else:
            if current_location == location_src:
                total_trip_time, travel_time_to_customer = self.get_same_pickup_time(
                    state, action
                )

                # calculate time at the end of the trip
                time_at_trip_end = int(current_hour + total_trip_time)
                # factor next day if time exceeds 23:00 hours
                time_at_trip_end, day_at_trip_end = self.update_revised_time(
                    time_at_trip_end, current_day
                )

                total_ride_time = total_trip_time
                next_state = (location_dst, time_at_trip_end, day_at_trip_end)
            else:
                (
                    total_trip_time,
                    travel_time_to_customer,
                    time_at_customer_location,
                    day_at_customer_location,
                ) = self.get_different_pickup_time(state, action)

                # calculate time at the end of the trip
                # use the computed time at customer location instead of current hour
                time_at_trip_end = int(time_at_customer_location + total_trip_time)
                # factor next day if time exceeds 23:00 hours
                (time_at_trip_end, day_at_trip_end) = self.update_revised_time(
                    time_at_trip_end, day_at_customer_location
                )

                total_ride_time = total_trip_time + travel_time_to_customer
                next_state = (location_dst, time_at_trip_end, day_at_trip_end)

        return next_state, total_rewards, total_ride_time


    def get_rewards_per_ride(self, state, action):
        """ Calculated Reward :
            (revenue earned from pickup point to drop point)
            - (electric cost from pickup point to drop point)
            - (electric cost from current point to pick-up point)
           Cost is per hour based usage
        """
        calculated_reward = 0
        total_trip_time = 0
        travel_time_to_customer = 0

        curr_loc = state[0]
        location_src = action[0]

        if action == (0, 0):
            reward = -self.hyperparams["C"]
        else:
            if curr_loc == location_src:
                total_trip_time, travel_time_to_customer = self.get_same_pickup_time(state, action)
            else:
                (total_trip_time,travel_time_to_customer, _,_,) = self.get_different_pickup_time(state, action)

            reward = (self.hyperparams['R'] * total_trip_time -(self.hyperparams['C'] * (total_trip_time + travel_time_to_customer)))
        return reward

    def get_same_pickup_time(self, state, action):
        """ Calculate total trip time when current location and pickup location are same
        """
        _, current_hour, current_day = state
        location_src, location_dst = action

        total_trip_time = (self.time_matrix
            [int(location_src - 1)]
            [int(location_dst - 1)]
            [int(current_hour)]
            [int(current_day)]
            )
        # set travel_time_to_customer as 0
        return total_trip_time, 0

    def get_different_pickup_time(self, state, action):
        """
        Calculate total trip time and time taken to reach the customer when current location
        and pickup location are different
        """
        current_location, current_hour, current_day = state
        location_from, location_to = action

        travel_time_to_customer = (self.time_matrix[int(current_location - 1)][int(location_from - 1)][int(current_hour)][int(current_day)])
        time_at_customer_location = int(current_hour + travel_time_to_customer)
        day_at_customer_location = current_day
        (
            time_at_customer_location,
            day_at_customer_location,
        ) = self.update_revised_time(
            time_at_customer_location, day_at_customer_location
        )
        total_trip_time = (self.time_matrix[int(location_from - 1)][int(location_to - 1)][int(time_at_customer_location)][int(day_at_customer_location)])
        return (total_trip_time,travel_time_to_customer,time_at_customer_location,day_at_customer_location,)


    def update_revised_time(self, day_time, week_day):
        """
        Update the time if its exceeding beyond 24 hrs or days of week is more than 6 days in week
        """
        if day_time >= 24:
            day_time = int(day_time - 24)
            if week_day == 6:
                week_day = 0
            else:
                week_day = int(week_day + 1)

        return day_time, week_day
