import json
import numpy as np
import pandas as pd


class CollegeRecommender:
    def __init__(self):
        # Load the college data
        self.colleges = pd.read_csv(
            "data/collegedata.csv"
        )  # Ensure you have a CSV file with the required data
        self.q_table = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.last_state_action = None
        self.last_recommended = None

        # Drop rows with any missing values or fill them with default values
        self.colleges.dropna(
            subset=["INSTNM", "CITY", "STABBR", "ZIP", "INSTURL"], inplace=True
        )

    def recommend(self):
        # Select a random college to recommend
        college = self.colleges.sample(1).iloc[0]
        self.last_recommended = college  # Store the last recommended college
        return college

    def update(self, state, action, reward):
        state_action = (state, action)
        if state_action not in self.q_table:
            self.q_table[state_action] = 0

        if self.last_state_action is not None:
            last_state_action = self.last_state_action
            self.q_table[last_state_action] = self.q_table.get(
                last_state_action, 0
            ) + self.alpha * (
                reward
                + self.gamma * self.q_table[state_action]
                - self.q_table[last_state_action]
            )

        self.last_state_action = state_action

    def handle_feedback(self, feedback):
        reward = 1 if feedback == True else -1
        if self.last_recommended is not None:
            state = json.dumps(self.last_recommended.to_dict())
            action = feedback
            self.update(state, action, reward)
        return self.recommend()


# Initialize the model
recommender = CollegeRecommender()
