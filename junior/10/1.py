from typing import Tuple, List
import numpy as np


ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

class Experiment:
    """Experiment class. Contains the logic for assigning users to groups."""

    def __init__(
        self,
        experiment_id: int,
        groups: Tuple[str] = ("A", "B"),
        group_weights: List[float] = None,
    ):
        """Constructor."""
        self.experiment_id = experiment_id
        self.groups = groups
        self.group_weights = group_weights
        # Define the salt for experiment_id.
        # The salt should be deterministic and unique for each experiment_id.
        experiment_id_str = str(experiment_id)

        chars = []
        for i in range(16):
            n = int(experiment_id_str[i % len(experiment_id_str)])
            chars.append(ALPHABET[n])
        self.salt = "".join(chars)

        # Define the group weights if they are not provided equaly distributed
        # Check input group weights. They must be non-negative and sum to 1.
        if not group_weights:
            self.group_weights = [1 / len(self.groups)] * len(self.groups)
        else:
            self.group_weights = group_weights

    def group(self, click_id: int) -> Tuple[int, str]:
        """Assigns a click to a group.

        Parameters
        ----------
        click_id: int :
            id of the click

        Returns
        -------
        Tuple[int, str] :
            group id and group name
        """

        # Assign the click to a group randomly based on the group weights
        # Return the group id and group name
        n = (hash(str(click_id) + self.salt) % 1000) / 1000
        group_id = next(i for i,x in  enumerate(np.cumsum(self.group_weights)) if x > n)

        return group_id, self.groups[group_id]
