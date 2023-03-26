from typing import Dict

import pandas as pd
from scipy.sparse import csr_matrix


class UserItemMatrix:
    def __init__(self, sales_data: pd.DataFrame):
        """Class initialization. You can make necessary
        calculations here.

        Args:
            sales_data (pd.DataFrame): Sales dataset.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36
            ...

        """
        users_id = sorted(sales_data['user_id'].unique())
        items_id = sorted(sales_data['item_id'].unique())
        self.map_user_id = {x: idx for idx, x in enumerate(users_id)}
        self.map_item_id = {x: idx for idx, x in enumerate(items_id)}

        self.n_row = len(set(self.map_user_id.values()))
        self.n_col = len(set(self.map_item_id.values()))

        row = [self.map_user_id[x] for _, x in sales_data['user_id'].iteritems()]
        col = [self.map_item_id[x] for _, x in sales_data['item_id'].iteritems()]
        data = [value for idx, value in sales_data['qty'].iteritems()]

        self.csr_matrix_ = csr_matrix((
            data,
            (row, col)),
            shape=(self.n_row, self.n_col))

    @property
    def user_count(self) -> int:
        """
        Returns:
            int: the number of users in sales_data.
        """
        return self.n_row

    @property
    def item_count(self) -> int:
        """
        Returns:
            int: the number of items in sales_data.
        """
        return self.n_col

    @property
    def user_map(self) -> Dict[int, int]:
        """Creates a mapping from user_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            user_map (Dict[int, int]):
                {1: 0, 2: 1, 4: 2, 5: 3}

        Returns:
            Dict[int, int]: User map
        """
        return self.map_user_id

    @property
    def item_map(self) -> Dict[int, int]:
        """Creates a mapping from item_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            item_map (Dict[int, int]):
                {118: 0, 285: 1, 1229: 2, 1688: 3, 2068: 4}

        Returns:
            Dict[int, int]: Item map
        """
        return self.map_item_id

    @property
    def csr_matrix(self) -> csr_matrix:
        """User items matrix in form of CSR matrix.

        User row_ind, col_ind as
        rows and cols indecies (mapped from user/item map).

        Returns:
            csr_matrix: CSR matrix
        """
        return self.csr_matrix_


if __name__ == "__main__":
    pd = pd.read_csv("data.csv")
    uim = UserItemMatrix(pd)
    print(uim.item_count)
