import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit


def items_embeddings(ui_matrix: csr_matrix, dim: int) -> np.ndarray:
    """Build items embedding using factorization model.
    The order of items should be the same in the output matrix.

    Args:
        ui_matrix (pd.DataFrame): User-Item matrix of size (N, M)
        dim (int): Dimention of embedding vectors

    Returns:
        np.ndarray: Items embeddings matrix of size (M, dim)
    """
    print(ui_matrix.shape)
    model = implicit.als.AlternatingLeastSquares(
        factors=dim, regularization=0.13, iterations=27, random_state=42)
    
    alpha_val = 40
    data_conf = (ui_matrix * alpha_val).astype('double')

    model.fit(data_conf.T)
    
    return model.user_factors


if __name__ == "__main__":
    from user_item_matrix import UserItemMatrix
    from normalization import Normalization

    df = pd.read_csv("data.csv")
    uim = UserItemMatrix(df)
    print(uim.n_row)
    matrix = Normalization.bm_25(uim.csr_matrix)
    emb = items_embeddings(matrix, 17)
    pd.DataFrame(emb).to_pickle("emb.pkl")
