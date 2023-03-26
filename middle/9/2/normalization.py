from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


class Normalization:
    @staticmethod
    def by_column(matrix: csr_matrix) -> csr_matrix:
        """Normalization by column

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        col_sum = matrix.sum(axis=0)
        col_sum[col_sum == 0] = 1
        norm_matrix = matrix.multiply(1. / col_sum)
        return norm_matrix.tocsr()

    @staticmethod
    def by_row(matrix: csr_matrix) -> csr_matrix:
        """Normalization by row

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        row_sum = matrix.sum(axis=1)
        row_sum[row_sum == 0] = 1
        norm_matrix = matrix.multiply(1. / row_sum)
        return norm_matrix.tocsr()

    @staticmethod
    def tf_idf(matrix: csr_matrix) -> csr_matrix:
        """Normalization using tf-idf

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        tf = Normalization.by_row(matrix)

        idf = np.log(matrix.shape[0] / matrix.astype(bool).sum(axis=0))
        # print(idf)

        norm_matrix = tf.multiply(idf)
        return norm_matrix.tocsr()

    @staticmethod
    def bm_25(
        matrix: csr_matrix, k1: float = 2.0, b: float = 0.75
    ) -> csr_matrix:
        """Normalization based on BM-25

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        tf = Normalization.by_row(matrix)
        idf = np.log(matrix.shape[0] / matrix.astype(bool).sum(axis=0))

        d = matrix.sum(axis=1)
        avgdl = np.average(d)
        delta = k1 * (1 - b + b * (d / avgdl))
        step1 = tf.power(-1).multiply(delta)
        step1.data += 1
        step1 = step1.power(-1).multiply(k1+1)

        return step1.multiply(idf).tocsr()


if __name__ == "__main__":
    from user_item_matrix import UserItemMatrix

    df = pd.read_csv("data.csv")
    uim = UserItemMatrix(df)
    a = Normalization.bm_25(uim.csr_matrix)
