"""Boosted factorization machine."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from helpers import sampling, simple_print

class BoostedFactorization:
    """
    LENS-NMF boosted matrix factorization model.
    """
    def __init__(self, residuals, args):
        """
        Initialization method.
        """
        self.args = args
        self.residuals = residuals
        simple_print("Matrix sum: ", self.residuals.sum())
        self.shape = residuals.shape
        indices = self.residuals.nonzero()
        self.index_1 = indices[0]
        self.index_2 = indices[1]
        self.edges = zip(self.index_1, self.index_2)
        print("\nFitting benchmark model.")
        base_score, __ = self.fit_and_score_NMF(self.residuals)
        simple_print("Benchmark loss", base_score.sum())

    def sampler(self, index):
        """
        Anchor sampling procedure.
        :param index: Matrix axis row/column chosen for anchor sampling.
        :return sample: Chosen sampled row/column id.
        """
        row_weights = self.residuals.sum(axis=index)
        if len(row_weights.shape) > 1:
            row_weights = row_weights.reshape(-1)
        sums = np.sum(np.sum(row_weights))
        to_pick_from = {i: float(row_weights[0, i])**2/sums for i in range(row_weights.shape[1])}
        sample = sampling(to_pick_from)
        return sample

    def reweighting(self, X, chosen_row, chosen_column):
        """
        Rescaling the target matrix with the anchor row and column.
        :param X: The target matrix rescaled.
        :param chosen_row: Anchor row.
        :param chosen_column: Anchor column.
        :return X: The rescaled residual.
        """
        row_sims = X.dot(chosen_row.transpose())
        column_sims = chosen_column.transpose().dot(X)
        X = sparse.csr_matrix(row_sims).multiply(X)
        X = X.multiply(sparse.csr_matrix(column_sims))
        return X

    def fit_and_score_NMF(self, new_residuals):
        """
        Factorizing a residual matrix, returning the approximate target and an embedding.
        :param new_residuals: Input target matrix.
        :return scores: Approximate target matrix.
        :return W: Embedding matrix.
        """
        model = NMF(n_components=self.args.dimensions,
                    init="random",
                    verbose=False,
                    alpha=self.args.alpha)

        W = model.fit_transform(new_residuals)
        H = model.components_
        print("Scoring started.\n")
        sub_scores = np.sum(np.multiply(W[self.index_1, :], H[:, self.index_2].T), axis=1)
        scores = np.maximum(self.residuals.data-sub_scores, 0)
        scores = sparse.csr_matrix((scores, (self.index_1, self.index_2)),
                                   shape=self.shape,
                                   dtype=np.float32)
        return scores, W

    def single_boosting_round(self, iteration):
        """
        A method to perform anchor sampling, rescaling, factorization and scoring.
        :param iteration: Number of boosting round.
        """
        row = self.sampler(1)
        column = self.sampler(0)

        chosen_row = self.residuals[row, :]
        chosen_column = self.residuals[:, column]
        new_residuals = self.reweighting(self.residuals, chosen_row, chosen_column)
        scores, embedding = self.fit_and_score_NMF(new_residuals)
        self.embeddings.append(embedding)
        self.residuals = scores

    def do_boosting(self):
        """
        Doing a series of matrix-factorizations on the anchor-sampled residual matrices.
        """
        self.embeddings = []
        for iteration in range(self.args.iterations):
            print("\nFitting model: "+str(iteration+1)+"/"+str(self.args.iterations)+".")
            self.single_boosting_round(iteration)
            simple_print("Boosting round "+str(iteration+1)+". loss", self.residuals.sum())

    def save_embedding(self):
        """
        Saving the embedding at the default path.
        """
        ids = np.array(range(self.residuals.shape[0])).reshape(-1, 1)
        self.embeddings = [ids] + self.embeddings
        self.embeddings = np.concatenate(self.embeddings, axis=1)
        feature_names = ["x_"+str(x) for x in range(self.args.iterations*self.args.dimensions)]
        columns = ["ID"] + feature_names
        self.embedding = pd.DataFrame(self.embeddings, columns=columns)
        self.embedding.to_csv(self.args.output_path, index=None)
