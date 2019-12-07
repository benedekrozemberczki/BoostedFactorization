"""Running the boosted machine."""

from helpers import parameter_parser,tab_printer, read_matrix, DeepWalker
from boosted_embedding import BoostedFactorization

def learn_boosted_embeddings(args):
    """
    Method to create a boosted matrix/network embedding.
    :param args: Arguments object.
    """
    if args.dataset_type == "graph":
        A = DeepWalker(args).A
    else:
        A = read_matrix(args.input_path)
    model = BoostedFactorization(A, args)
    model.do_boosting()
    model.save_embedding()

if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    learn_boosted_embeddings(args)
