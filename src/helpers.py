import argparse
import random
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import sparse
from collections import Counter
from texttable import Texttable

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Wikipedia Giraffes graph.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """

    parser = argparse.ArgumentParser(description = "Run LENS-NMF.")


    parser.add_argument('--input-path',
                        nargs = '?',
                        default = './input/giraffe_edges.csv',
	                help = 'Input folder with jsons.')

    parser.add_argument('--output-path',
                        nargs = '?',
                        default = './output/giraffe_embedding.csv',
	                help = 'Input folder with jsons.')
	
    parser.add_argument('--graph',
                        type = bool,
                        default = True,
			dest='graph',
			action='store_true',
	                help = 'Is the target is a graph. Default is True.')

    parser.add_argument('--dimensions',
                        type = int,
                        default = 8,
	                help = 'Number of dimensions. Default is 8.')

    parser.add_argument('--alpha',
                        type = float,
                        default = 0.001,
	                help = 'Regularization coefficient. Default is 0.001.')

    parser.add_argument('--window-size',
                        type = int,
                        default = 3,
	                help = 'Skip-gram window size. Default is 3.')

    parser.add_argument('--walk-length',
                        type = int,
                        default = 80,
	                help = 'Truncated random walk length. Default is 80.')

    parser.add_argument('--number-of-walks',
                        type = int,
                        default = 10,
	                help = 'Number of random walks per source. Default is 10.')

    parser.add_argument('--iterations',
                        type = int,
                        default = 10,
	                help = 'Number of boosting rounds. Default is 10.')

    parser.add_argument('--pruning-threshold',
                        type = int,
                        default = 10,
	                help = 'Co-occurence pruning rule. Default is 10.')

    
    return parser.parse_args()

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),v] for k,v in args.iteritems()])
    print t.draw()

def simple_print(name,value):
    """
    Print a loss value in a text table.
    :param name: Name of loss.
    :param loss: Loss value.
    """
    print("\n")
    t = Texttable() 
    t.add_rows([[name, value]])
    print t.draw()

def sampling(probs):
    """
    Dictionary key samplin -- the sampling probability of a key is proportional to the value.
    :param probs: Probability distribution over keys in a dictionary.
    :return key: Sampled key.
    """
    prob = random.random()
    score = 0
    for key, value in probs.items():
        score = score + value
        if prob <= score:
            return key
    assert False, 'unreachable'

def read_matrix(path):
    """
    Read the sparse target matrix which is in (user,item,score) csv format.
    :param path: Path to the dataset.
    :return scores: Sparse matrix returned.
    """
    dataset = pd.read_csv(path).values.tolist()
    index_1 = map(lambda x: x[0], dataset)
    index_2 = map(lambda x: x[1], dataset)
    scores = map(lambda x: x[2], dataset)
    shape = (max(index_1) + 1,max(index_2)+1)
    scores = sparse.csr_matrix(sparse.coo_matrix((scores,(index_1,index_2)),shape=shape,dtype=np.float32))
    return scores

class DeepWalker:
    """
    DeepWalker class for target co-occurence matrix approximation.
    """
    def __init__(self,args):
        """
        Initialization method which reads the arguments.
        :param args: Arguments object.
        """
        self.args = args
        self.graph = nx.from_edgelist(pd.read_csv(args.input_path).values.tolist())
        self.shape = (len(self.graph.nodes()), len(self.graph.nodes()))
        self.do_walks()
        self.do_processing()
        
    def do_a_walk(self, node):
        """
        Doing a single random walk from a source
        :param node: Source node.
        :return nodes: Truncated random walk
        """
        nodes = [node]
        while len(nodes)<self.args.walk_length:
            nebs = nx.neighbors(self.graph,nodes[-1])
            if len(nebs) > 0:
                nodes = nodes + random.sample(nebs,1)
            else:
                break
        return nodes

    def do_walks(self):
        """
        Doing a fixed number of random walks from each source.
        """
        self.walks = []
        
        for iteration in range(self.args.number_of_walks):
            print("\nRandom walk run: "+str(iteration+1)+"/"+str(self.args.number_of_walks)+".\n")
            for node in tqdm(self.graph.nodes()):
                walk = self.do_a_walk(node)
                self.walks.append(walk)

    def processor(self,walk):
        """
        Extracting the source-neighbor pairs.
        :param walk: Random walk processed.
        :return pairs: Source-target pairs.
        """
        pairs = []
        for omega in range(1,self.args.window_size+1):
            sources = walk[0:len(walk)-omega]
            nebs = walk[omega:]
            pairs = pairs + [(source,nebs[i]) for i, source in enumerate(sources)] +[(nebs[i],source) for i, source in enumerate(sources)]
                

        return pairs

    def do_processing(self):
        """
        Processing each sequence to create the sparse target matrix. Prunning the matrix for low occurence pairs.
        """
        self.container = []
        print("\nProcessing walks.\n")
        for walk in tqdm(self.walks):
            self.container.append(self.processor(walk))
        self.container = Counter([pair for pairs in self.container for pair in pairs])
        self.container = {k:v for k,v in self.container.iteritems() if v > self.args.pruning_threshold}
        index_1 = map(lambda x: x[0],self.container.keys())
        index_2 = map(lambda x: x[1],self.container.keys())
        scores = self.container.values()
        self.A = sparse.csr_matrix(sparse.coo_matrix((scores,(index_1,index_2)),shape=self.shape,dtype=np.float32))
   

    
