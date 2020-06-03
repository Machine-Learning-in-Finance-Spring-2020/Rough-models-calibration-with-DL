1)  You will need this file to run the notebook:

    https://drive.google.com/file/d/1p2wncAbgw6VNqWk7kchIYq-DFbPOPC1u/view?usp=sharing



    This file contains an array of implied volatilities of 100k randomly generated heston models, which the neural network is trained on. 
    It is the target data in the training process. Its shape is (10^6, 143). You could generate this yourself by uncommenting the line in block [6]: 

    h.create_training_data(100000,seed=1)

    It takes 5 hours on my computer.
    
2)  In the notebook, the prices of eu call opts are calculated with quantlib. From these prices the implied volatilities
are calculated with the formula provided in "An Explicit Implied Volatility Formula" by Dan Stefanica and Rados Radoicic:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2908494

