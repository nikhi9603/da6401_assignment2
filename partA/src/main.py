from train_local import *
from argument_parser import *
import libraries

if __name__=="__main__":
  args = parse_arguments()
  trainNeuralNetwork_local(args)
