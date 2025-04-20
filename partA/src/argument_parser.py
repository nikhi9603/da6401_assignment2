import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_Assignment2",
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="nikhithaa-iit-madras",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-bd", "--base_dir", type=str, default="inaturalist_12K",
                        help="Base directory where dataset (train/val folders) are present")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs to train neural network")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size used to train neural network")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "rmsprop", "adam"], default="sgd",
                        help="Choose one among these optimizers: ['sgd', 'rmsprop', 'adam']")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.9,
                        help="Momentum used by momentum and nag optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.9,
                        help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                        help="Beta1 used by adam and nadam optimizers")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                        help="Beta2 used by adam and nadam optimizers")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.00000001,
                        help="Epsilon used by optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0001,
                        help="Weight decay used by optimizers")
    parser.add_argument("-dp", "--dropout", type=float, default=0.0,
                        help="Dropout used in convolution neural network")
    parser.add_argument("-da", "--isDataAug", type=str, default="False",
                        help="Whether to use data augmentation or not")
    parser.add_argument("-bn", "--isBatchNormalization", type=str, default="False",
                        help="Whether to use batch normalization or not")
    parser.add_argument("-nf", "--num_filters", type=int, nargs=5,  default=[3, 3, 3, 3, 3],
                        help="Number of filters used in each convolution layer")
    parser.add_argument("-fsz", "--filter_sizes", type=int, nargs=5,  default=[32, 64, 64, 128, 256],
                        help="Size of filters in each convolution layer")
    parser.add_argument("-a", "--activation", type=str, choices=["ReLU", "SiLU", "GELU"], default="SiLU",
                        help="Choose one among these activation functions: ['ReLU', 'SiLU', 'GELU']")
    parser.add_argument("-ndl", "--n_neurons_denseLayer", type=int, default=128,
                        help="Number of neurons in dense layer")
    
    return parser.parse_args()
