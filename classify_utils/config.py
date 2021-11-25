import argparse

def obtain_classify_args():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--model', type=str, default='googlenet',
                        choices=['lenet', 'googlenet', 'mobilenet'],
                        help='model name (default: mobilenet)')

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist'],
                        help='dataset name (default: mnist)')

    args = parser.parse_args()
    return args