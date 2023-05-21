import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('architecture', type=str,
                    help='Model Architecture')
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='Epochs')
parser.add_argument('-b', '--batch_size', type=int, default=8,
                    help='Batch Size')
parser.add_argument('-l', '--learning_rate', type=float, default=0.0001,
                    help='Learning Rate')
parser.add_argument('-w', '--weighted_loss', action='store_true',
                    help='Weighted Loss')
parser.add_argument('-a', '--augment', action='store_true',
                    help='Augmentation of Data')
parser.add_argument('-s', '--scheduler',  action='store_true',
                    help='Learning Rate Scheduler')
parser.add_argument('-n', '--num_workers',  type=int, default=0,
                    help='Number of GPU Workers (Processes)')

parser.add_argument('-c', '--config', default=False,
                    help="Config name under 'configs/")
parser.add_argument('-D', '--data_path', default=False,
                    help="Path to locate (or download) data from")
parser.add_argument('-N', '--dataset_name', default=False,
                    help="Name of the dataset")
parser.add_argument('-S', '--save_path', default=False,
                    help="Path to save model weights to")
parser.add_argument('-L', '--load_path', default=False,
                    help="Path to load model weights from")
parser.add_argument('-P', '--plot_path', default=False,
                    help="Path to dump experiment plots")


parser.add_argument('--download', action='store_true',
                    help="Download dataset if it doesn't exist")
parser.add_argument('--pretrained', action='store_true',
                    help="Pretrained Weights")