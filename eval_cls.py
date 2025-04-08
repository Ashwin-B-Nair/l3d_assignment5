import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_seg
from tqdm import tqdm

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32)
    
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model(num_classes=args.num_cls_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    # ------ TO DO: Make Prediction ------
    test_data = test_data.to(args.device).float()
    test_label = test_label.to(args.device).long()
    batch_size = args.batch_size
    num_batch = (test_data.shape[0] // batch_size)+1
    pred_label = []
    
    for i in tqdm(range(num_batch)):
        output = model(test_data[i*batch_size: (i+1)*batch_size].to(args.device))
        prediction = output.max(dim=1)[1]
        prediction = list(prediction)
        pred_label.extend(prediction)
        
    # output = model(test_data).to(args.device)
    # pred_label = output.max(dim=1)[1]

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # Visualize Classification Result (Pred VS Ground Truth)
    
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)

