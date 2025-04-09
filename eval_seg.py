import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg
from tqdm import tqdm
import os
from pytorch3d.transforms import Rotate, axis_angle_to_matrix


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser

def rotate_point_cloud(batch_tensor, angle_degrees=45, axis='y'):
    angle_rad = torch.tensor(angle_degrees * np.pi / 180.0)
    axis_map = {'x': torch.tensor([1, 0, 0]),
                'y': torch.tensor([0, 1, 0]),
                'z': torch.tensor([0, 0, 1])}
    
    axis_vector = axis_map[axis].float()
    axis_angle = angle_rad * axis_vector  
    R = axis_angle_to_matrix(axis_angle)  
    return torch.matmul(batch_tensor, R.T)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    #Rotate point cloud
    # test_data = rotate_point_cloud(test_data, angle_degrees=45, axis='x')
    
    # ------ TO DO: Make Prediction ------
    print(test_data.shape[0])
    n = 20
    num_batch = (test_data.shape[0] // n)+1
    pred_label = torch.ones_like(test_label)
    # print(pred_label.shape)
    for i in tqdm(range(num_batch)):
        output = model(test_data[i*n: (i+1)*n].to(args.device))
        prediction = output.max(dim=2)[1]
        # prediction = torch.argmax(output, -1).cpu()
        # print(prediction.shape)
        # print(prediction)
        pred_label[i*n: (i+1)*n, :] = prediction

    
    
    pred_label = torch.Tensor(pred_label).cpu()
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # Visualize Segmentation Result (Pred VS Ground Truth)
    print("Creating Visualization")
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, args.i), args.device, args.num_points)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}_{}.gif".format(args.output_dir, args.exp_name, args.i), args.device, args.num_points)
    
    #Finding out which samples have low accuracy
    low_acc_file = os.path.join(args.output_dir, f"low_accuracy_{args.exp_name}.txt")
    high_acc_file = os.path.join(args.output_dir, f"high_accuracy_{args.exp_name}.txt")
    accuracy = []
    
    with open(low_acc_file, "w") as low_f, open(high_acc_file, "w") as high_f:
        low_f.write("Index\tAccuracy\n")
        high_f.write("Index\tAccuracy\n")

        for idx in tqdm(range(len(test_label))):
            test_accuracy = pred_label[idx].eq(test_label[idx].data).cpu().sum().item() / (test_label[idx].reshape((-1,1)).size()[0])
            if test_accuracy < 0.6:
                low_f.write(f"{idx}\t{test_accuracy:.4f}\n")
            elif test_accuracy > 0.6:
                high_f.write(f"{idx}\t{test_accuracy:.4f}\n")
                
    print(f"Saved low accuracy samples to: {low_acc_file}")
    print(f"Saved high accuracy samples to: {high_acc_file}")
                
    
    
     
    #q3 - rotate
    #test accuracy: 0.6465753646677471
    
    #q3- num_points
    #test accuracy: 0.8839967585089141
    