import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_seg
from tqdm import tqdm
from pytorch3d.transforms import Rotate, axis_angle_to_matrix

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
    
    
    return parser


#Rotate the point cloud
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

    #Rotate point cloud
    # test_data = rotate_point_cloud(test_data, angle_degrees=45, axis='x')
    
    # ------ TO DO: Make Prediction ------
     
    num_batch = (test_data.shape[0] // 40)+1
    pred_label = []
    
    for i in tqdm(range(num_batch)):
        output = model(test_data[i*40: (i+1)*40].to(args.device))
        prediction = list(output.max(dim=1)[1])
        pred_label.extend(prediction)
        
    # Compute Accuracy
    pred_label = torch.Tensor(pred_label).cpu()
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))
    
    # Visualize Classification Result (Pred VS Ground Truth)
    print("Creating Visualization")
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, args.i), args.device, args.num_points)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}_{}.gif".format(args.output_dir, args.exp_name, args.i), args.device, args.num_points)

    #Finding out which labels were incorrect
    test_label = test_label.cpu().numpy()
    pred_label = pred_label.cpu().numpy()

    incorrect_labels = []
    for i in range(len(test_label)):
        if test_label[i] != pred_label[i]:
            incorrect_labels.append(i)
    
    # print("Incorrect labels: ", incorrect_labels)
    
    
#test accuracy: 0.974816369359916
# Creating Visualizer
# Incorrect labels:  [406, 619, 621, 631, 651, 664, 670, 671, 673, 685, 690, 699, 707, 708, 714, 716, 750, 777, 787, 827, 859, 864, 916, 922]


#Q3- rotation
#test accuracy- 0.7313746

#Q3- num_points
#test accuracy: 0.9716684155299056