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
    test_data = rotate_point_cloud(test_data, angle_degrees=45, axis='x')
    
    # ------ TO DO: Make Prediction ------
    # test_data = test_data.to(args.device).float()
    # test_label = test_label.to(args.device).long()
    
    num_batch = (test_data.shape[0] // 40)+1
    pred_label = []
    
    for i in tqdm(range(num_batch)):
        output = model(test_data[i*40: (i+1)*40].to(args.device))
        prediction = list(output.max(dim=1)[1])
        pred_label.extend(prediction)
        
    # output = model(test_data).to(args.device)
    # pred_label = output.max(dim=1)[1]
    # test_label = test_label.cpu()
    
    # Compute Accuracy
    pred_label = torch.Tensor(pred_label).cpu()
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))
    # test_label = test_label.to(args.device).long()
    
    # Visualize Classification Result (Pred VS Ground Truth)
    print("Creating Visualization")
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, args.i), args.device)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}_{}.gif".format(args.output_dir, args.exp_name, args.i), args.device)

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

#Q3
# test accuracy: 0.7187827911857293
# Creating Visualization
# Incorrect labels:  [2, 3, 4, 9, 12, 13, 22, 24, 29, 30, 32, 35, 
#                     36, 40, 41, 43, 45, 47, 49, 51, 54, 55, 56, 57, 
#                     58, 64, 65, 67, 74, 77, 79, 80, 81, 84, 88, 90, 
#                     93, 96, 99, 100, 105, 106, 109, 110, 115, 116, 
#                     121, 122, 123, 127, 130, 136, 137, 138, 142, 145, 
#                     148, 149, 150, 151, 153, 155, 157, 160, 164, 165, 169, 
#                     170, 172, 173, 175, 180, 187, 189, 190, 192, 193, 197, 198, 
#                     200, 203, 204, 205, 206, 214, 216, 218, 219, 222, 225, 229, 232, 
#                     234, 235, 237, 242, 244, 249, 252, 254, 255, 256, 259, 268, 269, 273, 
#                     276, 280, 282, 283, 284, 289, 290, 294, 295, 296, 303, 306, 308, 309, 311, 
#                     312, 313, 315, 320, 322, 326, 328, 330, 331, 332, 335, 337, 339, 346, 349, 351,
#                     355, 356, 357, 358, 360, 361, 363, 364, 365, 368, 372, 377, 378, 380, 384, 385, 386, 
#                     387, 390, 391, 392, 395, 399, 406, 407, 411, 413, 414, 416, 417, 420, 421, 423, 427, 
#                     429, 435, 442, 443, 445, 446, 448, 449, 451, 453, 458, 459, 463, 468, 470, 473, 475, 487, 
#                     494, 496, 501, 502, 504, 506, 508, 510, 513, 514, 518, 519, 520, 522, 527, 534, 538, 539, 541, 
#                     543, 545, 546, 548, 549, 551, 554, 556, 557, 558, 564, 567, 572, 576, 578, 581, 583, 584, 585, 589, 
#                     595, 596, 601, 603, 605, 607, 611, 612, 615, 619, 621, 629, 631, 651, 664, 670, 673, 685, 699, 707, 708, 
#                     716, 758, 783, 786, 787, 796, 806, 827, 846, 859, 864, 868, 869, 893, 904, 916, 928, 946, 948]