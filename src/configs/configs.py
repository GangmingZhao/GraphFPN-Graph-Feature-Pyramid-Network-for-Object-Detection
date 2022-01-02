import os, sys
import argparse

from easydict import EasyDict as edict

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("Graph-FPN"):
    src_dir = os.path.dirname(src_dir) 
if src_dir not in sys.path:
    sys.path.append(src_dir)

def parse_configs():
    parser = argparse.ArgumentParser(description='The Implementation using tensorflow')
    parser.add_argument('--seed', type=int, default=2020,
                        help='re-produce the results with seed random')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('--root_dir', type=str, default=src_dir, metavar='PATH',
                        help='The ROOT working directory')
    parser.add_argument('--r50_nograph_weight', type=str, default="data_demo/data", metavar='PATH',
                        help='Retinanet weight path')
    parser.add_argument('--r50_graph_weight', type=str, default="checkpoint/", metavar='PATH',
                        help='Graph Retinanet weight path')
    parser.add_argument('--result_dir', type=str, default=os.path.join(src_dir, "results"), metavar='PATH',
                        help='The ROOT working directory')
    ####################################################################
    ##############     Model configs            ########################
    ####################################################################
    parser.add_argument('--backbone', type=str,
                        help='The name of the model backbone')
    parser.add_argument('--Arch', type=str, metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--no_graph', action = "store_true",
                        help='use or not use graph')
    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--hflip_prob', type=float, default=0.5,
                        help='The probability of horizontal flip')
    parser.add_argument('--no-val', action='store_true',
                        help='If true, dont evaluate the model on the val set')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 16), this is the total'
                             'batch size of all GPUs on the current node when using'
                             'Data Parallel or Distributed Data Parallel')
    parser.add_argument('--print_freq', type=int, default=50, metavar='N',
                        help='print frequency (default: 50)')
    parser.add_argument('--tensorboard_freq', type=int, default=50, metavar='N',
                        help='frequency of saving tensorboard (default: 50)')
    parser.add_argument('--checkpoint_freq', type=int, default=3, metavar='N',
                        help='frequency of saving checkpoints (default: 5)')
    ####################################################################
    ##############     Training strategy            ####################
    ####################################################################

    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='the starting epoch')
    parser.add_argument('--num_epochs', type=int, default=50, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr_type', type=str, default='cosin',
                        help='the type of learning rate scheduler (cosin or multi_step or one_cycle)')
    parser.add_argument('--lr', type=float, default = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05], metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--lr_boundaries', type=float, default = [125, 250, 500, 240000, 360000], metavar='MIN_LR',
                        help='minimum learning rate during training')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0., metavar='WD',
                        help='weight decay (default: 0.)')
    parser.add_argument('--optimizer_type', type=str, default='Adam', metavar='OPTIMIZER',
                        help='the type of optimizer, it can be sgd or adam')

    ####################################################################
    ##############     Evaluation configurations     ###################
    ####################################################################
    parser.add_argument('--resume_path', type=str, default=None, metavar='PATH',
                        help='the path of the resumed checkpoint')
    parser.add_argument('--eval_epoch', type=int, default=3,
                        help='do evaluation in each epoch')                
    configs = edict(vars(parser.parse_args()))

    configs.num_classes = 80
    configs.loss = "RetinaNetLoss"
    configs.data = os.path.join(configs.root_dir, configs.r50_nograph_weight)
    configs.backbone = "Resnet50"
    if configs.no_graph:
        configs.Arch = "Retinanet" 
        configs.weight = os.path.join(configs.root_dir, configs.r50_nograph_weight)
    else:
        configs.Arch = "Graph_Retinanet" 
        configs.weight = os.path.join(configs.root_dir, configs.r50_graph_weight)
    return configs

    