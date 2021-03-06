# coding=utf-8
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--dataset',
                        type=str,
                        help='Name of dataset, omniglot/mini_imagenet',
                        default='omniglot')

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='/home/misc/RnD_project/Third_expt/output')

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=50)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model',
                        default=0.0001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step',
                        default=30)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma',
                        default=0.5)


    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch',
                        default=100)

    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training',
                        default=20)

    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training',
                        default=5)

    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training',
                        default=15)

    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation',
                        default=5)

    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation',
                        default=5)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation',
                        default=15)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)
    parser.add_argument('-nI', '--num_inner',
                        type=int,
                        help='number of steps of inner loop, default = 1',
                        default=1)
    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')

    return parser
