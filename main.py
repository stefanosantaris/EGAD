import argparse

from trainer.Trainer import Trainer
from utils.dataset.DatasetDownloader import DatasetDownloader
import pickle as pkl
import os
from os import path


dataset_dict = {
    'large': {
        'url':'https://researchlab.blob.core.windows.net/publications/2020/CIKM/datasets/LiveStream-16K.zip',
        'path':'data_large',
        'zip_file': 'dataset.zip',
        'extract_folder':'edgelists'
    },
    'medium': {
        'url':'https://researchlab.blob.core.windows.net/publications/2020/CIKM/datasets/LiveStream-6K.zip',
        'path':'data_medium',
        'zip_file': 'dataset.zip',
        'extract_folder':'edgelists'
    },
    'small': {
        'url':'https://researchlab.blob.core.windows.net/publications/2020/CIKM/datasets/LiveStream-4K.zip',
        'path':'data_small',
        'zip_file': 'dataset.zip',
        'extract_folder':'edgelists'
    }
}


def prepare_data(args):
    dataset_downloader = DatasetDownloader()
    dataset_downloader.downloadDataset(dataset_dict[args.dataset])
    dataset_downloader.extractdataset(dataset_dict[args.dataset])


def start_exp(args):
    trainer = Trainer(dataset_dict[args.dataset], args.cuda)
    results = trainer.train_model(args)
    if not path.exists('results'):
        os.mkdir('results')
    f = open("results/" +
             args.dataset +
             '_teacher_emb_' + str(args.teacher_embed_size) +
             '_teacher_heads_'+str(args.teacher_n_heads) +
             '_window_'+ str(args.window) +
             '_student_emb_' + str(args.student_emb) +
             '_student_heads_' + str(args.student_heads) +
             '_distillation_' + str(args.distillation) +
             '.pkl', "wb")
    pkl.dump(results, f)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='large',
                            help='Dataset File Name')
    parser.add_argument('--start_graph', type=int, default=0, help="Starting graph")
    parser.add_argument('--end_graph', type=int, default=7, help="Ending graph")
    parser.add_argument('--num_exp', type=int, default=1, help="Number of experiments")
    parser.add_argument('--teacher_embed_size', type=int, default=64, help="Teacher Embedding size")
    parser.add_argument('--window', type=int, default=3, help='Window for evolution')
    parser.add_argument('--teacher_n_heads', type=int, default=3, help="Number of Head Attention for Teacher")
    parser.add_argument('--dropout', type=float, default=0., help="Dropout")
    parser.add_argument('--alpha', type=float, default=0.2, help="LeakyRelu alpha")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--student_emb', type=int, default=16, help="Student embedding")
    parser.add_argument('--student_heads', type=int, default=1, help="Student number of head attention")
    parser.add_argument('--distillation', type=int, default=1, help="Distillation enabled")
    parser.add_argument('--ns', type=int, default=1, help="Number of negative samples")
    parser.add_argument('--cuda', type=int, default=0, help="CUDA SUPPORT (0=FALSE/1=TRUE)")


    args = parser.parse_args()
    prepare_data(args)
    start_exp(args)
