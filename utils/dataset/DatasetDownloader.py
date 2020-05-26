import torchtext
import os
from os import path

class DatasetDownloader:

    def __init__(self):
        super(DatasetDownloader, self)

    def downloadDataset(self, dataset_args):
        if not path.exists(dataset_args['path']):
            os.mkdir(dataset_args['path'])

        if not path.exists(dataset_args['path'] + "/" + dataset_args['zip_file'])  :
            torchtext.utils.download_from_url(url=dataset_args['url'], path=dataset_args['path'] + "/" + dataset_args['zip_file'] , overwrite=True)


    def extractdataset(self, dataset_args):
        if not path.exists(dataset_args['path'] + "/" + dataset_args['extract_folder']):
            torchtext.utils.extract_archive(dataset_args['path'] + "/" + dataset_args['zip_file'], dataset_args['path'], overwrite=True)