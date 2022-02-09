import os
import glob


def load_source_images(data_path):
    if 'Lytro' in data_path.split('/'):
        files_source1 = glob.glob(os.path.join(data_path, '*-A.jpg'))
        files_source1.sort()
        files_source2 = glob.glob(os.path.join(data_path, '*-B.jpg'))
        files_source2.sort()
    else:
        # files = os.listdir(data_path)
        with open(os.path.join(data_path, 'list.txt'), 'r') as f:
            files = f.read().rstrip().split('\n')
        files_source1 = []
        files_source2 = []
        for file in files:
            files_source1.append(os.path.join(data_path, file, file+'_1.png'))
            files_source2.append(os.path.join(data_path, file, file+'_2.png'))
    return files_source1, files_source2

