import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Create Market Dataset')
parser.add_argument('--path', metavar='DIR',
                    help='path to market dataset')

def main():
    args = parser.parse_args()
    path = args.path + '/bounding_box_train'
    new_path = args.path + '/train_folder'
    files = os.listdir(path)
    for i in files:
        subdir = i[:4]
        sub_path = os.path.join(new_path, subdir)
        if os.path.isdir(sub_path) is False:
            os.makedirs(sub_path)
        shutil.copy(os.path.join(path,i), sub_path)


if __name__ == '__main__':
    main()
