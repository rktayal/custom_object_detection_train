import os
import sys
import argparse
import shutil

def read_imageset_file(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    return lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads ImageSet file and gets image & annotations files segregated. ')
    parser.add_argument('--annotations_dir',required=True, default='',  help='Specify directory containing annotations files')
    parser.add_argument('--img_dir',required=True, default='', help='Specify directory containing images')
    parser.add_argument('--output_dir',required=True, default='', help='Specify output directory for resulting files')
    args = vars(parser.parse_args())
    file_to_read = 'person_trainval.txt'

    # check for the existence of all the paths passed as argument
    if not all([os.path.exists(args['annotations_dir']), os.path.exists(args['img_dir']), os.path.exists(args['output_dir'])]):
        print ('one or more paths does not exists... Exiting')
        sys.exit(-1)

    lines = read_imageset_file(file_to_read)
    copied_files = 0
    print ('Copying...')
    for line in lines:
        values = line.split(' ')
        flag = values[-1]
        if flag == '1':
            # copy the image and corresponding anno. file
            img_file = os.path.join(args['img_dir'], values[0] + '.jpg')
            xml_file = os.path.join(args['annotations_dir'], values[0] + '.xml')
            shutil.copy(img_file, args['output_dir'])
            shutil.copy(xml_file, args['output_dir'])
            copied_files += 1
    print ('Done...')
    print ('Total images copied : {}'.format(copied_files))
