import glob
import os

xml_files = glob.glob('*.xml')
jpg_files = glob.glob('*.jpg')
lines = [line.rstrip('\n') for line in open('hoge')]
#print (lines)
print (xml_files)
deleted = 0
for xml_file in xml_files:
    if xml_file in lines:
        # Meaning file was processed so take a pass
        print ('file is present')
    else:
        # file was not present, just go ahead and delete xml & jpg file
        deleted += 1
        filename = xml_file.split('.')[0]
        jpg_file = filename + '.jpg'
        os.remove(xml_file)
        os.remove(jpg_file)
print ('Done... Deleted {}'.format(deleted))
