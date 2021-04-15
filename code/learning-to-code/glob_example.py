import glob
glob.glob('*.py')


for file in glob.iglob('**/*.py', recursive=True):
    print(file)