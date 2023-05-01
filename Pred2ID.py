import os, sys, requests

# Constructs the dictionary of IDs to predictions
biodct = {}
# !rm -rf 'data/input/.DS_Store'
for file in os.listdir('data/input/'):
    with open('data/input/'+file) as f:
        print(file)
        lines = [line.rstrip('\n') for line in f]
        for line in lines:
            if len(line.split('\t'))>1:
                biodct[line.split()[3]] = line.split()[-1]

# Writes the ID's according to the predictions to a file
with open('data/output/ID2Pred.txt', 'w') as f1:
    with open('data/output/BioRED_Test.PubTator') as f2:
        lines = [line.rstrip('\n') for line in f2]
        for line in lines:
            if len(line.split('\t')) > 1:
                try:
                    line = line + '\t' + biodct[line.split()[3]]
                except:
                    pass
            f1.write(line + '\n')
    f2.close
f1.close