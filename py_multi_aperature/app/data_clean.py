clean = False

data = []
file_name = "20220829_121237_DATA copy.csv"
with open(file_name, 'r') as f:
    for line in f.readlines():
        line = line.replace('\n','')
        if line == "Target Position Index,Asterism Index,Target Mode Index,Tomography Output":
            clean = True
        elif clean == True:
            line = line.replace(' ',',').replace(',,', ',').replace('[,', '[')
            if line[0] == ' ':
                line = line.replace(' ',',')
        if line[0] == ',':
            data[-1] += line
        else:
            data.append(line)

with open("CLEAN_" + file_name, 'w') as f:
    for line in data:
        f.write(line + '\n')