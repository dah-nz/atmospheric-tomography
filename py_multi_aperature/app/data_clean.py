clean = False

data = []
file_name = "TEST_20230327_120334.data"
with open(file_name, 'r') as f:
    for line in f.readlines():
        line = line.replace('\n','')
        if line == "Target Position Index,Asterism Index,Screen Index,Tomography Output":
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