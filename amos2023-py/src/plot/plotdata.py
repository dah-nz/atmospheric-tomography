import matplotlib.pyplot as plt

data = []
keys = []

with open("./results/results.data", "r") as f:
    raw_data = f.readlines()

keys = raw_data[0].strip.split(',')

for l in raw_data[1:]:
    data.append(l.strip().split(','))