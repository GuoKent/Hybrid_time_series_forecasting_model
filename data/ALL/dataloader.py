import numpy as np
import csv

temp = ['0C', '10C', '25C', '30C', '40C', '50C', 'RT']

for i in range(7):
    DST = 'DST_' + temp[i]
    FUDS = 'FUDS_' + temp[i]
    US06 = 'US06_' + temp[i]
    data_DST = np.loadtxt(open(f"../{temp[i]}/{DST}.csv", "rb"), delimiter=",", skiprows=1, usecols=[0, 1, 2, 3, 4])
    data_FUDS = np.loadtxt(open(f"../{temp[i]}/{FUDS}.csv", "rb"), delimiter=",", skiprows=1, usecols=[0, 1, 2, 3, 4])
    data_US06 = np.loadtxt(open(f"../{temp[i]}/{US06}.csv", "rb"), delimiter=",", skiprows=1, usecols=[0, 1, 2, 3, 4])
    '''data_DST = open(f"../{temp[i]}/{DST}.csv", mode='r', encoding='utf8')
    data_FUDS = open(f"../{temp[i]}/{FUDS}.csv", mode='r', encoding='utf8')
    data_US06 = open(f"../{temp[i]}/{US06}.csv", mode='r', encoding='utf8')'''
    with open(r'./DST_ALL2.csv', mode='a', newline='', encoding='utf8') as dst:
        w = csv.writer(dst)
        for j in data_DST:
            w.writerow(j)
    with open(r'./FUDS_ALL2.csv', mode='a', newline='', encoding='utf8') as fuds:
        w = csv.writer(fuds)
        for j in data_FUDS:
            w.writerow(j)
    with open(r'./US06_ALL2.csv', mode='a', newline='', encoding='utf8') as us:
        w = csv.writer(us)
        for j in data_US06:
            w.writerow(j)
