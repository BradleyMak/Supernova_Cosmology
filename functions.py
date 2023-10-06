import numpy as np

#function to read data from txt file
def read_data(filename):
    alldata = np.loadtxt(filename, dtype='str', comments='#')
    print(alldata)

read_data('sn_data.txt')
