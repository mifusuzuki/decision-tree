import numpy as np

class Dataset:
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.classes = np.array([])
        
    def load(self, filepath):
        x = []
        y = []
        
        for line in open(filepath):
            if line.strip() != '':
                row = line.split(',')
                x.append(row[:-1])
                y.append(row[-1].strip())
        x = np.array(x)
        self.x = x.astype(np.float)
        self.y = np.array(y)
        self.classes = np.unique(y)
        #print(self.x)
        #print(self.x.shape)
        #print(self.y)
        #print(self.y.shape)
        #print(self.classes)
        
        return self.x, self.y