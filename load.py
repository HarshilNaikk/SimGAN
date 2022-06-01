from matplotlib.pyplot import axis
import pandas as pd
import os
import numpy as np

class Load:
    def __init__(self):
        self.filenames = []
    
    def get_files(self, path, num_trajs):
        self.filenames = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
        # print(self.filenames)
        self.filenames = self.filenames[:num_trajs]
    
    def get_from_csv(self, filename, traj_length = None):
        df = pd.read_csv(filename)
        if df.shape[1] == 3:
            df.columns = ["x1", "x2", "u"]
        if df.shape[1] == 2:
            df.columns = ["c1", "c2"]
        # df.drop("t", axis=1, inplace=True)
        # # Remove human data because prediction is form leader and the follower
        # df.drop("v_human", axis=1, inplace=True)
        # df.drop("x_human", axis=1, inplace=True)
        # if traj_length:
        #     return df.iloc[:traj_length, :]
        # else:
        # print(np.shape(df))
        return df
    
    def get_data(self, dir_path, num_trajs, traj_length = None):
        print("Getting Data...")
        self.get_files(dir_path, num_trajs)
        df = pd.DataFrame()
        print("Preparing Data...")
        i = 1
        for filename in self.filenames:
            data = self.get_from_csv(filename, traj_length)
            # sas = self.get_sas(data)
            df = df.append(data,)
            if i % 10 == 0:
                print(f'Processed data for {i} file(s)')
            i += 1
        return df
    
    def get_sas(self, data):
        sas = pd.concat([data, data.shift(-1)], axis=1)
        sas = sas.iloc[:-1, :-1]
        return sas