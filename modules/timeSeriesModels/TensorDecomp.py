# Standard Imports
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mat4py

# Tensor Decompositions
import tensorly as tl
from tensorly.decomposition import parafac






class TensorDecomp:

    def __init__(self, num_dims, data_path):
        """
        num_dim = 2 or 3
        """
        self.num_dims = num_dims
        self.data_path = data_path
        pass
    
    
    
    def fit(self, data_type, sensor_nums):
        """
        Run either 2D or 3D decomposition. 
        
        data_type = 'Building' or 'Aircraft'
        sensor_nums = list of integers
        
        Data is stacked differently in both 
        2D and 3D scenarios. 
        
        N.B. Weird .fit() method because don't
        ask user for X,y just one of two analysis
        options.
        """
        self.data_type = data_type
        self.sensor_nums = sensor_nums
        X_list, y_list = self._collect_data()
        
        if self.num_dims == 2:
            self.arr_stacked_X = np.vstack(X_list)
            self.arr_stacked_y = np.vstack(y_list)

            tensor = tl.tensor(self.arr_stacked_X)
            weights, factors = parafac(tensor, rank=self.num_dims)

            # Joint Experiment - Sensor Space
            self.dim1_x = np.array([factors[0][i][0] for i in range(len(factors[0]))])
            self.dim1_y = np.array([factors[0][i][1] for i in range(len(factors[0]))])

            # Frequency Space - NOT USEFUL
            self.dim2_x = np.array([factors[1][i][0] for i in range(len(factors[1]))])
            self.dim2_y = np.array([factors[1][i][1] for i in range(len(factors[1]))])
        
        elif self.num_dims == 3:
            self.arr_stacked_X = np.dstack(X_list) # 3D stacking -- Events x Freq x SensorNum
            self.arr_y = np.array(y_list[0]) # Choose any list object arbitrarily

            tensor = tl.tensor(self.arr_stacked_X)
            weights, factors = parafac(tensor, rank=self.num_dims)

            # Event Space
            self.dim1_x = np.array([factors[0][i][0] for i in range(len(factors[0]))])
            self.dim1_y = np.array([factors[0][i][1] for i in range(len(factors[0]))])

            # Frequency Space - NOT USEFUL
            self.dim2_x = np.array([factors[1][i][0] for i in range(len(factors[1]))])
            self.dim2_y = np.array([factors[1][i][1] for i in range(len(factors[1]))])

            # Sensor Space
            dim3_x = np.array([factors[2][i][0] for i in range(len(factors[2]))])
            dim3_y = np.array([factors[2][i][1] for i in range(len(factors[2]))])
            # Modify for plotting clarity (add randomness for vertical separation)
            self.dim3_x = abs(np.power(dim3_x,-2))+np.random.rand()*2
            self.dim3_y = abs(np.power(dim3_y+np.random.rand(),-2))+np.random.rand()*5+3 
            
        return [self.dim1_x,self.dim1_y] if self.num_dims==2 else [[self.dim1_x,self.dim1_y],
                                                                   [self.dim3_x,self.dim3_y]]

    
    
    def _collect_data(self):
        """
        Reading data in and assembling
        into list.
        """
        X_list = []
        y_list = []
        
        for i in self.sensor_nums:
            if self.data_type == 'Building':
                str_in = self.data_path + f'Building_Sensor{i}.mat'
                data_in = mat4py.loadmat(str_in)
                
                X_list.append(data_in['X'])
                y_temp = np.array(data_in['y']).flatten()
                # Label increasing damage levels
                y_temp[25:29] = 1
                y_temp[29:32] = 2
                y_temp[33:36] = 3
                y_temp[37:40] = 4
                y_list.append(list(y_temp))
                
            if self.data_type == 'Aircraft':
                str_in = self.data_path + f'Aircraft_Sensor{i}.mat'
                data_in = mat4py.loadmat(str_in)
                
                X_list.append(data_in['X'])
                y_temp = np.array(data_in['y']).flatten()
                
                y_temp[0] = 0 # Take-off
                y_temp[1] = 1 # Climb
                y_temp[2] = 2 # Climb Damage
                y_list.append(list(y_temp))
        
        return X_list, y_list
                
                
            
    def plot(self):
        
        font_kwargs = {'size': 18,
                      'weight': 'bold'}
        plt.rc('font', **font_kwargs)
        
        if self.num_dims == 2:
            plt.figure(figsize=(12,12))
            y_labels = self.arr_stacked_y.flatten()
            
            if self.data_type == "Building":
                plt.scatter(self.dim1_x[y_labels==0], 
                            self.dim1_y[y_labels==0], 
                            s=60)
                plt.scatter(self.dim1_x[y_labels==1], 
                            self.dim1_y[y_labels==1],  
                              s=60)
                plt.scatter(self.dim1_x[y_labels==2], 
                            self.dim1_y[y_labels==2],  
                              s=60)
                plt.scatter(self.dim1_x[y_labels==3], 
                            self.dim1_y[y_labels==3],  
                              s=60)
                plt.scatter(self.dim1_x[y_labels==4], 
                            self.dim1_y[y_labels==4],  
                              s=60)
                plt.xlabel('Magnitude')
                plt.ylabel('Magnitude')
                plt.legend(['Healthy','Damage Level 1',
                           'Damage Level 2', 'Damage Level 3',
                           'Damage Level 4'])

            if self.data_type == "Aircraft":
                plt.scatter(self.dim1_x[y_labels==0],self.dim1_y[y_labels==0], s=60)
                plt.scatter(self.dim1_x[y_labels==1], self.dim1_y[y_labels==1],  s=60)
                plt.scatter(self.dim1_x[y_labels==2], self.dim1_y[y_labels==2],  s=60)
                plt.xlabel('Magnitude')
                plt.ylabel('Magnitude')
                plt.legend(['Healthy Take-Off','Healthy Climb',
                            'Damage Climb'])

            
        elif self.num_dims == 3:
            fig, axs = plt.subplots(2, figsize=(12,15))
            y_labels = self.arr_y.flatten()

            if self.data_type == "Building":
                axs[0].scatter(self.dim1_x[y_labels==0], 
                            self.dim1_y[y_labels==0], 
                            s=60)
                axs[0].scatter(self.dim1_x[y_labels==1], 
                            self.dim1_y[y_labels==1],  
                              s=60)
                axs[0].scatter(self.dim1_x[y_labels==2], 
                            self.dim1_y[y_labels==2],  
                              s=60)
                axs[0].scatter(self.dim1_x[y_labels==3], 
                            self.dim1_y[y_labels==3],  
                              s=60)
                axs[0].scatter(self.dim1_x[y_labels==4], 
                            self.dim1_y[y_labels==4],  
                              s=60)
                axs[0].set_xlabel('Magnitude')
                axs[0].set_ylabel('Magnitude')
                axs[0].legend(['Healthy','Damage Level 1',
                               'Damage Level 2', 'Damage Level 3',
                               'Damage Level 4'])
                axs[0].set_title('Event Space')

                axs[1].scatter(self.dim3_x, self.
                               dim3_y,s=60)
                axs[1].set_xlabel('Magnitude')
                axs[1].set_ylabel('Magnitude')
                axs[1].set_title('Sensor Space')

                idx = 0
                for i in self.sensor_nums:
                    axs[1].annotate(f'{i}', (self.dim3_x[idx], 
                                             self.dim3_y[idx]))
                    idx+=1
                plt.tight_layout()


            if self.data_type == "Aircraft":
                axs[0].scatter(self.dim1_x[y_labels==0], 
                               self.dim1_y[y_labels==0], 
                               s=60)
                axs[0].scatter(self.dim1_x[y_labels==1], 
                               self.dim1_y[y_labels==1],  
                               s=60)
                axs[0].scatter(self.dim1_x[y_labels==2], 
                               self.dim1_y[y_labels==2],  
                               s=60)
                axs[0].set_xlabel('Magnitude')
                axs[0].set_ylabel('Magnitude')
                axs[0].legend(['Healthy Take-Off','Healthy Climb',
                               'Damage Climb'])
                axs[0].set_title('Event Space')

                axs[1].scatter(self.dim3_x, self.
                               dim3_y,s=60)
                axs[1].set_xlabel('Magnitude')
                axs[1].set_ylabel('Magnitude')
                axs[1].set_title('Sensor Space')

                idx = 0
                for i in self.sensor_nums:
                    axs[1].annotate(f'{i}', (self.dim3_x[idx], 
                                             self.dim3_y[idx]))
                    idx+=1
                plt.tight_layout()












