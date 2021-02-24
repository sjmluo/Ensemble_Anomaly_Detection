# Standard Imports
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mat4py

# Change Point Analyzer
import ruptures as rpt 

# Sampling Tickery
from scipy.interpolate import interp1d



class ChangePoint:
    
    def __init__(self):
        """
        User not specify anything here. 
        Keep 'min_size', and 'n_bkps' hidden.
        """   
        
        
    def fit(self,data_type, sensor_nums):
        """
        Perform a change point analysis on sequence
        """
        self.data_type = data_type
        self.sensor_nums = sensor_nums
        
        # np.array of [sensors x signal length]
        self.np_seq_X = self._collect_data()
        
        # Extract change points
        if data_type == 'Building':
            self.min_size = 400
        else:
            self.min_size = 200
        self.chg_pts = self._change_points(min_size = self.min_size)
        


    
        
    def plot(self):
        font_kwargs = {'size': 18,
                          'weight': 'bold'}
        plt.rc('font', **font_kwargs)
        fig, ax = plt.subplots(figsize=(15,8)) 
        
        for i in range(self.np_seq_X.shape[0]):
            linear_decrease = (-4/115)*self.np_seq_X.shape[0] + 119/115
            plt.plot(self.np_seq_X[i,:],'k',alpha=0.5*linear_decrease,
                     label="")
        
        if self.data_type == 'Building':
            # Personal Note: 13*470-940 = 5170   
            rect = Rectangle((5170,1), 2000, 6, facecolor='darksalmon',label="")
            ax.add_patch(rect)
            rect = Rectangle((7170,1), 2000, 6, facecolor='salmon',label="")
            ax.add_patch(rect)
            rect = Rectangle((9170,1), 2000, 6, facecolor='tomato',label="")
            ax.add_patch(rect)
            rect = Rectangle((11170,1), 2000, 6, facecolor='red',label="")
            ax.add_patch(rect)


            ax.annotate('Damage\n Level 1', (5300,6), color='k',fontsize=15)
            ax.annotate('Damage\n Level 2', (7300,6), color='k',fontsize=15)
            ax.annotate('Damage\n Level 3', (9300,6), color='k',fontsize=15)
            ax.annotate('Damage\n Level 4', (11300,6), color='k',fontsize=15)

            # Plotting change points
            relevant_chg_pts = self.chg_pts[self.chg_pts > 5170][:-1]
            plt.vlines(relevant_chg_pts,1.4,5.8,color='b', linestyle='dashed',
                      linewidth=5,label='Change Points')
            plt.legend()
            ax.set_ylim([1.2,6.5])
            ax.set_xlabel('Signal Counts')
            ax.set_ylabel('Signal Magnitude')
            
        else:  # Aircraft
            rect = Rectangle((2000,-2.8), 2000, 6, facecolor='lightgreen')
            ax.add_patch(rect)
            rect = Rectangle((4000,-2.8), 2000, 6, facecolor='red')
            ax.add_patch(rect)

            ax.annotate('Take-off Phase', (1000, 2.5), color='k',fontsize=14)
            ax.annotate('Climb Phase', (3000,2.5), color='k',fontsize=14)
            ax.annotate('Climb Phase (Damaged)', (4500,2.5), color='k',fontsize=14)

            # Plotting change points
            relevant_chg_pts = self.chg_pts[self.chg_pts > 2000][:-1]
            plt.vlines(relevant_chg_pts,-2.8,2.1,color='b', linestyle='dashed',
                      linewidth=5,label='Change Points')
            
            ax.set_ylim([-2.8,3.2])

            ax.set_title('Sensor Signals')
            ax.set_title('Matrix Profile')
            ax.set_title('Semantic Segmenter')
            ax.set_xlabel('Signal Counts')
        
        
        
        
    def _change_points(self, min_size, kernel='rbf', reduc_factor=3):
        """
        Perform rbf multi-variate change point analysis
        """
        
        signal_list = []
        for i in range(len(self.sensor_nums)):   
            # Under-sampling (for speed up)
            Xs,f = self._sampler(self.np_seq_X[i,:])
            Xs_truncate = Xs[::reduc_factor] # Reduce by factor
            signal_list.append(f(Xs_truncate).reshape(-1,1))
            
        signal_array = np.hstack(signal_list)
        chg_pt_analysis = rpt.KernelCPD(kernel="linear", min_size=min_size).fit(signal_array) 
        
        if self.data_type == 'Building':
            out = np.array(chg_pt_analysis.predict(n_bkps=5))*reduc_factor
        else: # Aircraft
            out = np.array(chg_pt_analysis.predict(n_bkps=3))*reduc_factor
        return out
            
        
                                                                       
        
        
    def _collect_data(self):
        """
        Reading data in and assembling
        into list.
        
        Signals will be appended into sequnce
        of healthy --> unhealty to see 
        damage progression.
        """
        X_list = []
        for i in self.sensor_nums:
            if self.data_type == 'Building':
                str_in = f'Building_Sensor{i}.mat'
                data_in = mat4py.loadmat(str_in)

                X = np.array(data_in['X'])
                y = np.array(data_in['y'])

                healthy_idx = np.argwhere(y==0)[:,0]
                healthy_idx_half = healthy_idx[::2] # Less computation
                unhealthy_idx = np.argwhere(y==1)[:,0]

                healthy_X = X[healthy_idx_half]
                unhealthy_X = X[unhealthy_idx]

                # pre-processing (nSensors x Signal Length)
                X_all = np.append(healthy_X.flatten(), unhealthy_X.flatten())
                X_all = self._moving_average(np.log(X_all),40)

                X_list.append(X_all)
                
            if self.data_type == 'Aircraft':
                str_in = f'Aircraft_Sensor{i}.mat'
                data_in = mat4py.loadmat(str_in)

                X = np.array(data_in['X'])
                y = np.array(data_in['y'])

                healthy_idx = np.argwhere(y==0)[:,0]
                unhealthy_idx = np.argwhere(y==1)[:,0]

                healthy_X = X[healthy_idx]
                unhealthy_X = X[unhealthy_idx]

                # pre-processing (nSensors x Signal Length)
                X_all = np.append(healthy_X.flatten(), unhealthy_X.flatten())
                X_all = self._moving_average(X_all,40) # Don't need moving average

                X_list.append(X_all)
 
        return np.array(X_list)

        
    
    def _moving_average(self,x, w):
        """
        Helper function
        """
        return np.convolve(x, np.ones(w), 'valid') / w   
    
    
    
    def _sampler(self,np_seq_Xi):
        """
        Helper function for down/upsampling
        signals (for increased speed).
        
        Not very elegant. 
        """
        
        Xs = np.arange(len(np_seq_Xi))
        f = interp1d(Xs, np_seq_Xi, kind='cubic',
                    fill_value='interpolate')
        return Xs,f   