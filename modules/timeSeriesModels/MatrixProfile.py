# Standard Imports
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mat4py

# Matrix Profilers
import stumpy

# Barycentre
from tslearn.barycenters import softdtw_barycenter, euclidean_barycenter

# Sampling Tickery
from scipy.interpolate import interp1d



class MatrixProfile:
    
    def __init__(self):
        """
        User not specify anything here. 
        Keep 'm' hidden.

        mp_dict[f'Sensor{self.sensor_nums[i]}'] = [mp_upscale, unanchored_chain_upscale, 
                                                      cac_upscale, np_seq_X[i,:]]
		mp_upscale -- matrix profile
		unanchored_chain_upscale -- location of motif 
		cac_upscale -- semantic segmenter
		np_seq_X[i,:] -- actual signal analyzed

		"upscale" beacuse signal was downsampled, analyzed, then upsampled
		(for speed boost). Downsampling done by factor of 2, otherwise matrix
		profiling deterioriates very quickly.
        """
        
        
    def fit(self, data_type, sensor_nums):
        """
        Fit Matrix Profiler to Sequence
        """
        self.data_type = data_type
        self.sensor_nums = sensor_nums
        
        # np.array of [sensors x signal length]
        self.np_seq_X = self._collect_data()

        # Fit Matrix profile signal sequence
        # Collect into dictionary
        if self.data_type == 'Building':
            self.m=940
            self.reduc_factor=2
        else:
            self.m=600
            self.reduc_factor=1 # Reduction is too damaging for aircraft sensors
        self.mp_dict = self._matrix_profile_sequence(self.np_seq_X, 
                                                     m=self.m, 
                                                     reduc_factor=self.reduc_factor)
 

    def plot(self):
    
        font_kwargs = {'size': 18,
                          'weight': 'bold'}
        plt.rc('font', **font_kwargs)
        fig, axs = plt.subplots(4, sharex=True, gridspec_kw={'hspace':0.4},
                               figsize=(20,15))

        reduc_factor = 3 # Different than in _matrix_profile_sequence
        motif_list = []
        mp_list = []
        cac_list = []
        f_list = []
        motif_flag = True
        weight_list = []
        
        for sensor in self.mp_dict:
            # Collecting Dict Structures
            sensor_X = self.mp_dict[sensor][3]
            # Pick a random motif
            num = np.random.choice(len(self.mp_dict[sensor][1]))
            sensor_chain_val = self.mp_dict[sensor][1][num]
            
            sensor_motif = sensor_X[sensor_chain_val:sensor_chain_val+self.m]
            sensor_mp = self._moving_average(self.mp_dict[sensor][0],20) # Smoothed
            sensor_cac = self._moving_average(self.mp_dict[sensor][2],30) # Smoothed

            # Collecting Motif Structures
            if motif_flag:
                X_motif = np.arange(sensor_chain_val,sensor_chain_val+self.m)
                axs[0].plot(X_motif, sensor_motif,'b',linewidth=4) 
                axs[0].axis('off')
                axs[0].annotate('Significant Motif', (sensor_chain_val+self.m*1.1,
                                                      sensor_motif.mean()))
                motif_flag = False # Only one motif OK to plot (for now)

            # Plotting Raw Signals
            axs[1].plot(sensor_X,'k',alpha=0.4)

            # Plotting Matrix Profiles
            axs[2].plot(sensor_mp,'k',alpha=0.4)

            # Plotting Semantic Segmenter
            axs[3].plot(abs(1-sensor_cac),'k',alpha=0.4)

            motif_list.append(sensor_motif[::reduc_factor])
            mp_list.append(sensor_mp[::reduc_factor])
            cac_list.append(sensor_cac[::reduc_factor])

            if self.data_type == 'Building':
                # Weights for Euclidean Barycentre ('fancy weighted average')
                if sensor[-1] in ('1','2','3','4','5','6','7','8'):
                    weight_list.append(50)
                elif sensor[-1] in ('9','10','11','12','13','14','15','16'):
                     weight_list.append(25)
                else:
                    weight_list.append(5)
            else: # Aircraft
                if sensor[-1] in ('1','2'):
                    weight_list.append(50)
                elif sensor[-1] in ('4','5'):
                    weight_list.append(10)
                else:
                    weight_list.append(200)


        # Under-sampling (speeding up things)
        bary_mp = euclidean_barycenter(np.vstack(mp_list), weights=weight_list)
        Xs, f_mp = self._sampler(bary_mp.flatten())
        Xs_upscale = np.linspace(0,len(Xs)-1,len(sensor_mp))
        axs[2].plot(f_mp(Xs_upscale),'r', linewidth=3)

        bary_cac = euclidean_barycenter(np.vstack(cac_list), weights=weight_list)
        Xs, f_cac = self._sampler(bary_cac.flatten())
        Xs_upscale = np.linspace(0,len(Xs)-1,len(sensor_mp))
        axs[3].plot(self._moving_average(abs(1-f_cac(Xs_upscale)),100),'r', linewidth=3)


        if self.data_type == 'Building':
            # Personal Note: 13*470-940 = 5170
            rect = Rectangle((5170,1), 2000, 6, facecolor='darksalmon')
            axs[1].add_patch(rect)
            rect = Rectangle((7170,1), 2000, 6, facecolor='salmon')
            axs[1].add_patch(rect)
            rect = Rectangle((9170,1), 2000, 6, facecolor='tomato')
            axs[1].add_patch(rect)
            rect = Rectangle((11170,1), 2000, 6, facecolor='red')
            axs[1].add_patch(rect)

            axs[1].annotate('Damage Level 1', (5300,5.9), color='k',fontsize=14)
            axs[1].annotate('Damage Level 2', (7300,5.9), color='k',fontsize=14)
            axs[1].annotate('Damage Level 3', (9300,5.9), color='k',fontsize=14)
            axs[1].annotate('Damage Level 4', (11300,5.9), color='k',fontsize=14)
            axs[1].set_ylim([1.2,6.5])

            axs[1].set_title('Sensor Signals')
            axs[2].set_title('Matrix Profile')
            axs[3].set_title('Semantic Segmenter')
            axs[3].set_xlabel('Signal Counts')

        else:
            rect = Rectangle((2000,-2.5), 2000, 6, facecolor='lightgreen')
            axs[1].add_patch(rect)
            rect = Rectangle((4000,-2.5), 2000, 6, facecolor='red')
            axs[1].add_patch(rect)

            axs[1].annotate('Take-off Phase', (1000, 2.5), color='k',fontsize=14)
            axs[1].annotate('Climb Phase', (3000,2.5), color='k',fontsize=14)
            axs[1].annotate('Climb Phase (Damaged)', (4500,2.5), color='k',fontsize=14)

            axs[1].set_ylim([-2.5,3.2])

            axs[1].set_title('Sensor Signals')
            axs[2].set_title('Matrix Profile')
            axs[3].set_title('Semantic Segmenter')
            axs[3].set_xlabel('Signal Counts')
            
            
    
    
    def _matrix_profile_sequence(self, np_seq_X, m, reduc_factor=2):
        """
        Perform matrix profile against all the chosen sensors
        and their signals
        
        m controls sweep size
        reduc_factor > 2 results deterioriate
        """
        self.reduc_factor = reduc_factor
        
        mp_dict = {}
        for i in range(len(self.sensor_nums)):
            # Under-sampling 
            Xs,f = self._sampler(np_seq_X[i,:])
            Xs_truncate = Xs[::reduc_factor] # Reduce by factor
            f_red = f(Xs_truncate)

            # Matrix Profiling
            mp = stumpy.stump(f_red, m, normalize=False)
            # Chain Analysis
            all_chain_set, unanchored_chain = stumpy.allc(mp[:, 2], mp[:, 3])
            # Semantic Segmentation
            if self.data_type == 'Building':
                excl_factor = 0
                L = m
            else:
                excl_factor = 1
                L = int(m*1.5)
            cac, regime_locations = stumpy.fluss(mp[:, 1], L=L, n_regimes=5, excl_factor=excl_factor)

            # Over-sampling
            _, f_mp = self._sampler(mp[:,0])
            _, f_cac = self._sampler(cac)
            end = int(len(mp[:,0])-1)
            mp_upscale = f_mp(np.linspace(0,end,len(Xs)))
            cac_upscale = f_cac(np.linspace(0,end,len(Xs)))
            unanchored_chain_upscale = unanchored_chain * reduc_factor

            # Storing
            mp_dict[f'Sensor{self.sensor_nums[i]}'] = [mp_upscale, unanchored_chain_upscale, 
                                                      cac_upscale, np_seq_X[i,:]]

        return mp_dict  
    
    
    
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
        f = interp1d(Xs, np_seq_Xi, kind='cubic',fill_value='interpolate')
        return Xs,f


















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