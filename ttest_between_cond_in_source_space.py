

import mne
import os
import os.path as op
import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats import multitest as mul
mne.viz.set_3d_options(antialias=False)

os.environ['SUBJECTS_DIR'] = '/media/kristina/storage/interoception/freesurfer'
subjects_dir = '/media/kristina/storage/interoception/freesurfer'

subjects = ['ICS01','ICS03','ICS04','ICS05','ICS06','ICS07','ICS08','ICS09','ICS10','ICS12',
            'ICS13','ICS15', 'ICS17','ICS18','ICS19','ICS20','ICS21','ICS22',
            'ICS23','ICS24','ICS25','ICS26','ICS27','ICS28','ICS29','ICS30','ICS31']

def signed_p_val(t, p_val):
    if t >= 0:
        return 1 - p_val
    else:
        return -(1 - p_val)   
vect_signed_p_val = np.vectorize(signed_p_val)
######### vect_signed_p_val - transform p-value based on t -value after applying
######## this function significant value will be from 0.95 to 1!!! you can plot results of this func  using limits (0.95,0.97,1)




trial_number=[1,2,3,4]

### path to stc 
data_dir = '/media/kristina/storage/interoception/beta_by_cond_bl_rest_prestim'

### setup time intervals on which we will do ttest
#### if you want you can perform ttest not on the averaged time interval but in all time points p_val_full_fdr will made corection N sources*n time points
intervals = [[-0.250, 0], [0.0, 0.100], [0.100, 0.200], [0.200, 0.300], [0.300, 0.400], [0.400, 0.500],[0.500, 0.600], [0.600, 0.700], [0.700, 0.800], [0.800, 0.900]] 

for idx, inter in enumerate(intervals):
    # download donor stc
    stc_test = mne.read_source_estimate('/media/kristina/storage/interoception/beta_by_cond/ICS26_extercept_async_3.fif-rh.stc', 'fsaverage').crop(tmin=inter[0], tmax=inter[1], include_tmax=True)
    
    stc_test = stc_test.mean()
    comp1_per_sub = np.zeros(shape=(len(subjects), stc_test.data.shape[0], stc_test.data.shape[1]))
    comp2_per_sub = np.zeros(shape=(len(subjects), stc_test.data.shape[0], stc_test.data.shape[1]))
    for ind, subj in enumerate(subjects):
        ### if you already avearage have avereged stc by trials skip iteration by trial_number
        trial_list1 =[]
        trial_list2 =[]
        for n in trial_number:
            try:
                #### load your stc by condions
                temp1 = mne.read_source_estimate(os.path.join(data_dir, "{0}_intercept_async_{1}.fif-lh.stc".format(subj, n))).crop(tmin=inter[0], tmax=inter[1], include_tmax=True)
                temp1 = temp1.mean() # averaging on interval (between time points)
                trial_list1.append(temp1.data)
                
                temp2 = mne.read_source_estimate(os.path.join(data_dir, "{0}_extercept_async_{1}.fif-lh.stc".format(subj, n))).crop(tmin=inter[0], tmax=inter[1], include_tmax=True)
                temp2 = temp2.mean() # averaging on interval (between time points)
                trial_list2.append(temp2.data)
            except OSError:
                print('no file') 
        comp1_per_sub[ind, :, :] = np.mean(np.array(trial_list1),axis=0)
        comp2_per_sub[ind, :, :] = np.mean(np.array(trial_list2),axis=0)
       
    print(comp1_per_sub.shape)
    print(comp2_per_sub.shape)
    ### tteest no FDR correction
    t_stat, p_val = stats.ttest_rel(comp1_per_sub, comp2_per_sub, axis=0)
    
    p_val_nofdr = vect_signed_p_val(t_stat, p_val)
    # stc_test.data = p_val_nofdr
    
    #### ttest with FDR-Correction n sources
    width, height = p_val.shape
    p_val_resh = p_val.reshape(width * height)
    _, p_val_full_fdr = mul.fdrcorrection(p_val_resh,alpha=0.05)
    p_val_full_fdr = p_val_full_fdr.reshape((width, height))
   
    p_val_full_fdr = vect_signed_p_val(t_stat, p_val_full_fdr)
    
    #### there I create mask and thershold power differences by p-value (you will plot only significant vertices)
   
    mask = (p_val_full_fdr > 0.95) | (p_val_full_fdr < -0.95)
    mean_power = np.mean(comp1_per_sub,axis=0) - np.mean(comp2_per_sub,axis=0)
    output_array = np.zeros_like(p_val_full_fdr)

    output_array[mask] = mean_power[mask]

    stc_test.data = output_array
   ######### set up pos_lims based on your power!!!!!!!!!
    
    brain = mne.viz.plot_source_estimates(stc_test, hemi='split', time_viewer=False, background='white', foreground = 'black', cortex='bone', size = (1200, 600),
                                         views = ['lat', 'med'],smoothing_steps=8,clim = dict(kind = 'value', pos_lims =(0.3,0.5,1)), spacing ='oct6')

    # brain.add_label(roi4, borders=True, color='tomato')
    # # brain.add_annotation("HCPMMP1_combined",borders=True)
    
    brain.save_image('/media/kristina/storage/interoception/ttest/intercept_vs_extercept_async_pval_fdr_{0}.png'.format(np.mean(inter)))
    
    brain.close()
    
