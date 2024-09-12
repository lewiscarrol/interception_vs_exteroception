
import mne
import os
import os.path as op
import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats import multitest as mul
mne.viz.set_3d_options(antialias=False)

data_dir = '/media/kristina/storage/interoception/beta_by_cond'
os.environ['SUBJECTS_DIR'] = '/media/kristina/storage/interoception/freesurfer'
subjects_dir = '/media/kristina/storage/interoception/freesurfer'

subjects = ['ICS01','ICS03','ICS04','ICS05','ICS06','ICS07','ICS08','ICS09','ICS10','ICS12',
            'ICS13','ICS15', 'ICS17','ICS18','ICS19','ICS20','ICS21','ICS22',
            'ICS23','ICS24','ICS25','ICS26','ICS27','ICS28','ICS29','ICS30','ICS31']
# Correct the intervals to be consistent
intervals = [[-0.250, 0], [0.0, 0.100], [0.100, 0.200], [0.200, 0.300], [0.300, 0.400], 
             [0.400, 0.500], [0.500, 0.600], [0.600, 0.700], [0.700, 0.800]]



trial_number=[1,2,3,4,5,6]
all_p_values = []
all_t_stats = []
all_mean = []
conditions=['intercept','extercept']
for idx, inter in enumerate(intervals):
    # Download donor stc
    stc_test = mne.read_source_estimate('/media/kristina/storage/interoception/beta_by_cond/ICS01_extercept_sync_3-rh.stc', 'fsaverage').crop(tmin=inter[0], tmax=inter[1], include_tmax=True)
    
    stc_test = stc_test.mean()
    num_vertices, num_times = stc_test.data.shape  # num_times should now reflect the cropped interval time points
    
    comp1_per_sub = np.zeros((len(subjects), num_vertices, num_times))
    comp2_per_sub = np.zeros((len(subjects), num_vertices, num_times))
    
    for ind, subj in enumerate(subjects):
        con_list=[]
        for cond in conditions:
            trial_list1 = []
            for n in trial_number:
                try:
                # Load and crop data for subject and trial, then compute mean
                    temp1 = mne.read_source_estimate(os.path.join(data_dir, f"{subj}_{cond}_sync_{n}-lh.stc")).crop(tmin=inter[0], tmax=inter[1], include_tmax=True)
                    temp1 = temp1.mean()
                    trial_list1.append(temp1.data)
            
                
                except OSError:
                    print(f"No file found for {subj}, trial {n}")
            con_list.append(np.mean(np.array(trial_list1),axis=0))
        comp1_per_sub[ind, :, :] = np.mean(con_list, axis=0)

    
    t_stat, p_val = stats.ttest_1samp(comp1_per_sub,0, axis=0)
   
    mean_power = np.mean(comp1_per_sub,axis=0)

    all_mean.append(mean_power)
    all_t_stats.append(t_stat)
    all_p_values.append(p_val)
    

# Now apply FDR correction after collecting all p-values
# Flatten the p-values across all intervals for FDR correction
all_p_values_flat = np.concatenate([p.flatten() for p in all_p_values])


_, p_val_full_fdr_flat = mul.fdrcorrection(all_p_values_flat, alpha=0.001)



data_dir = '/media/kristina/storage/interoception/beta_by_cond'

# Now we can loop over intervals again to update the data and apply masks
for idx, inter in enumerate(intervals):
    # Download donor stc again
    stc_test = mne.read_source_estimate('/media/kristina/storage/interoception/beta_by_cond/ICS01_extercept_sync_3-rh.stc', 'fsaverage').crop(tmin=inter[0], tmax=inter[1], include_tmax=True)
    
    stc_test = stc_test.mean()
    
    # Apply signed correction to p-values for this interval
    mask = (p_val_full_fdr_flat[idx]> 0.05) 
    mean_power = all_mean[idx]
    mean_power[mean_power > 0] = 0
    output_array = np.zeros_like(all_mean[idx])
    
    output_array[mask] = mean_power[mask]

    stc_test.data = output_array
    brain = mne.viz.plot_source_estimates(stc_test, hemi='split', time_viewer=False, background='white', foreground = 'black', cortex='bone', size = (800, 600),
                                         views = ['lat', 'med'],clim = dict(kind = 'value', pos_lims =(0.4,0.9,1.2)), spacing ='oct6')
   
    #brain.add_annotation("HCPMMP1_combined",borders=True)
    
    
    brain.save_image('/media/kristina/storage/interoception/ttest/grand_averaged_pval_fdr_{0}.png'.format(np.mean(inter)))
    
    brain.close()   
    
    


