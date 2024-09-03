import mne
import os
import os.path as op
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as mul
from mne.stats import  fdr_correction
mne.viz.set_3d_options(antialias=False)

os.environ['SUBJECTS_DIR'] = '/media/kristina/storage/interoception/freesurfer'
subjects_dir = '/media/kristina/storage/interoception/freesurfer'
subjects = ['ICS01','ICS03','ICS04','ICS05','ICS06','ICS07','ICS08','ICS09','ICS10','ICS12',
            'ICS13','ICS15', 'ICS17','ICS18','ICS19','ICS20','ICS21','ICS22',
            'ICS23','ICS24','ICS25','ICS26','ICS27','ICS28','ICS29','ICS30','ICS31',
            'mediawork5','mediawork6','mediawork10','mediawork11','mediawork13',
            'mediawork14','mediawork15','mediawork16','mediawork17','mediawork18',
            'mediawork20','mediawork22','mediawork23','mediawork25','mediawork26','mediawork27','mediawork28',
            'mediawork29','mediawork31','mediawork32','mediawork33','mediawork39','mediawork40','mediawork50','mediawork51']



L_freq =10
H_freq = 15
f_step = 2
freqs = np.arange(L_freq, H_freq, f_step)
time_bandwidth = 2  # (by default = 4)
n_cycles = freqs//2
lambda2 = 1. / 3 ** 2


bands = dict(beta=[L_freq, H_freq])
src = mne.setup_source_space('fsaverage', spacing='oct6',subjects_dir=subjects_dir,add_dist=True,n_jobs=-1)
model=mne.make_bem_model('fsaverage', ico=4, conductivity=0.3, subjects_dir=subjects_dir, verbose=None)
bem = mne.make_bem_solution(model)

for subj in subjects:
    raw = mne.io.Raw('/media/kristina/storage/interoception/epo_fif/{0}_mne_raw_tsss.fif'.format(subj),preload=True)

    trans= '/media/kristina/storage/interoception/trans/{0}-trans.fif'.format(subj)

    events=mne.find_events(raw,'STI101', shortest_event=0, min_duration=0.005,uint_cast=True)
    rank = mne.compute_rank(raw)
    ######## use empty room covariance
    cov =mne.read_cov('/media/kristina/storage/interoception/empty_room/{0}_er-cov.fif'.format(subj))
    info = raw.info
    #cov = mne.cov.make_ad_hoc_cov(info)
    fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src, bem=bem,  eeg=False, n_jobs=-1)
    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2, depth=0.8)
   
    
    intercept_condition= (events==16384)|(events==16384+1024)|(events==16384+2048)
    intercept= np.where(intercept_condition)[0]
    intercept_triggers=[]
    for i in intercept:
        search_range = events[i:i+6, 2]
        if 2 in search_range:
            
            event_range = events[i+1:i+6, :]
            event_rows = event_range[(event_range[:, 2] == 2)]
            intercept_triggers.append(event_rows)
            
    intercept_triggers=np.array(intercept_triggers).reshape(len(intercept_triggers),3)  
    epochs_intercept= mne.Epochs(raw,intercept_triggers,tmin=-1.000, tmax=19.0, baseline =None,reject_by_annotation=True,preload=True)
    epochs_intercept.subtract_evoked(epochs_intercept.average())
    
    beta_intercept= mne.minimum_norm.source_band_induced_power(epochs_intercept, inv, bands=bands,lambda2=lambda2, method='sLORETA', n_cycles=8, use_fft=True,nave=len(epochs_intercept), baseline=None)['beta']
    bl1= beta_intercept.copy().crop(tmin=-1.000,tmax=-0.100)
    bl_data =np.mean(bl1.data, axis=1).reshape(8196,1)
    beta_intercept.data = 10*np.log10(beta_intercept.data) - 10*np.log10(bl_data)
   
    beta_intercept.save('/media/kristina/storage/interoception/alpha_by_long_condition/{0}_interception_async'.format(subj))
    
    
    extercept_condition= (events==32768)|(events==32768+1024)|(events==32768+2048)
    extercept= np.where(extercept_condition)[0]
    extercept_triggers=[]
    for e in extercept:
        search_range = events[e:e+6, 2]
        if 2 in search_range:
            
            event_range = events[e+1:e+6, :]
            event_rows = event_range[(event_range[:, 2] == 2)]
            extercept_triggers.append(event_rows)
            
    extercept_triggers=np.array(extercept_triggers).reshape(len(extercept_triggers),3)
    extercept_triggers=np.unique(extercept_triggers,axis=0)
    
    ########### collect the epochs #########
    epochs_extercept= mne.Epochs(raw,extercept_triggers,tmin=-1.000, tmax=19.0, baseline =None,reject_by_annotation=True,preload=True)
    epochs_extercept.subtract_evoked(epochs_extercept.average())
    
    ######### made the source localization with parallel TF in selected freq
    beta_extercept= mne.minimum_norm.source_band_induced_power(epochs_extercept, inv, bands=bands,lambda2=lambda2, method='sLORETA', n_cycles=8, use_fft=True,nave=len(epochs_intercept), baseline=None)['beta']
    
    bl1= beta_extercept.copy().crop(tmin=-1.000,tmax=-0.100) ### crop interval for baseline
    bl_data =np.mean(bl1.data, axis=1).reshape(8196,1) ####### Average baseline interval in time
    ######## baseline correction (made log10 to transform to the dB)
    beta_extercept.data = 10*np.log10(beta_extercept.data) - 10*np.log10(bl_data)
    beta_extercept.save('/media/kristina/storage/interoception/alpha_by_long_condition/{0}_exteroception_async'.format(subj))
    
