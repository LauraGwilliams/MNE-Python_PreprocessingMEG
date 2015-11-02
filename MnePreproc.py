# hamlet preprocess beta

# setup

import mne
import numpy as np
import os
import pylab as pl
from scipy import io
from mne.epochs import combine_event_ids
from mne.minimum_norm import apply_inverse, read_inverse_operator
from eelbrain import *
import glob
import os.path as op
import time

os.environ['MNE_ROOT'] = '/Applications/MNE-2.7.3-3268-MacOSX-i386/'		# make sure this line includes the exact version of MNE that you are using! #
os.environ['SUBJECTS_DIR'] = '/Users/lauragwilliams/Documents/experiments/comp_stereo/new_analysis/mri'
os.environ['FREESURFER_HOME'] = '/Applications/freesurfer/'


# trigger scheme
hamlet_trigs = {

"comp_legal"    :   1,
"comp_illegal"  :   2,
"pseudo_legal"  :   3,
"pseudo_illegal"    :   4,
"unseg_illegal" :   5, # analyse these 5 cells in a factorial design, and also look at continuous variables
"NW_NW_illegal" :   6,
"NW_W_illegal"  :   7,
"W_NW_illegal"  :   8,
"goodRes"   :   101,
"badRes"    :   102,
"noRes" :   103,
"other" : 50

}

class MnePreproc:
    
    """
    
    Preprocess MEG data with MNE-Python
    
    Useage:
    
    proc = MnePreproc(subject,subject_n,experiment,root...)
    
    """
    
    def __init__(self,subject,experiment,root,trigger_scheme):
        """
        
        subject:        string, name of subject
        subject_n:      string, number of subject
        experiment:     string, name of experiment
        root:           string, directory to root of data files
        trigger_scheme: dict, string:value of conditions and triggers
        
        """
    
        # add variables to the class to be called on later
        self.subject = subject
        self.subj_n = self.subject.split('-')[0]
        self.subj_code = "CA%s" % str(self.subj_n)
        self.experiment = experiment
        
        # directories to load from:
        self.root = root
        self.raw_files = self.root + '/raw_files/'
        self.ptp_dir = self.raw_files + subject + '/meg/'
        self.subjects_dir = root + '/raw_files/'
        self.coreg_files = self.raw_files + subject + '/localize/'
        
        
        # directories to populate:
        self.processed_files = self.root + '/processed_files/' + subject + '/'
        self.stc_path = self.root + '/processed_files/stcs/'
        self.behavioural_subj = self.root + '/processed_files/behavioural/by_subject/'
        self.stc_fact = self.stc_path + 'factorial/'
        self.stc_cont = self.stc_path + 'continuous_var/'
        
        # make these dirs above:
        if not os.path.isdir(self.processed_files):
            os.makedirs(self.processed_files)
        
        if not os.path.isdir(self.stc_path):
            os.makedirs(self.stc_path)
        
        # and finally, add other objects
        self.trigger_scheme = trigger_scheme
        self.processing_info = 'preprocessing date: %s' % time.strftime("%d/%m/%Y")
    

    def add_preprocessing_notes(self,note_to_add):
        """add information to the preprocessing note-taker."""
        
        self.processing_info = self.processing_info + '\n' + note_to_add
        
    
    def save_preprocessing_notes(self):
        """saves notes to disk"""
        
        save_dest = '%s%s_preprocessing_notes.txt' % (self.processed_files, self.subject)
        
        file = open(save_dest,'a') 
        file.write(self.processing_info)
        file.close()
    
    def get_dev_t(self):
        """Calculates digital space to sensor-space tranform for CTF data"""
    
        # get all files that end with _raw.fif, and just pick the first one
        raw_files = glob.glob(self.ptp_dir + '*_raw.fif')
        raw = mne.io.read_raw_fif(raw_files[1])

        fid_names = ['nasion', 'left', 'right']
        fids = {fid['ident']: fid for fid in raw.info['dig'] if fid['kind'] == 2}
        src = [fids[1]['r'], fids[2]['r'], fids[3]['r']]
        
        # read the head positions
        file = open(op.join(self.coreg_files, self.subj_n + '_headshape.txt')).readlines()
        
        # for some reason, some of the files are read differently.. so this is to make up for that difference
        if len(file) == 1:
            lines = file[0].split('\r')[1:]
        elif len(file) == 2:
            lines = file[1].split('\r')
        else:
            lines = file[1:]
            
        # read in the dig points and sweeps
        target = []
        for fid in fid_names:
            line = [line for line in lines if line.startswith(fid)]
            target.append(line[0].split()[1:])
        points = [line.split()[1:] for line in lines if not line.startswith(fid)]
        target = np.array(target, float)
        
        # makes a copy of the fid for the dev_head_t
        target = np.vstack((target, target))
        src = np.array(src, float)
        points =  np.array(points, float)

        fid_names = ['nasion', 'lpa', 'rpa', '1', '2', '3']
        dig = mne.channels.montage.read_dig_montage(hsp=points, hpi=src, elp=target,
                                                    point_names=fid_names, unit='cm',
                                                    transform=True, dev_head_t=True)

        raw.set_montage(dig)
        raw.save(self.processed_files + self.subject + '_coreg_info.fif', verbose=False, overwrite=True)
        # dev_head_t = {'from': 1, 'to': 4, 'trans': trans}
        # hsp = [{'kind': 4, 'ident': ii, 'coord_frame': 4, 'r': pt} for ii, pt in enumerate(points)]
        #
        # raw.info['dig'].extend(hsp)
        # raw.info['dev_head_t'] = dev_head_t
    
    def coreg(self):
        """Pulls up coreg gui loading the ctf file with digital data"""
        
        raw = self.processed_files + self.subject + '_coreg_info.fif'
        mne.gui.coregistration(raw,subjects_dir=self.subjects_dir)

    
    # need to add something that checks the number of triggers present in each raw file... number 3 isn't getting
    # the correct number.
    def load_raw(self,block_ns=['1']):
        """
        block_ns:       list of strings, corresponding to number of raw files to combine
        
        
        """

        raws = []
        
        # loads each of the raw blocks - assumes file is called _01_raw.fif, _02_raw.fif...
        for blockn in block_ns:
            raw_oneblock = self.ptp_dir + self.subj_code + '_0%s_raw.fif' % blockn
            raw = mne.io.Raw(raw_oneblock, preload=True, verbose=False)
    
            # make sure all have the same name of the trigger line, and same bad channels marked (i.e., none)
            raw.info['ch_names'][0] = 'STI 014'
            raw.info['bads'] = []
            
            # check number of events in each raw
            event_check = mne.find_events(raw=raw, stim_channel='STI 014', min_duration=0.002, verbose=False)
            print len(event_check), "events found for raw_0%s" % blockn
            
            self.add_preprocessing_notes( "%s events found for raw_0%s" % (str(len(event_check)), str(blockn)) )
            
            # append raws together
            raws.append(raw)
            
        # put all the raws together into one object
        for number in map(int,block_ns)[:-1]:
            raws[0].append(raws[number])
            raw = raws[0]
        
        # save the unfiltered raw to the class
        self.raw_unfiltered = raw
           
        return self.raw_unfiltered
    
    
    def filter_raw(self,low_bound=0,high_bound=40,method='iir'):
        """docstring for filter_raw"""
        
        # filter the raw data, loading from the class directly
        self.raw_unfiltered.filter(low_bound, high_bound, method = method)
        
        # add the filtered raw to the class
        self.raw = self.raw_unfiltered
        
        return self.raw
     
        
    def find_events(self,raw=None, stim_channel = 'STI 014', min_duration = 0.002):
        """docstring for find_events"""
        
        # use mne.find_events to get the events from the raw file
        events = mne.find_events(raw=self.raw, stim_channel=stim_channel, min_duration=min_duration)
        
        # add raw events to class
        self.raw_events = events
        
        return self.raw_events
        
    
    def add_info_to_events(self):
        """
        Makes a dataset of the events, and adds in condition information.
        
        """
        
        # gets the mne events and put the data into a readable dataset, with columns for each
        # informative entry. the mne events are structures as n_time lists of three entries - 
        # one for time, one for trigger down, one for trigger line
        ds = Dataset()
        ds['time'] = Var(self.raw_events[:,0]) # get just the first column of the events and add to ds
        ds['trigger'] = Var(self.raw_events[:,2])
        
        # switch key-value pariring in trigger dict:
        my_dict = self.trigger_scheme
        my_dict_flipped = {y:x for x,y in my_dict.iteritems()}
        
        # look at the trigger dict to add condition values to the ds
        cond_values = []
        for trig in ds['trigger']:
            cond_values.append(my_dict_flipped[trig])
        
        ds['condition'] = Factor(cond_values)
        
        self.ds_events = ds
        
        return self.ds_events


    def label_events(self,save=False):
        """
        Adds condition column to daatase
        
        # need to make a way for the use to define all the things that go into this.
        
        """
        
        # run the function above
        ds = self.add_info_to_events()
               
         
        # - # all of this below is specific to the hamlet study.. figures out RT and acc based on the sent triggers
        acc = []
        RT = []
        
        for line in xrange(len(ds[ds.keys()[0]])):
                        
            # add RT and acc information, up til the last event
            if line+1 < len(ds['condition']):
                
                RT.append(ds['time'][line+1] - ds['time'][line])
                
                res = ds['condition'][line+1]
                trig = ds['trigger'][line]
                                
                # correct responses
                if res == 'goodRes' and trig < 6:
                    acc.append(1)
                elif res == 'badRes' and trig > 5:
                    acc.append(1)
                    
                # incorrect responses
                elif res == 'badRes' and trig < 6:
                    acc.append(0)
                elif res == 'goodRes' and trig > 5:
                    acc.append(0)
                
                # no response and other
                elif res == 'noRes':
                    acc.append(0)
                else:
                    acc.append(99)
            
            # if we are on the last line, act as if incorrect  
            else:
                RT.append(0)
                acc.append(99)
        
        # check dimensions match        
        if len(acc) != len(ds['condition']):
            raise ValueError("length mismatch")
        
        ds['accuracy'] = Var(acc)
        ds['RT'] = Var(RT)
        
        # compare trigger value to user-defined value.. only keep events that match the test here, both in
        # the mne-events and the ds events
        ds_clean = ds.sub(ds['trigger'] < 6)
        old_events_just_cond = self.raw_events[ds['trigger'] < 6]
        
        # save behav. results
        if save == True:
            
            ds_clean.save_txt(self.behavioural_subj + self.subject + '_behavioural.csv',delim=',')
        
        # this is an index of when the ptp answered incorrectly
        behavioural_rejections = ds_clean['accuracy'] == 1
        
        self.old_events_just_cond = old_events_just_cond
        self.behavioural_rejections = behavioural_rejections
        self.labeled_events = ds
        self.events_clean = ds_clean
        
        return self.events_clean
    
    
    def reject_incorrect_responses(self):
        """
        Deletes events with incorrect response.
        """
        
        # remove events based on behavioural response index    
        events_correct_resp = self.events_clean.sub(self.behavioural_rejections)
        old_events_correct_resp = self.old_events_just_cond[self.behavioural_rejections]
        
        # get the number of events before and after rejection
        events_before = len(self.events_clean[self.events_clean.keys()[0]])
        events_after =len(events_correct_resp[events_correct_resp.keys()[0]])
        
        print "Number of incorrect responses removed: %s" % str(events_before - events_after)
        
        # add mne-events and ds events to class, and add behavioural info to the events, too
        self.events_clean = events_correct_resp
        self.processing_info = self.processing_info + "\nbehaviourally rejected %s trials" % str(events_before - events_after)
        self.old_events_correct_resp = old_events_correct_resp
        
        # save this kind of info to file - how many lost per condition.
        
        return events_correct_resp
        
    
    # def add_epochs(self,ds,tmin,tmax,i_start='time'):
    #     """docstring for add_epochs"""
    #
    #     epochs = load.fiff.add_mne_epochs(ds=ds,tmin=tmin,tmax=tmax,target='epochs',raw=self.raw,i_start='time')
    #
    #     self.epochs = epochs
    #     return self.epochs
    
    #this is the old way to do it... going to try christian's way, but i'll keep this here in case.
    def make_epochs(self,events,tmin,tmax,proj=True,baseline=(None, 0),
                    preload=True):
        """docstring for make_epochs"""

        epochs = mne.Epochs(raw=self.raw, events=events, event_id=self.trigger_scheme,
                            tmin=tmin, tmax=tmax, proj = True, baseline=baseline, preload = True, on_missing = 'ignore')
                 
        self.epochs = epochs
        
        return self.epochs
        
    
    def prepare_raw_and_epoch(self,block_ns,trigger_scheme,tmin,tmax):
        """docstring for prepare_raw_and_epoch"""
        
        self.load_raw(block_ns)
        self.filter_raw()
        self.find_events()
        self.add_info_to_events()
        self.label_events()
        
        epochs = self.add_epochs(tmin,tmax,trigger_scheme)
        
        self.epochs = epochs
        
        return epochs
    
    
    def gui_blink_reject(self,epochs):
        """docstring for blink_reject"""
        
        ds = Dataset()
        ds['epochs'] = epochs
        ds['trigger'] = self.events_clean['trigger']
        g = gui.select_epochs(ds, data='epochs', mark=['MEG 087','MEG 130'], vlim=3e-12, path= (self.processed_files + self.subject + '_rejected.txt')) # path to save rejections #
     
        
        epochs_clean = self.apply_blink_rejections()
        
        n_before = len(epochs)
        n_after = len(epochs_clean)
        
        print "Rejected %s trials" % str(n_before - n_after)
        self.processing_info = self.processing_info + "\nrejected %s trials in artifact rejection" % str(n_before - n_after)
        
        
        # should find a way to save these for regression analysis!!
        
        return epochs_clean
    
            
    def apply_blink_rejections(self):
        """docstring for apply_blink_rejections"""
        
        # need to change this so that it subsets just the epochs alone, and then the events alone, then maybe combines them,
        # although this seems to do more harm than good.
        
        blink = load.tsv( self.processed_files + self.subject + '_rejected.txt')
        idx = blink['accept'].x
        self.epochs_clean = self.epochs[idx]
        self.events_clean['survived_rej'] = Factor(idx)
        
        return self.epochs_clean
    
    
    def make_evoked_per_condition(self):
        """docstring for make_evoked_per_condition"""
        
        self.grand_average_evoked = self.epochs_clean.average()
        
        evokeds = []
        conds = []
        
        for i, key in enumerate(self.trigger_scheme):
            
            if key in self.events_clean['condition']:
                print i,key
            
                self.epochs[key].get_data()
                evokeds.append(self.epochs[key].average())
                conds.append(key)
        
        self.evoked_list = zip(evokeds,conds)
               
        self.evokeds = evokeds
        
        return self.evokeds
     
        
    def compute_covariance_matrix(self,epochs=None,tmax=0,method='auto'):
        """"""
        
        fname = self.processed_files + self.subject + '_cov.fif'
        
        if not op.isfile(fname):
        
            cov = mne.cov.compute_covariance(self.epochs, tmax = 0, method=method)
            cov_reg = mne.cov.regularize(cov, self.grand_average_evoked.info, mag=0.05,
                                         grad = 0.05, proj = True, exclude = 'bads')
        
            mne.write_cov(fname, cov_reg)
        
        else:
            cov_reg = mne.read_cov(fname)
        
        self.cov_reg = cov_reg
        
        return self.cov_reg
    
    
    def compute_forward_and_inverse_solutions(self, bem = None, orientation = 'fixed'):
        """docstring for compute_forward_solution"""
        
        info = self.grand_average_evoked.info
        trans = mne.read_trans(self.processed_files + self.subject + '-trans.fif')
        src = glob.glob(self.subjects_dir + '/' + self.subject + '/bem/' + '*-ico-4-src.fif')[0]
        bem = glob.glob(self.subjects_dir + '/' + self.subject + '/bem/' + '*-bem-sol.fif')[0]
        fname = self.processed_files + self.subject + '_forward.fif'

        # check if fwd exists, if not, make it
        if not op.isfile(fname):
            fwd = mne.make_forward_solution(info = info, trans = trans, src = src,
                                            bem = bem, fname = fname, meg = True, eeg = False,
                                            overwrite = True)
        
        if orientation == 'fixed':
            force_fixed = True
        else:
            force_fixed = False
                                        
        fwd = mne.read_forward_solution(fname,force_fixed=force_fixed)
        
        self.forward_solution = fwd
        
        inv = mne.minimum_norm.make_inverse_operator(info, self.forward_solution, self.cov_reg, loose = None, depth = None, fixed = force_fixed)        
        self.inverse_solution = inv
        mne.minimum_norm.write_inverse_operator(self.processed_files + self.subject + '_inv.fif',self.inverse_solution)
        
        return fwd, inv
    
    def make_epoch_stcs(self, epochs, snr = 2.0, method='dSPM', save_to_disk = False):
        """Apply inverse operator to epochs to get source estimates of each item"""
        
        
        lambda2 = 1.0 / snr ** 2.0
        
        inverse = mne.minimum_norm.read_inverse_operator( self.processed_files + self.subject + '_inv.fif' )
        
        eps = mne.minimum_norm.apply_inverse_epochs(epochs=epochs,inverse_operator=inverse,lambda2=lambda2,method = method)
        
        if save_to_disk == True:
            save.pickle(obj=eps,dest=self.stc_cont + self.subject + '_stc_epochs')
        
        return eps
        
    
    def make_stcs(self, snr = 3.0, method='dSPM', save_to_disk = False, morph=True):
        """docstring for make_stcs"""
        
        morph_status = 'no_morph'
        
        lambda2 = 1.0 / snr ** 2.0
        pick_ori = None
        
        stcs = []
        
        for evoked_object, cond in self.evoked_list:
            stc = mne.minimum_norm.apply_inverse(evoked_object, self.inverse_solution, method = method, lambda2 = lambda2, pick_ori = pick_ori)
        
            if morph == True:
            
                morph_status = 'morphed'
            
                print morph
                # create morph map
            
                # get vertices to morph to (we'll take the fsaverage vertices)
                subject_to = 'fsaverage'
                fs = mne.read_source_spaces(self.subjects_dir + '%s/bem/%s-ico-4-src.fif' % (subject_to, subject_to))
                vertices_to = [fs[0]['vertno'], fs[1]['vertno']]
            
                # info of the stc we're going to morph is just the present stc
                subject_from = self.subject
                stc_from = stc
            
                # use the morph function
                stc = mne.morph_data(subject_from, subject_to, stc_from, n_jobs=1,
                                        grade=vertices_to, subjects_dir=self.subjects_dir)
                                    
                # stc_to.save('%s_audvis-meg' % subject_from)
                # mne.compute_morph_matrix('2-COMB','3-COMB',stcs[0].vertices, stcs[5].vertices, subjects_dir=self.subjects_dir)
            
            if save_to_disk == True:
                if not os.path.isdir("%s/%s/%s/" % (self.stc_fact, morph_status, cond)):
                    os.makedirs("%s/%s/%s/" % (self.stc_fact, morph_status, cond))
                        
                stc.save("%s/%s/%s/%s_%s_%s" % (self.stc_fact, morph_status, cond, self.subject, cond, morph_status))
        
     
        