import mne, os, glob, time, pickle, pandas
import numpy as np
import os.path as op
from mne.epochs import combine_event_ids
from mne.minimum_norm import apply_inverse, read_inverse_operator
from eelbrain import *
from copy import copy

os.environ['MNE_ROOT'] = '/Applications/MNE-2.7.0-3106-MacOSX-i386/'		# make sure this line includes the exact version of MNE that you are using! #
os.environ['SUBJECTS_DIR'] = '/.../mri'
os.environ['FREESURFER_HOME'] = '/Applications/freesurfer/'


# trigger scheme
trigger_scheme = {'A1_B1': 1,'A2_B1': 2,'A2_B1': 3,'A2_B2': 4}
condition_format = 'factor1_factor2'

class MnePreproc:

    """

    Preprocess MEG data with MNE-Python

    Useage:

    proc = MnePreproc(subject,subject_n,experiment,root...)

    """

    def __init__(self,subject,experiment,root,trigger_scheme,condition_format=None):
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
        self.raw_files = op.join(self.root, 'raw_files', '')
        self.ptp_dir = op.join(self.raw_files, subject, 'meg', '')
        self.subjects_dir = op.join(self.root, 'mri', '')
        self.coreg_files = op.join(self.raw_files, subject, 'localize', '')


        # directories to populate:
        self.processed_files = op.join(self.root, 'processed_files', subject, '')
        self.stc_path = op.join(self.root, 'processed_files', 'stcs', '')
        self.behavioural_subj = op.join(self.root, 'processed_files', 'behavioural', 'by_subject', '')
        self.stc_fact = op.join(self.stc_path, 'factorial', '')
        self.stc_cont = op.join(self.stc_path, 'continuous_var', '')

        # make these dirs above:
        for directory in [self.processed_files, self.stc_path, self.stc_fact, self.stc_cont]:
            if not op.isdir(directory):
                os.makedirs(directory)

        # and finally, add other objects
        self.trigger_scheme = trigger_scheme
        self.condition_format = condition_format
        self.processing_info = 'Preprocessing initialized: %s\n' %time.strftime('%d %b %Y, %H:%M:%S') + 48*'-' + '\n'


    def add_preprocessing_notes(self,note_to_add):
        """add information to the preprocessing note-taker."""

        self.processing_info += '%s: %s\n' %(time.strftime('%d %b %Y, %H:%M:%S'), note_to_add)


    def save_preprocessing_notes(self, filename=None):
        """saves notes to disk"""

        if filename is None:
            save_dest = op.join(self.processed_files, '%s_preprocessing_notes.txt' %self.subject)
        else:
            save_dest = op.join(self.processed_files, '%s_%s.txt' %(self.subject, filename))

        self.processing_info += 48*'-' + '\n' + 48*'-' + '\n'

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
        """Opens coregistration GUI"""

        mne.gui.coregistration(subjects_dir=self.subjects_dir)


    def load_raw(self, filename=None, mark_bads=True):

        if filename is None:
            filename = op.join(self.ptp_dir, '%s_%s-raw.fif' %(self.subject, self.experiment))
        print "Loading raw data from file %s" %filename
        raw = mne.io.read_raw_fif(filename, preload=True, verbose=False)
        event_check = mne.find_events(raw=raw, stim_channel='STI 014', min_duration=0.002, verbose=False)
        exp_event_check = event_check[np.in1d(event_check[:,2], self.trigger_scheme.values())]
        print len(event_check), "total events found."
        print len(exp_event_check), "events found with experiment-relevant triggers."

        if mark_bads:
            bad_file = glob.glob(self.processed_files + '*bad_channels.txt')
            if len(bad_file) == 0:
                print "No bad channels file found."
            elif len(bad_file) > 1:
                raise RuntimeError("Two or more bad channels files were found")
            else:
                bads = [i.strip() for i in open(bad_file[0]).readlines()]
                raw.info['bads'] += bads
                print "Marking the following channels as bad: " + str(bads)

        self.add_preprocessing_notes("Raw data loaded from file %s" %filename)
        for bad in raw.info['bads']:
            self.add_preprocessing_notes("%s marked as bad channel." %bad)

        self.raw_unfiltered = raw

        return raw

    # need to add something that checks the number of triggers present in each raw file... number 3 isn't getting
    # the correct number.

    def load_raw_from_blocknums(self,block_ns=['1']):
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

    def load_filtered(self, filter='0-40', filename=None):

        if filename is None:
            filename = op.join(self.processed_files, '%s_%s_%sHz.fif' %(self.subject, self.experiment, filter))

        print "Loading filtered data (%s Hz)" %filter
        raw = mne.io.read_raw_fif(filename, preload=True, verbose=False)

        self.add_preprocessing_notes("Filtered data loaded from file %s" %filename)

        self.raw = raw

        return raw

    def filter_raw(self,low_bound=0,high_bound=40,method='iir',save_to_disk=False):
        """docstring for filter_raw"""

        # filter the raw data, loading from the class directly
        print "Filtering raw data from %d to %d Hz." %(low_bound, high_bound)
        self.raw_unfiltered.filter(low_bound, high_bound, method = method)

        # add the filtered raw to the class
        self.raw = self.raw_unfiltered
        del self.raw_unfiltered

        self.add_preprocessing_notes("Raw data filtered from %d to %d Hz with %s method." %(low_bound, high_bound, method))

        if save_to_disk:
            dest = op.join(self.processed_files, '%s_%s_%s-%sHz.fif' %(self.subject, self.experiment, str(low_bound), str(high_bound)))
            self.raw.save(dest)

        return self.raw


    def find_events(self, stim_channel = 'STI 014', min_duration = 0.002):
        """docstring for find_events"""

        # use mne.find_events to get the events from the raw file
        events = mne.find_events(raw=self.raw, stim_channel=stim_channel, min_duration=min_duration)

        # add raw events to class
        self.raw_events_all = events
        self.raw_events = events[np.in1d(events[:,2], self.trigger_scheme.values())]

        self.add_preprocessing_notes("%d total events found." %(len(self.raw_events_all)))
        self.add_preprocessing_notes("%d events found with triggers in trigger scheme." %(len(self.raw_events)))
        return self.raw_events

    def configure_conditions(self, cond_splitter='_'):
        """
        Requires the factor names given in condition_format. Generates a nested dictionary of factors and their levels with trigger lists as values.

        Also generates inverted trigger schemes for use in labeling events in the add_info_to_events method.
        """

        if self.condition_format is None:
            raise RuntimeError("No condition format was specified when initializing the MnePreproc class. Cannot create factorial trigger scheme.")
        else:
            factors = self.condition_format.split(cond_splitter)
            fts = {f: {} for f in factors}
            for cond in self.trigger_scheme.keys():
                cond_split = cond.split(cond_splitter)
                for i, l in enumerate(cond_split):
                    fts[factors[i]].setdefault(l, [])
                    fts[factors[i]][l].append(self.trigger_scheme[cond])

            self.factors = factors
            self.factorial_trigger_scheme = fts
            self.inverted_triggers = {trig: cond for cond, trig in self.trigger_scheme.iteritems()}
            self.inverted_triggers_levels = {trig: cond.split(cond_splitter) for trig, cond in self.inverted_triggers.iteritems()}

    def add_info_to_events(self):
        """
        Makes a dataset of the events, and adds in condition information.

        User can add code to parse a logfile to retrieve single trial information.
        """

        ds = pandas.DataFrame()
        ds['time'] = self.raw_events[:,0] # get just the first column of the events and add to ds
        ds['trigger'] = self.raw_events[:,2]

        try:
            self.inverted_triggers
        except AttributeError:
            self.inverted_triggers = {trig: cond for cond, trig in self.trigger_scheme.iteritems()}

        # look at the trigger dict to add condition values to the ds
        ds['condition'] = [self.inverted_triggers[t] for t in ds['trigger']]

        try:
            for i, f in enumerate(self.factors):
                ds[f] = [self.inverted_triggers_levels[t][i] for t in ds['trigger']]
        except AttributeError:
            pass

        ## PARSE LOGFILE HERE AND ADD RELEVANT COLUMNS TO THE DATASET

        self.ds_events = ds

        return self.ds_events


    def reject_incorrect_responses(self):
        """
        Deletes events with incorrect responses.
        """
        # USER-DEFINED CODE TO SUBSET ds_events AND/OR raw_events

    def make_epochs(self,events,tmin,tmax,baseline=(None, 0), proj=True,
                    preload=True):
        """
        Epochs filtered data given the relevant event array, epoch timing parameters, and baseline correction information.
        """

        picks = mne.pick_types(self.raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
        epochs = mne.Epochs(raw=self.raw, events=events, event_id=self.trigger_scheme,
                            tmin=tmin, tmax=tmax, proj=proj, baseline=baseline, picks=picks, preload = True, on_missing = 'ignore')

        self.add_preprocessing_notes("%d epochs of length %d ms extracted." %(len(epochs), (tmax-tmin)*1000))

        self.epochs = epochs

        return self.epochs


    def prepare_raw_and_epoch(self, tmin, tmax, baseline=(None, 0), filter='0-40', mark_bads=True):
        """docstring for prepare_raw_and_epoch"""

        if op.exists(op.join(self.processed_files, '%s_%s_%sHz.fif' %(self.subject, self.experiment, filter))):
            self.load_filtered()
        else:
            self.load_raw(mark_bads=mark_bads)
            self.filter_raw(low_bound=int(filter.split('-')[0]), high_bound=int(filter.split('-')[1]))

        self.find_events()

        if self.condition_format is not None:
            self.configure_conditions()

        self.add_info_to_events()

        epochs = self.make_epochs(events=self.raw_events, tmin=tmin, tmax=tmax, baseline=baseline)

        self.epochs = epochs

        return epochs


    def gui_blink_reject(self,epochs,eye_channels='AD'):
        """docstring for blink_reject"""

        if eye_channels == 'NY':
            eye_channels = ['MEG 143', 'MEG 151']
        elif eye_channels == 'AD':
            eye_channels = ['MEG 087', 'MEG 130']

        g = gui.select_epochs(epochs, mark=eye_channels, vlim=3e-12, path= op.join(self.processed_files, '%s_rejected.txt' %self.subject)) # path to save rejections #

        epochs_clean = self.apply_blink_rejections()

        return epochs_clean

    def apply_blink_rejections(self):
        """docstring for apply_blink_rejections"""

        # need to change this so that it subsets just the epochs alone, and then the events alone, then maybe combines them,
        # although this seems to do more harm than good.

        rej_files = glob.glob(self.processed_files + self.subject + '*reject*')

        if len(rej_files) > 1:
            raise RuntimeError('Two or more rejection files were detected.')
        elif len(rej_files) == 0:
            raise RuntimeError('No epoch rejection file was detected.')
        else:
            if op.splitext(rej_files[0])[1] == '.pickled': # if rejection file is an eelbrain pickle from an old eelbrain analysis, load into eelbrain Dataset
                blink = load.unpickle(rej_files[0])
                idx = blink['accept'].x
            elif op.splitext(rej_files[0])[1] == '.txt': # if it's a regular text file, load into pandas DataFrame
                blink = pandas.DataFrame.from_csv(rej_files[0], sep='\t', index_col=False)
                idx = blink['accept']

            self.epochs_clean = self.epochs[idx]
            self.ds_events['survived_rej'] = idx

            print "Rejecting %d epochs as determined by the rejection file %s" %(len(idx[idx==False]), op.basename(rej_files[0]))
            self.add_preprocessing_notes("Rejected %d epochs as determined by the rejection file %s" %(len(idx[idx==False]), op.basename(rej_files[0])))

            return self.epochs_clean


    def make_evoked_per_condition(self, factorial=False, combine_ids=None):
        """docstring for make_evoked_per_condition"""

        self.grand_average_evoked = self.epochs_clean.average()

        evokeds = []
        conds = []

        for i, key in enumerate(self.trigger_scheme):
            if key in self.ds_events['condition'].unique():
                print "Making evoked for %s." %key
                evokeds.append(self.epochs_clean[key].average())
                conds.append(key)

        if factorial:
            for factor in self.factorial_trigger_scheme.keys():
                for level in self.factorial_trigger_scheme[factor].keys():
                    print "Making evoked for %s" %level
                    ep = self.epochs_clean[np.in1d(self.epochs_clean.events[:,2], self.factorial_trigger_scheme[factor][level])]
                    av = ep.average()
                    av.comment = level
                    evokeds.append(av)
                    conds.append(level)

        if combine_ids is not None:

            for combo, subs in combine_ids.iteritems():

                if not np.all(np.in1d(subs, self.trigger_scheme.keys())):
                    raise RuntimeError("One of the conditions you are trying to combine does not exist in the trigger scheme.")

                print "Making evoked for %s" %combo
                trigs = [self.trigger_scheme[sub] for sub in subs]
                ep = self.epochs_clean[np.in1d(self.epochs_clean.events[:,2], trigs)]
                av = ep.average()
                av.comment = combo
                evokeds.append(av)
                conds.append(combo)

        self.evoked_list = zip(evokeds,conds)
        self.evokeds = evokeds

        self.add_preprocessing_notes("Averaged evoked response generated for conditions: %s" %(', '.join(conds)))

        return self.evokeds


    def compute_covariance_matrix(self,tmax=0,method='auto'):
        """"""

        fname = op.join(self.processed_files, '%s_cov.fif' %self.subject)

        if not op.exists(fname):

            cov = mne.cov.compute_covariance(self.epochs_clean, tmax = 0, method=method)
            cov_reg = mne.cov.regularize(cov, self.grand_average_evoked.info, mag=0.05,
                                         grad = 0.05, proj = True, exclude = 'bads')

            mne.write_cov(fname, cov_reg)

            self.add_preprocessing_notes("Noise covariance matrix computed, regularized, and saved to %s" %fname)

        else:
            cov_reg = mne.read_cov(fname)

        self.cov_reg = cov_reg

        return self.cov_reg


    def compute_forward_and_inverse_solutions(self, orientation = 'fixed'):
        """docstring for compute_forward_solution"""

        info = self.grand_average_evoked.info
        trans = mne.read_trans(op.join(self.processed_files, '%s-trans.fif' %self.subject))
        src = glob.glob(op.join(self.subjects_dir, self.subject, 'bem', '*-ico-4-src.fif'))[0]
        bem = glob.glob(op.join(self.subjects_dir, self.subject, 'bem', '*-bem-sol.fif'))[0]
        fname = op.join(self.processed_files, '%s_forward.fif' %self.subject)

        # check if fwd exists, if not, make it
        if not op.exists(fname):
            fwd = mne.make_forward_solution(info = info, trans = trans, src = src,
                                            bem = bem, fname = fname, meg = True, eeg = False,
                                            overwrite = True, ignore_ref = True)

            self.add_preprocessing_notes("Forward solution generated and saved to %s" %fname)

        if orientation == 'fixed':
            force_fixed = True
        else:
            force_fixed = False

        fwd = mne.read_forward_solution(fname,force_fixed=force_fixed)

        self.forward_solution = fwd

        inv = mne.minimum_norm.make_inverse_operator(info, self.forward_solution, self.cov_reg, loose = None, depth = None, fixed = force_fixed)
        self.inverse_solution = inv
        mne.minimum_norm.write_inverse_operator(op.join(self.processed_files, '%s_inv.fif' %self.subject), self.inverse_solution)

        self.add_preprocessing_notes("Inverse solution generated and saved to %s" %op.join(self.processed_files, '%s_inv.fif' %self.subject))
        return fwd, inv

    def make_epoch_stcs(self, epochs, snr = 2.0, method='dSPM', morph=True, save_to_disk = False):
        """Apply inverse operator to epochs to get source estimates of each item"""


        lambda2 = 1.0 / snr ** 2.0

        inverse = mne.minimum_norm.read_inverse_operator( self.processed_files + self.subject + '_inv.fif' )

        eps = mne.minimum_norm.apply_inverse_epochs(epochs=epochs,inverse_operator=inverse,lambda2=lambda2,method = method)

        if morph == True:
            eps_morphed = []
            counter = 1
            morph_status = 'morphed'
            # create morph map
            # get vertices to morph to (we'll take the fsaverage vertices)
            subject_to = 'fsaverage'
            fs = mne.read_source_spaces(self.subjects_dir + '%s/bem/%s-ico-4-src.fif' % (subject_to, subject_to))
            vertices_to = [fs[0]['vertno'], fs[1]['vertno']]
            subject_from = self.subject

            for stc_from in eps:
                print "Morphing source estimate for epoch %d" %counter
            # use the morph function
                morph_mat = mne.compute_morph_matrix(subject_from, subject_to, vertices_from=stc_from.vertices, vertices_to=vertices_to, subjects_dir=self.subjects_dir)
                stc = mne.morph_data_precomputed(subject_from, subject_to, stc_from, vertices_to, morph_mat)
                # stc = mne.morph_data(subject_from, subject_to, stc_from, n_jobs=1,
                #                     grade=vertices_to, subjects_dir=self.subjects_dir)

                eps_morphed.append(stc)
                counter += 1

            eps = eps_morphed

        if save_to_disk:
            with open(op.join(self.stc_cont, '%s_stc_epochs.pickled' %self.subject), 'w') as fileout:
                pickle.dump(eps, fileout)
            #save.pickle(obj=eps,dest=self.stc_cont + self.subject + '_stc_epochs')

        return eps


    def make_stcs(self, snr = 3.0, method='dSPM', save_to_disk = False, morph=True):
        """docstring for make_stcs"""

        morph_status = 'no_morph'

        lambda2 = 1.0 / snr ** 2.0
        pick_ori = None

        stcs = []

        for evoked_object, cond in self.evoked_list:

            print "Making source estimates for %s." %cond
            stc = mne.minimum_norm.apply_inverse(evoked_object, self.inverse_solution, method = method, lambda2 = lambda2, pick_ori = pick_ori)

            if morph == True:

                morph_status = 'morphed'
                # create morph map
                # get vertices to morph to (we'll take the fsaverage vertices)
                subject_to = 'fsaverage'
                fs = mne.read_source_spaces(self.subjects_dir + '%s/bem/%s-ico-4-src.fif' % (subject_to, subject_to))
                vertices_to = [fs[0]['vertno'], fs[1]['vertno']]

                # info of the stc we're going to morph is just the present stc
                subject_from = self.subject
                stc_from = stc

                # use the morph function
                morph_mat = mne.compute_morph_matrix(subject_from, subject_to, vertices_from=stc_from.vertices, vertices_to=vertices_to, subjects_dir=self.subjects_dir)
                stc = mne.morph_data_precomputed(subject_from, subject_to, stc_from, vertices_to, morph_mat)


                # stc_to.save('%s_audvis-meg' % subject_from)
                # mne.compute_morph_matrix('2-COMB','3-COMB',stcs[0].vertices, stcs[5].vertices, subjects_dir=self.subjects_dir)

            if save_to_disk:
                fact_morph_cond_path = op.join(self.stc_fact, morph_status, cond)
                if not os.path.isdir(fact_morph_cond_path):
                    os.makedirs(fact_morph_cond_path)

                stc.save(op.join(fact_morph_cond_path, '%s_%s_%s' %(self.subject, cond, morph_status)))

                self.add_preprocessing_notes("Saved source estimates (%s) for %s." %(morph_status, cond))
