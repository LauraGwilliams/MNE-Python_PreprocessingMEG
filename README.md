# MnePreproc Class for Factorial and Single-Trial Designs
[Laura Gwilliams](https://github.com/LauraGwilliams) & [Samir Reddigari](https://github.com/reddigari), [Neuroscience of Language Lab, NYU](https://psych.nyu.edu/nellab)


## Overview
This class is a wrapper for [`mne-python`](http://martinos.org/mne/stable/index.html) designed to simplify the preprocessing of MEG data from filtering through source estimation.

## Dependencies
- [`mne-python`](http://martinos.org/mne/stable/index.html)
- [`eelbrain`](https://github.com/christainbrodbreck/Eelbrain)

## Usage
### Initialization
The preprocessing class is instantiated as follows. In this tutorial, the object will arbitrarily be called `exp`.

`exp = MnePreproc(subject, experiment, experiment-root, trigger_scheme, condition_format)`

with these arguments:
- `subject`: the subject number or code
- `experiment`: the experiment name (must be consistent with data filenames as explained below)
- `experiment-root`: directory containing raw and processed MEG and MRI data
- `trigger_scheme`: dictionary with keys corresponding to condition names and values indicating the relevant trigger
- `condition_format`: a string indicating the structure of the condition names for a factorial design, e.g. `'factorA_factorB_factorC'`

Initializing the class will automatically create several directories where processed are to be stored.

### Loading data

#### `exp.load_raw(filename=None)`

-  `raw_name`: name of the data file to be loaded. If unspecified, defaults to  `<root>/raw_files/<subject>/meg/<subject>_<experiment>-raw.fif`. If your naming convention differs, provide the full file path.

- Creates `exp.raw_unfiltered`


If the raw data has been split into multiple files, the method assumes any subsequent files are included in the header to the initial `*-raw.fif` file. This is the default case if your `.fif` files were generated using mne's `kit2fiff` functionality. If the files are truly separate and not loaded together, you may investigate the `load_raw_from_blocknos` method, which loads and concatenates multiple files. The code for this method will need to be significantly altered to fit your particular filenames.

When the data are loaded, the function will report the total number of triggers detected in the recording, as well as the number of triggers detected whose values are included in the `trigger_scheme` argument. This is a good sanity check to ensure all events were identified. If these numbers are inaccurate, you may be able to correct the issue by fidgeting with the parameters of the `find_event` method documented below.

Finally, this method searches for a file with the naming convention `*bad_channels.txt` in the subjects processed files directory (`<root>/processed_files/<subject>/`), which should include the channel names of any channels to be excluded from the analysis, and should contain one channel name per line. If such channels were identified during the recording, they should be added prior to the start of preprocessing. If they are identified during epoch rejection, the safest way to ensure their removal is to add them to the text file and begin preprocessing anew. Alternatively, one can use the `mne.Epochs.drop_channels` method to remove the channels from the cleaned epochs object after epoch rejection is complete.

#### `exp.load_filtered(filter='0-40', filename=None)`

- `filter`: string indicating the low and high frequency bounds for the filtered data file, separated by a hyphen.
- `filename`: filtered data to be loaded. If left unspecified, defaults to
`processed_files/<subject>/<subject>_<experiment>_<low>-<high>Hz.fif`. If your naming convention differs, provide the full file path as `filename`.
- Creates `exp.raw`

This method may be used if you wish to start with already filtered data.

### Filtering data

####  `exp.filter_raw(low_bound=0, high_bound=40, method='iir', save_to_disk=False)`
- `low_bound` and `high_bound`: cutoff frequencies for filter
- `method`: filtering method (see `mne.io.Raw.filter` for documentation)
- `save_to_disk`: if True, saves filtered data to file `processed_files/<subject>/<subject>_<experiment>_<low>-<high>Hz.fif`
- Creates `exp.raw`

Operates only on `exp.raw_unfiltered`.

### Working with events

#### `exp.find_events(stim_channel='STI 014', min_duration=0.002)`
- `stim_channel`: channel to read for triggers. Uses MNE default.
- `min_duration`: minimum length (in seconds) of trigger to be reported
- Creates `exp.raw_events_all` (`numpy.ndarray` returned by `mne.find_events`)
- Creates `exp.raw_events` by subsetting all events to triggers in `trigger_scheme`

Operates only on `exp.raw`.

#### `exp.configure_conditions()`
- Creates `exp.factors`, `exp.factorial_trigger_scheme`, `exp.inverted_triggers`, and `exp.inverted_triggers_levels`

 This method uses `condition_format` given at initialization to identify the factorial design from the condition names in `trigger_scheme`. To ensure this process occurred correctly, examine `exp.factorial_trigger_scheme`, which is a dictionary (keys=factors) of dictionaries (keys=levels, values=triggers). For example:
```
{factorA: {'level1': [1, 2, 4, 8],
             'level2': [16, 32, 64, 128]},
   factorB: {'level1': [1, 2, 16, 32],
             'level2': [4, 8, 64, 128]}}
```
The other objects created by this method are easy to understand if examined, but are not critical to understand.

#### `exp.add_info_to_events()`
- Creates `exp.ds_events` (Eelbrain dataset)

This method operates on `exp.raw_events`, and generates an `eelbrain.Dataset`, (which is a turbo-charged `OrderedDict` from the `collections` module), that contains timing, trigger, and condition information derived from `trigger_scheme`. If there is information regarding a factorial design obtained from `configure_conditions`, columns are added specifying the level of each factor for each event.

### Epoching

#### `exp.make_epochs(events, tmin, tmax, baseline=(None,0), proj=True, preload=True)`
- `events`: events array indicating triggers to be used for epoching (e.g., `exp.raw_events`)
- `tmin` and `tmax`: start and end times (in seconds) for epochs_clean
- `baseline`: tuple indicating what segment of the epochs to baseline correct. Default (None, 0) means beginning of epoch to time 0.
- Creates `exp.epochs`

Epoch data around provided array of events. Channels present in `exp.raw.info['bads']` are excluded prior to epoching.


#### `exp.gui_blink_reject(epochs, eye_channels='AD')`
- `epochs`: epochs to visualize for epoch rejection (e.g. `exp.epochs`)
- `eye_channels` (list of channels | 'AD' | 'NY'): channels to mark as nearest to the eyes. If 'AD' or 'NY', uses the appropriate channels for NYU's two systems.

Opens the eelbrain epoch selection GUI and displays the epochs provided. Click on an epoch to reject it.

**IMPORTANT: Adding channels to the Bad Channels dialog in the GUI does not automatially remove them from the data. This must be done through one of the methods described in `load_raw()`.**

#### `exp.apply_blink_rejections()`
- Creates `exp.epochs_clean`

Method is called automatically upon closure of the epoch selection GUI. If a rejection file is already present, this method can be used independently of the GUI. Any file containing the string `'reject'` is considered a valid rejection file, and the user will be warned if zero or more than one are found. Currently loads text or eelbrain pickled files into an `eelbrain.Dataset`, and removes epochs tagged for rejection.

### Creating evokeds from epochs

#### `exp.make_evoked_per_condition(factorial=False, combine_ids=None)`
- `factorial`: boolean indicating whether to create evoked objects for individual levels of factors
- `combine_ids`: dictionary indicating conditions to be combined and the combination name
- Creates `exp.evokeds` and `exp.evoked_list`

Operates on `exp.epochs_clean`. Generates evoked object for each condition present in the cleaned epochs. If some of these conditions need to combined, provide `combine_ids` as a dictionary (key=new condition name, value=list of conditions to be combined), e.g. `{'combo_condition': ['condition1', 'condition3']}`

### Source estimation

#### `exp.compute_covariance_matrix(tmax=0, method='auto')`
- `tmax`: maximum timepoint in epoch to include in calculation of noise covariance matrix. Default of 0 corresponds to baseline period of each epoch.
- `method`: method for covariance computation (see `mne.cov.compute_covariance`)
- Creates `exp.cov_reg`

Computes noise covariance matrix from `exp.epochs_clean` using the method provided, and subsequently regularizes the matrix using factors of 0.05 for magnetometers and gradiometers.

If regularized covariance matrix already exists in `<root>/processed_files/<subject>`, it is loaded instead of computed. If not, it is computed and saved.

#### `exp.compute_forward_and_inverse_solutions(orientation='fixed')`
- `orientation` ('fixed' | 'free'): signed or unsigned source estimates (constrained normal to cortical surface or not)
- Creates `exp.forward_solution` and `exp.inverse_solution`

This method requires the following files in the indicated locations:
- `MRI trans file`: `<root>/processed_files/<subject>/<subject>_trans.fif`
- `source space`: `<root>/mri/<subject>/bem/*-ico-4-src.fif`
- `bem`: `<root>/mri/<subject>/bem/*-bem-sol.fif`

Computes forward solution (or loads from `<root>/processed_files/<subject>/<subject>_forward.fif` if already saved) for the provided orientation. Computes associated inverse operator.

#### `exp.make_stcs(self, snr = 3.0, method='dSPM', save_to_disk=False, morph=True)`
- `snr`: signal to noise ratio
- `method`: source estimation method (see `mne.minimum_norm.apply_inverse` for documentation)
- `save_to_disk`: if True, saves stcs as `<root>/processed_files/stcs/factorial/<morph_status>/condition/<subject>_<condition>_<morph_status>-<hemisphere>.stc`

Performs source estimation on each evoked object in `exp.evoked_list`, generating an stc file for each condition and level. These are the source space data you will eventually use for statistical testing of differences between conditions.

#### `exp.make_epochs_stcs(epochs, snr=2.0, method='dSPM', save_to_disk=False)`
- `epochs`: epochs object on which to perform source estimation
- `snr`: signal to noise ratio (2.0 is recommended for single-trial analysis)
- `method`: source estimation method (see `mne.minimum_norm.apply_inverse` for documentation)
- `save_to_disk`: if True, saves pickled stcs as `<root>/processed_files/stcs/continuous_var/<subject>_stc_epochs.pickled`, but this is not recommended as files may be prohibitively large.

Performs source estimation on provided epochs for use in regression analysis of source space data.

## Example Pipeline
```
triggers = {'condition1': 1, 'condition2': 4,
            'condition3': 8, 'condition4': 16}

subject = 'A0001'

exp = MnePreproc(subject, 'MyExp', '/Users/user/Experiments/MyExp', triggers)

exp.load_raw()
exp.filter_raw(low_bound=1)
exp.find_events()
exp.configure_conditions()
exp.add_info_to_events()
exp.make_epochs(exp.raw_events, tmin=-0.2, tmax=1.2, baseline=(-0.2,0))
exp.gui_blink_reject()
exp.make_evoked_per_condition()
exp.compute_covariance_matrix()
exp.compute_forward_and_inverse_solutions(orientation='free')
exp.make_stcs(save_to_disk=True)
exp.save_preprocessing_notes()
```
