from config import USER_CONFIG_DEFAULT_DATA_DIR, USER_CONFIG_DATESTAMP
import os.path as osp, time, atexit, os
import time
import json
import joblib
import numpy as np
from modules.serialization import *

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        """
        Initialize
        """
        self.output_dir = output_dir or "/tmp/experiments/%i"%int(time.time())
        if osp.exists(self.output_dir):
            print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
        else:
            os.makedirs(self.output_dir)
        # open the file to write
        self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        print(self.colorize("Logging data to %s"%self.output_file.name, 'green', bold=True))
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name



    def print_log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        print(self.colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    def write_config(self, config):
        """
        Log an experiment configuration.

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
        print(self.colorize('Saving config:\n', color='cyan', bold=True))
        print(output)
        with open(osp.join(self.output_dir, "config.json"), 'w') as out:
            out.write(output)

    def save_state(self, state_dict, itr=None):
        """
        Saves the state of an experiment.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.
        """
        filename = 'vars.pkl' if itr is None else 'vars%d.pkl'%itr
        try:
            joblib.dump(state_dict, osp.join(self.output_dir, filename))
        except:
            self.print_log('Warning: could not pickle state_dict.', color='red')

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        print_len = 22 + max_key_len
        print("-"*print_len)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g"%val if hasattr(val, "__float__") else val
            print(fmt%(key, valstr))
            vals.append(val)
        print("-"*print_len)
        self.write_tabular(vals)

        self.log_current_row.clear()
        self.first_row=False

    def write_tabular(self,vals):
        """
        Write all of the diagnostics to the output file.
        """
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers)+"\n")
            self.output_file.write("\t".join(map(str,vals))+"\n")
        self.output_file.flush()

    def colorize(self,string, color, bold=False, highlight=False):
        """
        Colorize a string.
        """
        attr = []
        num = color2num[color]
        if highlight: num += 10
        attr.append(str(num))
        if bold: attr.append('1')
        return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
    

def get_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of x

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    mean, std = x.mean(), x.std()
    if with_min_and_max:
        min_v = x.min()
        max_v = x.max()
        return mean, std, min_v, max_v
    return mean, std

class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to 
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use 

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you 
    would use 

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.

        To prevent problems, let value be either a numpy array, or a single scalar value
        """
        for k,v in kwargs.items():
            if not k in self.epoch_dict:
                self.epoch_dict[k] = []
            if isinstance(v, np.ndarray): # used to prevent problems due to shape issues
                v = v.reshape(-1)
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key,val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = get_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key, stats[0])
            if not(average_only):
                super().log_tabular('Std'+key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max'+key, stats[3])
                super().log_tabular('Min'+key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        return get_statistics_scalar(vals, with_min_and_max=True)



def subfoloder_setup(exp_name,seed,datestamp=False):
    # Make a seed-specific subfolder in the experiment directory.
    if datestamp:
        hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
    else:
        subfolder = ''.join([exp_name, '_s', str(seed)])
    return subfolder


def logger_kwargs_setup(exp_name, seed=None, data_dir=None, datestamp=False):
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.

    If no seed is given and datestamp is false, 

    ::

        output_dir = data_dir/exp_name

    If a seed is given and datestamp is false,

    ::

        output_dir = data_dir/exp_name/exp_name_s[seed]

    If datestamp is true, amend to

    ::

        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]

    You can force datestamp=True by setting ``USER_CONFIG_DATESTAMP=True`` in 
    ``config.py``. 

    Args:

        exp_name (string): Name for experiment.

        seed (int): Seed for random number generators used by experiment.

        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``.

        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.

    Returns:

        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Datestamp forcing
    datestamp = datestamp or USER_CONFIG_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])

    if seed is not None:
        subfolder = subfoloder_setup(exp_name,seed,datestamp)
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or USER_CONFIG_DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath),
                         exp_name=exp_name)
    return logger_kwargs


