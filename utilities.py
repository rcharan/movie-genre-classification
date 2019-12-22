import datetime
import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os
from joblib import dump, load


################################################################################
#
# Part 1 : Some simple convenience functions
#
################################################################################

def compose(*funcs):
    outer = funcs[:-1][::-1] # Python only provides left folds
    def composed_func(*args, **kwargs):
        inner = funcs[-1](*args, **kwargs)
        return functools.reduce(lambda val, func : func(val), outer, inner)
    return composed_func

# Aliases for filters and maps
lfilter        = compose(list, filter)
lmap           = compose(list, map)
afilter        = compose(np.array, list, filter)
filternull     = functools.partial(filter, None)
lfilternull    = compose(list, filternull)
filternullmap  = compose(filternull, map)
lfilternullmap = compose(lfilternull, map)
def lmap_filter(map_, predicate_):
    def f(list_):
        return lmap(map_, filter(predicate_, list_))
    return f

def list_diff(a, b):
    return list(set(a).difference(b))

def print_dict(d):
    key_len = max(map(len, d.keys()))
    for k, v in d.items():
        print(f'{k.ljust(key_len)} : {v}')

def drop_col(df, *cols):
    df.drop(columns = list(cols), inplace = True)

################################################################################
#
# Part 2: Plotting
#
################################################################################

# Convenience to deal with turning off interactive mode
def plot(fn, *args, **kwargs):
    if 'figsize' in kwargs:
        fig, ax = plt.subplots(figsize = kwargs['figsize'])
        del kwargs['figsize']
    else:
        fig, ax = plt.subplots()
    kwargs['ax'] = ax
    fn(*args, **kwargs)
    return fig

# Minimal implementation to deal with use cases encountered
#  does not implement the full seaborn API
# For simplicity, pass a list of series with the data to be counted
def stacked_countplot(*series, normalize = False, dropna = False):
    df   = pd.concat(map(lambda s : s.value_counts(normalize = normalize,
                                                   dropna    = dropna),
                         series),
                     axis = 'columns',
                     sort = True
                    ).fillna(0)
    temp = df.sort_values(df.columns[0]).values.cumsum(axis = 0)
    df = pd.DataFrame(temp, index = df.index, columns=df.columns
                     ).sort_values(df.columns[0], ascending = False)

    fig, ax = plt.subplots()
    cmp = sns.color_palette('muted', n_colors = 9)

    for i in range(len(df)):
        sns.barplot(x = 'variable', y = 'value',
                    data = df.iloc[[i]].melt(),
                    ax = ax,
                    label = df.index[i],
                    color = cmp.pop())
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_xlabel('')
    if normalize:
        ax.set_ylabel('Frequency')
    else:
        ax.set_ylabel('Count')
    return fig, ax

def rotate_labels(axes):
  labels = axes.get_xticklabels()
  for l in labels:
    l.set_rotation('vertical')


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Slight modification of sklearn's version to avoid re-predicting
#  See documentation there
def plot_confusion_matrix(estimator, y_pred, y_true, labels=None,
                          sample_weight=None, normalize=None,
                          display_labels=None, include_values=True,
                          xticks_rotation='vertical',
                          values_format=None,
                          cmap=sns.cubehelix_palette(light=1, as_cmap=True),
                          ax=None):
    if normalize not in {'true', 'pred', 'all', None}:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
                          labels=labels, normalize=normalize)

    if display_labels is None:
        if labels is None:
            display_labels = estimator.classes_
        else:
            display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    return disp.plot(include_values=include_values,
                     cmap=cmap, ax=ax, xticks_rotation=xticks_rotation)

################################################################################
#
# Part 2 : Timing
#
################################################################################

# Super simple timer
#  Timing implemented as class methods
#  to avoid having to instantiate
class Timer:

    @classmethod
    def start(cls):
        cls.start_time = datetime.datetime.now()

    @classmethod
    def end(cls):
        delta     = datetime.datetime.now() - cls.start_time
        sec       = delta.seconds
        ms        = delta.microseconds // 1000
        cls.time  = f'{sec}.{ms}'
        print(f'{sec}.{ms} seconds elapsed')


################################################################################
#
# Part 3 : Persistence and timing conveniences
#
################################################################################

def persist(obj_name, model, method, *data, task = 'model', force_fit = False, **kwargs):
    '''
        Persist a call (e.g. fit, fit_transform, score)

        Attempts to load from disk, otherwise makes the call and saves it

        task: either model, data, or both
    '''
    file_name = f'./models/{obj_name}.joblib'

    if os.path.exists(file_name) and not force_fit:
        print(f'Loading {obj_name} from disk')
        obj      = load(file_name)
        job_time = get_fit_time(obj_name)
        return obj

    else:
        assert task in ['model', 'data', 'both']
        print(f'Running {method}')
        Timer.start()
        return_val = model.__getattribute__(method)(*data, **kwargs)
        Timer.end()
        if task == 'model':
            out = model
        elif task == 'data':
            out = return_val
        elif task == 'both':
            out = (return_val, model)

        dump(out, file_name)
        write_fit_time(obj_name, Timer.time)
        return out


fit_time_fname = './models/fit_times.joblib'

def _fit_time_interface(model_name, write = None):
    if os.path.exists(fit_time_fname):
        fit_time_dict = load(fit_time_fname)
    else:
        if not write:
            print('No job time info found')
            return None
        fit_time_dict = {}

    if write:
        fit_time_dict[model_name] = write
        dump(fit_time_dict, fit_time_fname)
    else:
        if model_name in fit_time_dict:
            job_time = fit_time_dict[model_name]
            print(f'{job_time} seconds elapsed in original job')
            return job_time
        else:
            print(f'No job time found for {model_name}')
            return None

def get_fit_time(model_name):
    return _fit_time_interface(model_name)

def write_fit_time(model_name, fit_time):
    return _fit_time_interface(model_name, fit_time)



def get_learner(pipe_):
    return pipe_.named_steps['learn']

def get_best_learner(pipe_):
    return get_learner(pipe_).best_estimator_

def get_best_params(pipe_):
    return get_best_learner(pipe_).get_params()

def print_best_params(pipe_):
    return print_dict(get_best_params(pipe_))
