"""
@author: onion-nikolay
"""

import pandas as pd
import numpy as np
import datetime
import os
import cv2 as cv
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from sklearn.svm import LinearSVC as svc
from inspect import getargspec
from joblib import load, dump
from os.path import join as pjoin

import cf
import helpers as hlp
from helpers import returnImages, returnFiles
from fft import ifft
from cf import synthesizeHolo
from image_processing import square
from image_processing import cfPreprocessing as preproc, cfProcessing as proc


def getDiscrChar(peaks, names, title=None, is_save=False, **kwargs):
    """\n    Returns image of discriminatory characteristic (to the file
    or figure).
    Parameters:
    -----------
    peaks : list of lists of floats
        Correlation peaks.
    names = list of str
        Names of objects in dataset.
    title : str (default=None)
        Title of plot.
    is_save : bool, default=False
        If True, images are saved, else they are shown in figures.
    **kwargs
        Can be used for sending of dataset name, threshold and other
        parameters.

    Returns
    -------
    error_key : int
        If 0, everything is OK.
    """
    error_key = 0
    plt.figure()
    norma = np.max(hlp.flattenList(peaks))
#    print(norma)
#    print(np.shape(peaks))
    x_range = max([len(cur_peaks) for cur_peaks in peaks])
    max_x = np.arange(x_range)

    for index in range(len(peaks)):
        x = np.arange(len(peaks[index]))
#        cur_peaks = np.array(peaks[index])/norma
        cur_peaks = [peak/norma for peak in peaks[index]]
        plt.plot(x, cur_peaks, label=names[index])
    try:
        threshold = kwargs['threshold']
        plt.plot(max_x, [threshold/norma]*len(max_x), 'k--', label='Threshold')
    except KeyError:
        pass
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.ylim((0, 1.05))

    if is_save:
        try:
            dataset = kwargs['dataset']
        except KeyError:
            error_key = 1
            dataset = 'Unknown'
        fig = plt.gcf()
        folder = pjoin('data', 'graph')
        try:
            os.mkdir(pjoin(folder, dataset))
        except OSError:
            pass
        fig.set_size_inches(18.5, 10.5)
        full_name = pjoin(folder, dataset, title) + '.png'
        fig.savefig(full_name, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return error_key


def getMetrics(true_labels, pred_labels, threshold=0.5):
    """\n    Returns matrics for classification experiment.
    Parameters
    ----------
    true_labels : list of float
    pred_labels : list of float
    threshold : float, default=0.5
        Threshold in CPR experiments.

    Returns
    -------
    _metric : dict
        Includes all of calculated metrics.
    """

    def getConfusionMatrix(confusion_matrix):
        s = "\n         |   Predicted   |\n\
        -----+-------+-------+\n\
        Real |   1   |   0   |\n\
        -----+-------+-------+\n\
           1 |{TP: ^7d}|{FN: ^7d}|\n\
        -----+-------+-------+\n\
           0 |{FP: ^7d}|{TN: ^7d}|\n\
        -----+-------+-------+"
        TP = confusion_matrix[1, 1]
        TN = confusion_matrix[0, 0]
        FN = confusion_matrix[1, 0]
        FP = confusion_matrix[0, 1]
        return s.format(TP=TP, FN=FN, FP=FP, TN=TN)

    from sklearn import metrics as mtr
    t = threshold
    pred_classes = hlp.flattenList([[int(elem > t) for elem in seq
                                     ] for seq in pred_labels])
    _metric = {}
    try:
        _metric.update({'accuracy': mtr.accuracy_score(true_labels,
                                                       pred_labels)})
    except ValueError:
        _metric.update({'accuracy': mtr.accuracy_score(true_labels,
                                                       pred_classes)})
    try:
        _metric.update({'confusion_matrix': getConfusionMatrix(
                mtr.confusion_matrix(true_labels, pred_labels))})
    except ValueError:
        _metric.update({'confusion_matrix': getConfusionMatrix(
                mtr.confusion_matrix(true_labels, pred_classes))})
    try:
        _metric.update({'f1': mtr.f1_score(true_labels, pred_labels)})
    except ValueError:
        _metric.update({'f1': mtr.f1_score(true_labels, pred_classes)})
    try:
        _metric.update({'precision': mtr.precision_score(true_labels,
                                                         pred_labels)})
    except ValueError:
        _metric.update({'precision': mtr.precision_score(true_labels,
                                                         pred_classes)})
    try:
        _metric.update({'recall': mtr.recall_score(true_labels, pred_labels)})
    except ValueError:
        _metric.update({'recall': mtr.recall_score(true_labels, pred_classes)})
    try:
        _metric.update({'report': mtr.classification_report(true_labels,
                                                            pred_labels)})
    except ValueError:
        _metric.update({'report': mtr.classification_report(true_labels,
                                                            pred_classes)})
    try:
        _metric.update({'ROC_AUC': mtr.roc_auc_score(true_labels,
                                                     pred_labels)})
    except ValueError:
        _metric.update({'ROC_AUC': mtr.roc_auc_score(true_labels,
                                                     pred_classes)})
    return _metric


def getPrediction(clf, path, return_class=True):
    """\n    Returns prediction for images from 'path'.
    Parameters
    ----------
    clf : cpr.classifier
    path : str
        Path to the folder with images to predict.
    return_class : bool, default=True
        See cpr.classifier.predict for more information.

    Returns
    -------
    predictions : list of float or lists of float
    """
    files = returnFiles(path)
    images = returnImages(files)
    if type(images[0]) is list:
        return [clf.predict(element, return_class) for element in images]
    else:
        return(clf.predict(images, return_class))


class classifier:
    """\n    Class for  classification model. The main part of the module.
    Use for predictions.
    """
    CLASSIFIER_TYPES = {'cf', 'cf_holo', 'svm'}
    CLASSIFIER_TYPES_CF = {'cf', 'cf_holo'}

    def __init__(self, clf_type, clf_name, clf_processing, **clf_args):
        """\n    Parameters
        ----------
        clf_type : str
            Should be in classifier.CLASSIFIER_TYPES
        clf_name : str
        clf_processing : int, str or list
            See image_processing.cfProcessing
        **clf_args
            For additional args of classifiers.
        """
        self.type = clf_type
        self.name = clf_name
        self.processing = clf_processing
        self.args = clf_args

    def __fitcf__(self, **args):
        train_objects_files = returnFiles(args['train_object_folder'])
        train_objects = returnImages(train_objects_files)
        transformed_train_objects = preproc(train_objects)
        filter_raw = cf.synthesize(
                transformed_train_objects,
                train_object_labels=args['train_object_labels'],
                filter_type=self.args['filter_type'])

        if self.type is 'cf':
            self.data = filter_raw
            self.__setthr__(transformed_train_objects,
                            args['train_object_labels'])
        else:
            filter_holo = synthesizeHolo(ifft(filter_raw))
            self.data = proc(filter_holo, self.processing, **self.args)
            transformed_train_objects = [
                    square(element, np.shape(self.data)[0], 0, True)
                    for element in transformed_train_objects
                                        ]
            self.__setthr__(transformed_train_objects,
                            args['train_object_labels'])

        if args['is_save']:
            self.__save()

    def __fitsvm__(self, **args):
        train_objects_files = returnFiles(args['train_object_folder'])
        train_objects = returnImages(train_objects_files)
        train_objects = preproc(train_objects)
        svc_args = dict(zip(getargspec(svc.__init__)[0][1:],
                            getargspec(svc.__init__)[3]))
        classifier_raw = svc(**hlp.chooseArgs(svc_args, self.args))
        number_of_objects = hlp.listLengths(train_objects)
        shp = np.shape(train_objects)
        train_objects = np.reshape(train_objects, (shp[0]*shp[1],
                                                   shp[2]*shp[3]))
        train_object_labels_expended = hlp.listExpend(
                args['train_object_labels'], number_of_objects)
        self.data = classifier_raw.fit(train_objects.tolist(),
                                       train_object_labels_expended)
        self.threshold = 0.5
        if args['is_save']:
            self.__save()

    def __load(self):
        """\n    For loading of classifier.
        In progress...
        """
        full_classifier_name = pjoin(
                'data', 'model', self.type, self.name) + '.pkl'
        with open(full_classifier_name, 'rb') as output:
            self.data = load(output)

    # Need to save all parameters of classificator
    def __save(self):
        """\n    For saving of classifier.
        In progress...
        """
        full_path = pjoin('data', 'model', self.type)
        full_classifier_name = pjoin(full_path, self.name) + '.pkl'
        if not(os.path.isdir(full_path)):
            try:
                os.mkdir(full_path)
            except OSError:
                raise OSError("directory {} can't be created!".format(
                        full_path))

        with open(full_classifier_name, 'wb') as output:
            dump(self.data, output)

        if self.type is 'cf':
            full_image_name = pjoin(full_path, self.name) + '.png'
            img = np.real(ifft(self.data))
            img = img.astype(float) * 255 / np.max(img)
            cv.imwrite(full_image_name, img)
        elif self.type is 'cf_holo':
            full_image_name = pjoin(full_path, self.name) + '.png'
            img = np.abs(self.data)
            img = img.astype(float) * 255 / np.max(img)
            cv.imwrite(full_image_name, img)

    # better calculation algotithm, remove DUMMY_THRESHOLDING
    def __setthr__(self, train_objects, train_object_labels):
        """\n    Set classifier's threshold.
        In progress...
        """
        is_holo = (self.type == 'cf_holo')
        true_objects = []
        false_objects = []
        for obj, label in zip(train_objects, train_object_labels):
            if label == 1:
                true_objects.append(obj)
            else:
                false_objects.append(obj)
        true_corr_outputs = cf.predict(self.data,
                                       hlp.flattenList(true_objects), 0,
                                       return_class=False, is_holo=is_holo)
        false_corr_outputs = cf.predict(self.data,
                                        hlp.flattenList(false_objects), 0,
                                        return_class=False, is_holo=is_holo)
        DUMMY_THRESHOLDING = True
        if DUMMY_THRESHOLDING:
            self.threshold = (np.mean(true_corr_outputs) + np.mean(
                    false_corr_outputs)) / 2
            return
        norma = np.max(true_corr_outputs + false_corr_outputs)
        x = np.arange(0, 1, 1e-5)
        norm_dist_true = hlp.norm_dist(np.array(true_corr_outputs)/norma, x)
        norm_dist_false = hlp.norm_dist(np.array(false_corr_outputs)/norma, x)
        nd_difference = norm_dist_true - norm_dist_false
        x0 = np.argmax(norm_dist_false)
        x1 = np.argmax(norm_dist_true)
        try:
            threshold = norma*(np.argmin(np.abs(nd_difference[x0:x1]))+x0)*1e-5
        except ValueError:
            threshold = norma*0.9*x1*1e-5
        if x0 > x1:
            print("Error! Threshold can't be set.")
        else:
            y0 = np.abs(nd_difference)
            for dx, dy in enumerate(y0[1:]):
                if np.abs(dy-y0[dx-1]) < np.max([1e-6, np.min(y0)]):
                    y0[dx] = 1
                else:
                    y0[dx] = 0
            y0[dx+1] = 0
            for dx in range(len(y0)-2):
                if dx < x0:
                    y0[dx+1] = 0
                elif ((y0[dx] == 1) and (y0[dx+2] == 1)):
                    y0[dx+1] = 1
            final_x = 0
            for dx in np.arange(x0, x1):
                if (y0[dx] == 1) and (final_x == 0):
                    final_x = dx
                elif (y0[dx] == 0) and (final_x != 0) and (y0[dx-1] == 1):
                    final_x = (final_x + dx) / 2
            if final_x != 0:
                threshold = norma * final_x * 1e-5
        self.threshold = threshold

    def fit(self, train_object_folder, train_object_labels, is_save):
        """\n    Fit classifier with train objects.
        Parameters
        ----------
        train_object_folder : str or list of str
            Path(s) of folders with train objects.
        train_object_labels : int or list of ints
            Labels of train objects
        is_save : bool
            If True, classifier are saving to the 'data' folder.
        """
        if self.type is 'svm':
            return self.__fitsvm__(train_object_folder=train_object_folder,
                                   train_object_labels=train_object_labels,
                                   is_save=is_save)
        elif self.type in classifier.CLASSIFIER_TYPES_CF:
            return self.__fitcf__(train_object_folder=train_object_folder,
                                  train_object_labels=train_object_labels,
                                  is_save=is_save)
        else:
            raise NameError('classifier {} not found'.format(self.type))

    def predict(self, data_to_predict, return_class=True):
        """\n    Returns predictions for input data.
        Parameters
        ----------
        data_to_predict : ndarray
            Image to be classified.
        return_class : bool, default=True
            If True, returns class, else returns correlation peak values.

        Returns
        -------
        prediction : float or int
        """
        if self.type in self.CLASSIFIER_TYPES_CF:
            is_holo = (self.type is 'cf_holo')
            size = np.shape(self.data)[0]
            try:
                sq_data_to_predict = square(data_to_predict, size, 0, True)
                return cf.predict(self.data, sq_data_to_predict,
                                  self.threshold, return_class=return_class,
                                  is_holo=is_holo)
            except KeyError:
                print("Error! Threshold value is not set!")
                return [0]*len(data_to_predict)

        elif self.type is 'svm':
            return self.data.predict([hlp.flattenImage(element)
                                      for element in data_to_predict])

        else:
            return []


class session():
    """\n    Class for one pattern recognition session with different
    parameters. Sholud be like Pipeline from sklearn in future."""

    def __init__(self):
        self.data = pd.DataFrame(columns=[
                'date', 'elapsed_time', 'classifier_type', 'classifier_name',
                'classifier_is_saved', 'classifier_processing',
                'classifier_args', 'train_object_folder',
                'train_object_labels', 'train_object_size', 'train_object_num',
                'test_object_folder', 'test_object_labels', 'test_object_num',
                'metrics_accuracy', 'metrics_confusion_matrix',
                'metrics_f1', 'metrics_precision', 'metrics_recall',
                'metrics_report', 'metrics_ROC_AUC'])

    def start(self, list_of_params):
        """\n    Start new session with input parameters."""
        try:
            os.makedirs(r'data\model')
        except OSError:
            pass
        try:
            os.makedirs(r'data\graph')
        except OSError:
            pass
        __start = timer()
        print("Session started succesfully.")
        params_combinations = list_of_params
        index = 0
        for params_sample in params_combinations:
            index += 1
            row = session.run(self, params_sample, index)
            self.data = self.data.append(row, sort=False)
        today = "-".join(str(datetime.datetime.today().isoformat()).replace(
                '.', ':').split(':'))
        name = pjoin('dataa', 'report_') + today[:19] + '.csv'
        self.data.to_csv(name)
        __finish = timer()
        __dt = __finish - __start
        __h = int(__dt/3600)
        __m = int((__dt - __h*3600)/60)
        __s = int(__dt - 3600*__h - 60*__m)
        print("Total elapsed time: {}h{}m{}s".format(__h, __m, __s))
        return name

    def run(self, params, index):
        """\n    Start session with fixed parameters. Returns row of DataFrame
        with input and output parameters."""
        __start = timer()
        try:
            clf_type = params['classifier_type']
            train_object_folder = params['train_object_folder']
            train_object_labels = params['train_object_labels']
            test_object_folder = params['test_object_folder']
            test_object_labels = params['test_object_labels']
            is_save = params['classifier_is_save']
            clf_name = params['classifier_name']
        except KeyError:
            print("Error! Some of necessary data is not found!")
            return None
        try:
            filter_type = params['filter_type']
        except KeyError:
            filter_type = None
        try:
            processing = params['classifier_processing']
        except KeyError:
            processing = None
        clf = classifier(clf_type, clf_name, processing,
                         filter_type=filter_type)
        clf.fit(train_object_folder, train_object_labels, is_save)
        folders = train_object_folder + test_object_folder
        labels = train_object_labels + test_object_labels
        labels_full = hlp.flattenList([([a] * len(b)) for a, b in zip(
                labels, returnFiles(folders))])
        predictions = getPrediction(clf, folders, False)
        names = [folder.split(os.sep)[-1] for folder in folders]
        dataset = folders[0].split(os.sep)[-2]
        getDiscrChar(predictions, names=names, title=clf_name,
                     is_save=is_save, threshold=clf.threshold,
                     dataset=dataset)
        metric = getMetrics(labels_full, predictions, threshold=clf.threshold)
        __finish = timer()
        clf_raw = clf.type+('' if filter_type is None else '_'+filter_type)
        if type(processing) is list:
            _processing = str(processing[0]) + '_' + str(processing[1])
        else:
            _processing = str(processing)
        clf_raw += ('_ideal' if processing is None else '_'+_processing)
        print("Dataset: {dat}, classifier: {clf}, elapsed time: {t} s".format(
                dat=dataset, clf=clf_raw, t=__finish-__start))

        df = pd.DataFrame(data=dict(
                date=datetime.datetime.today().isoformat(),
                elapsed_time=__finish-__start,
                classifier_type=clf_type,
                classifier_name=clf_name,
                classifier_is_saved=is_save,
                classifier_processing=_processing,
                classifier_args=None,
                train_object_folder=str(train_object_folder),
                train_object_labels=str(train_object_labels),
                train_object_size=None,
                train_object_num=None,
                test_object_folder=str(test_object_folder),
                test_object_labels=str(test_object_labels),
                test_object_num=None,
                metrics_accuracy=metric['accuracy'],
                metrics_confusion_matrix=metric['confusion_matrix'],
                metrics_f1=metric['f1'],
                metrics_precision=metric['precision'],
                metrics_recall=metric['recall'],
                metrics_report=metric['report'],
                metrics_ROC_AUC=metric['ROC_AUC']),
                          index=[index])
        return df
