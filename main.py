"""
@author: onion-nikolay
"""
import datetime
import pandas as pd
from os import listdir, rename
from cf import LINEAR_FILTERS, TEST
from cpr import classifier, session
from image_processing import BIN_METHODS

CLASSIFIER_TYPES = classifier.CLASSIFIER_TYPES
CLASSIFIER_TYPES_CF = classifier.CLASSIFIER_TYPES_CF

# PARAMS = {'folder': r'D:\Data\CPR\Images',
#          'classifier_type': ['cf_holo'],
#          'dataset': ['256_black', '256_white'],
#          'train_object': [{'T72 (train)': 1, 'Abrams': 0}],
#          'test_object': [{'T72 (true)': 1, 'Chieftain': 0, 'Leopard': 0}],
#          'classifier_is_save': True,
#          'filter_type': LINEAR_FILTERS,
#          'processing_method': [['bradley', 'random'],\
#                                ['bradley', 'quadratic'],\
#                                ['otsu', 'random'],\
#                                ['otsu', 'quadratic']]}

# PARAMS = {'folder': r'D:\Data\CPR\Images',
#          'classifier_type': CLASSIFIER_TYPES,
#          'dataset': listdir(r'D:\Data\CPR\Images'),
#          'train_object': [{'T72 (train)': 1, 'Abrams': 0}],
#          'test_object': [{'T72 (true)': 1, 'Chieftain': 0, 'Leopard': 0}],
#          'classifier_is_save': True,
#          'filter_type': LINEAR_FILTERS,
#          'processing_method': [8, 4, 2, [8, 'random'], [4, 'random'],
#                                [2, 'random'], [8, 'quadratic'],
#                                [4, 'quadratic'], [2, 'quadratic']]}

# PARAMS = {'folder': r'D:\Data\CPR\Images',
#          'classifier_type': ['cf_holo', 'cf'],
#          'dataset': listdir(r'D:\Data\CPR\Images'),
#          'train_object': [{'T72 (train)': 1, 'Abrams': 0}],
#          'test_object': [{'T72 (true)': 1, 'Chieftain': 0, 'Leopard': 0}],
#          'classifier_is_save': True,
#          'filter_type': LINEAR_FILTERS,
#          'processing_method': BIN_METHODS}

# TEST_PARAMS = {'folder': r'D:\Data\CPR\Images',
#               'classifier_type': ['cf_holo'],
#               'dataset': [listdir(r'D:\Data\CPR\Images')[-1]],
#               'train_object': [{'T72 (train)': 1, 'Abrams': 0}],
#               'test_object':
#                   [{'T72 (true)': 1, 'Chieftain': 0, 'Leopard': 0}],
#               'classifier_is_save': True,
#               'filter_type': TEST,
#               'processing_method': [BIN_METHODS[4]]}
#
# PARAMS_BIN_METHODS = {'folder': r'D:\Data\CPR\Images',
#                      'classifier_type': ['cf_holo'],
#                      'dataset': [listdir(r'D:\Data\CPR\Images')[0]],
#                      'train_object': [{'T72 (train)': 1, 'Abrams': 0}],
#                      'test_object':
#                          [{'T72 (true)': 1, 'Chieftain': 0, 'Leopard': 0}],
#                      'classifier_is_save': True,
#                      'filter_type': TEST,
#                      'processing_method': BIN_METHODS}

PARAMS_RENDER = {'folder': r'D:\cpr\datasets',
                 'classifier_type': ['cf'],
                 'dataset': ['object_50_f3_c30'],
                 'train_object': [{'T72_50_1': 1, 'abr_50_1': 0}],
                 'test_object':
                     [{'T72_50_2': 1, 'chi_50_1': 0, 'leo_50_1': 0}],
                 'classifier_is_save': True,
                 'filter_type': LINEAR_FILTERS,
                 'processing_method': [8]}

PARAMS_FACE = {'folder': r'G:\projects\patter-recognition-basics\cpr\datasets',
               'classifier_type': ['cf'],
               'dataset': ['face'],
               'train_object': [{'true_train': 1, 'false_train': 0}],
               'test_object': [{'true_p': 1, 'false_p': 0}],
               'classifier_is_save': True,
               'filter_type': LINEAR_FILTERS,
               'processing_method': [8]}


PARAMS = PARAMS_RENDER


def getParams():
    is_save = PARAMS['classifier_is_save']
    LIST_OF_PARAMS = []
    for dataset in PARAMS['dataset']:
        for train_objects, test_objects in zip(PARAMS['train_object'],
                                               PARAMS['test_object']):
            train_object_folder = [(PARAMS['folder']+'\\'+dataset+'\\'+key
                                    ) for key in train_objects.keys()]

            train_object_labels = [
                    train_objects[key] for key in train_objects.keys()]

            test_object_folder = [(PARAMS['folder']+'\\'+dataset+'\\'+key
                                   ) for key in test_objects.keys()]

            test_object_labels = [
                    test_objects[key] for key in test_objects.keys()]

            for clf in PARAMS['classifier_type']:
                if not(clf in CLASSIFIER_TYPES_CF):
                    name = clf + '_' + dataset
                    string = {'classifier_type': clf,
                              'train_object_folder': train_object_folder,
                              'train_object_labels': train_object_labels,
                              'test_object_folder': test_object_folder,
                              'test_object_labels': test_object_labels,
                              'classifier_is_save': is_save,
                              'classifier_name': name}
                    LIST_OF_PARAMS.append(string)
                else:
                    for flt_type in PARAMS['filter_type']:
                        if clf is 'cf':
                            name = clf + '_' + flt_type + '_ideal_' + dataset
                            string = {'classifier_type': clf,
                                      'train_object_folder':
                                          train_object_folder,
                                      'train_object_labels':
                                          train_object_labels,
                                      'test_object_folder': test_object_folder,
                                      'test_object_labels': test_object_labels,
                                      'classifier_is_save': is_save,
                                      'classifier_name': name,
                                      'filter_type': flt_type}
                            LIST_OF_PARAMS.append(string)
                        else:
                            for method in PARAMS['processing_method']:
                                if type(method) is list:
                                    method_name = "".join(str(
                                            method[0]).split())
                                    method_name = method_name + '_' + "".join(
                                            method[1].split())
                                else:
                                    method_name = "".join(str(method).split())
                                name = clf + '_' + flt_type + '_' + method_name
                                name += '_' + dataset
                                string = {'classifier_type': clf,
                                          'train_object_folder':
                                              train_object_folder,
                                          'train_object_labels':
                                              train_object_labels,
                                          'test_object_folder':
                                              test_object_folder,
                                          'test_object_labels':
                                              test_object_labels,
                                          'classifier_is_save': is_save,
                                          'classifier_name': name,
                                          'filter_type': flt_type,
                                          'classifier_processing': method}
                                LIST_OF_PARAMS.append(string)
    return LIST_OF_PARAMS


if __name__ == '__main__':

    s = session()
    name = s.start(getParams())
    __today = "-".join(str(datetime.datetime.today().isoformat()).replace(
            '.', ':').split(':'))
    rename('data\\graph', 'data\\'+__today[:19]+' graph')
    rename('data\\model', 'data\\'+__today[:19]+' model')
    df = pd.read_csv(name)
