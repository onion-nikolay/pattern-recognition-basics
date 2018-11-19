"""
@author: onion-nikolay
"""
import datetime
import pandas as pd
from os import listdir, rename
from os.path import join as pjoin
from cf import LINEAR_FILTERS, TEST
from cpr import classifier, session
from image_processing import BIN_METHODS

CLASSIFIER_TYPES = classifier.CLASSIFIER_TYPES
CLASSIFIER_TYPES_CF = classifier.CLASSIFIER_TYPES_CF

#PARAMS = {'folder': '/media/onion-nikolay/ADATA NH13/datasets/',
#          'classifier_type': ['cf'],
#          'dataset': ['render_energies'],
#          'train_object': [{'render_150': 1, 'false_train': 0}],
#          'test_object': [{'render_237': 1, 'false_test': 0}],
#          'classifier_is_save': True,
#          'filter_type': LINEAR_FILTERS,
#          'processing_method': [8]}

PARAMS = {'folder': '/media/onion-nikolay/ADATA NH13/datasets/',
          'classifier_type': ['cf'],
          'dataset': ['render_vs_photo'],
          'train_object': [{'render_b_256': 1, 'false_train': 0}],
          'test_object': [{'photo_b_256': 1, 'Abrams': 0, 'Chieftain': 0,
                           'Leopard': 0}],
          'classifier_is_save': True,
          'filter_type': LINEAR_FILTERS,
          'processing_method': [8]}


def getParams():
    is_save = PARAMS['classifier_is_save']
    LIST_OF_PARAMS = []
    for dataset in PARAMS['dataset']:
        for train_objects, test_objects in zip(PARAMS['train_object'],
                                               PARAMS['test_object']):

            train_object_folder = [pjoin(PARAMS['folder'], dataset, key)
                                   for key in train_objects.keys()]

            train_object_labels = [train_objects[key]
                                   for key in train_objects.keys()]

            test_object_folder = [pjoin(PARAMS['folder'], dataset, key)
                                  for key in test_objects.keys()]

            test_object_labels = [test_objects[key]
                                  for key in test_objects.keys()]

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
    rename(pjoin('data', 'graph'), pjoin('data', __today[:19]+' graph'))
    rename(pjoin('data', 'model'), pjoin('data', __today[:19]+' model'))
    df = pd.read_csv(name)
