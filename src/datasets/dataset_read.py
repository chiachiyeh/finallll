import sys

sys.path.append('../loader')
from .adapt_data_loader import UnalignedDataLoader
from .svhn import load_svhn
from .mnist import load_mnist
from .usps import *
from .gtsrb import load_gtsrb
from .synth_traffic import load_syntraffic

from .real_object import *


def return_dataset(data, scale=False, usps=False, all_use='no'):
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist(scale=scale, usps=usps, all_use=all_use)
        print(train_image.shape)
    if data == 'usps':
        train_image, train_label, \
        test_image, test_label = load_usps(all_use=all_use)
    if data == 'synth':
        train_image, train_label, \
        test_image, test_label = load_syntraffic()
    if data == 'gtsrb':
        train_image, train_label, \
        test_image, test_label = load_gtsrb()



    if data == 'object':
        train_image, train_label,test_image, test_label = load_objects()


        return train_image, train_label, test_image, test_label

    if data == 'mnist_style_object':
        train_image, train_label, test_image, test_label  = load_mnist_style_objects()
        return train_image, train_label, test_image, test_label



def dataset_read(source, target, batch_size, scale=False, all_use='no'):
    S = {}
    S_test = {}
    T = {}
    T_test = {}
    usps = False
    if source == 'usps' or target == 'usps':
        usps = True

    train_source, s_label_train, test_source, s_label_test = return_dataset(source, scale=scale,
                                                                            usps=usps, all_use=all_use)

    # t_label doesn't exist
    train_target, t_label_train, test_target, t_label_test = return_dataset(target, scale=scale, usps=usps,
                                                                            all_use=all_use)

    S['imgs'] = train_source
    S['labels'] = s_label_train
    T['imgs'] = train_target
    T['labels'] = t_label_train # dummy
    print('train_source:',S['imgs'].shape)
    print('train_target:',T['imgs'].shape)
    # input target samples for both
    S_test['imgs'] = test_source
    S_test['labels'] = s_label_test
    T_test['imgs'] = test_target
    T_test['labels'] = t_label_test #dummy
    print('test_source:',S_test['imgs'].shape)
    print('test_target:',T_test['imgs'].shape)

    scale = 40 if source == 'synth' else 28 if source == 'usps' or target == 'usps' else 32
    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()
    test_loader = UnalignedDataLoader()
    # origin method
    #test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)

    # don't know source
    test_loader.initialize(T_test, T_test, batch_size, batch_size, scale=scale)

    dataset_test = test_loader.load_data()
    return dataset, dataset_test
