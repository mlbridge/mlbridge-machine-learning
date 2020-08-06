import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import confusion_matrix
from elasticsearch import Elasticsearch
import csv


def string_to_ascii(string):
    ascii_arr = np.zeros(len(string))
    for i in range(len(string)):
        ascii_arr[i] = ord(string[i])
    return ascii_arr


def import_data(string_to_ascii, data_path, labels, header, lateral_skip,
                no_of_entries, csv_txt):
    if csv_txt == 0:
        data = open(data_path, "r")
        data = list(data.readlines())
    else:
        data = open(data_path, 'rt')
        reader = csv.reader(data, delimiter=',', quoting=csv.QUOTE_NONE)
        data = list(reader)
        data = list(np.asarray(data[:no_of_entries + header])[:, 1])

    ret_data = np.zeros((no_of_entries, 256))

    for i in range(header, no_of_entries + header):
        ret_data[i - header, 0: len(data[i].strip('\"'))] = \
            string_to_ascii(data[i].strip('\"'))

    labels = np.ones((no_of_entries, 1)) * labels

    return ret_data, labels


def data_preprocessing(import_data, number_of_samples,
                       mal_data_address, benign_data_address):
    ret_data_mal, labels_mal = \
        import_data(string_to_ascii, mal_data_address, 1, 1, 0,
                    int(number_of_samples / 2), 0)
    ret_data_nmal, labels_nmal = \
        import_data(string_to_ascii, benign_data_address, 0, 1, 1,
                    int(number_of_samples / 2), 1)

    train_split = int(number_of_samples / 2 * 0.8)
    valid_split = int(number_of_samples / 2 * 0.9)
    test_split = int(number_of_samples / 2)

    train_set = np.append(ret_data_mal[0:train_split],
                          ret_data_nmal[0:train_split], axis=0)
    train_set = np.reshape(train_set, (train_split * 2, 16, 16, 1))
    np.random.seed(43)
    np.random.shuffle(train_set)
    labels_train_set = np.append(labels_mal[0:train_split],
                                 labels_nmal[0:train_split], axis=0)
    np.random.seed(43)
    np.random.shuffle(labels_train_set)

    valid_set = np.append(ret_data_mal[train_split:valid_split],
                          ret_data_nmal[train_split:valid_split], axis=0)
    valid_set = np.reshape(valid_set, ((valid_split - train_split) * 2, 16, 16, 1))
    np.random.seed(44)
    np.random.shuffle(valid_set)
    labels_valid_set = np.append(labels_mal[train_split:valid_split],
                                 labels_nmal[train_split:valid_split], axis=0)
    np.random.seed(44)
    np.random.shuffle(labels_valid_set)

    test_set = np.append(ret_data_mal[valid_split:test_split],
                         ret_data_nmal[valid_split:test_split], axis=0)
    test_set = np.reshape(test_set, ((test_split - valid_split) * 2, 16, 16, 1))
    np.random.seed(45)
    np.random.shuffle(test_set)
    labels_test_set = np.append(labels_mal[valid_split:test_split],
                                labels_nmal[valid_split:test_split], axis=0)
    np.random.seed(45)
    np.random.shuffle(labels_test_set)

    print('Train Shape:', np.shape(train_set), np.shape(labels_train_set))
    print('Validation Shape:', np.shape(valid_set), np.shape(labels_valid_set))
    print('Test Shape:', np.shape(test_set), np.shape(labels_test_set))

    return train_set, labels_train_set, valid_set, labels_valid_set, test_set, \
           labels_test_set


def model_definition():
    model = models.Sequential(name='DNS_Alert_Net')
    model.add(layers.Conv2D(16, (2, 2), activation='relu',
                            input_shape=(16, 16, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (2, 2), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    adam_ = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam_,
                  metrics=['accuracy'])
    return model


def training(es, model, model_name, epochs, batch_size, train_set,
             labels_train_set, validation_set, labels_validation_set):
    for i in range(epochs):
        history = model.fit(train_set, labels_train_set, batch_size=batch_size,
                            epochs=1, validation_data=(validation_set,
                                                       labels_validation_set))

        try:
            body = es.get(index=es.get(index='model', id=1)['_source']['name'],
                          id=1)['_source']
            body['training']['loss'].append(history.history['loss'][0] * 100)
            body['training']['val_loss'].append(history.history['val_loss'][0] * 100)
            body['training']['acc'].append(history.history['acc'][0] * 100)
            body['training']['val_acc'].append(history.history['val_acc'][0] * 100)
            body['training']['epochs'].append((i + 1))

            update_body = {'doc':
                               {'training':
                                    {'loss': (body['training']['loss']),
                                     'val_loss': (body['training']['val_loss']),
                                     'acc': (body['training']['acc']),
                                     'val_acc': (body['training']['val_acc']),
                                     'epochs': body['training']['epochs']
                                     }
                                }
                           }
            es.update(index=es.get(index='model', id=1)['_source']['name'],
                      id=1, body=update_body)
        except:
            print('Please check the Elasticsearch Server')

    print('Training Completed')
    return model


def model_evaluation_metrics(es, model, train_set, labels_train_set, valid_set,
                             labels_valid_set, test_set, labels_test_set):
    loss_train, acc_train = model.evaluate(train_set, labels_train_set)
    loss_valid, acc_valid = model.evaluate(valid_set, labels_valid_set)
    loss_test, acc_test = model.evaluate(test_set, labels_test_set)

    y_pred = model.predict(train_set)
    cf_matrix_train = confusion_matrix(labels_train_set, y_pred.round())

    y_pred = model.predict(valid_set)
    cf_matrix_valid = confusion_matrix(labels_valid_set, y_pred.round())

    y_pred = model.predict(test_set)
    cf_matrix_test = confusion_matrix(labels_test_set, y_pred.round())

    acc_train = (cf_matrix_train[0, 0] + cf_matrix_train[1, 1]) / \
                np.sum(cf_matrix_train)
    pres_train = (cf_matrix_train[1, 1]) / (cf_matrix_train[1, 1] +
                                            cf_matrix_train[0, 1])
    rec_train = (cf_matrix_train[1, 1]) / (cf_matrix_train[1, 1] +
                                           cf_matrix_train[1, 0])
    f1_train = 2 * rec_train * pres_train / (rec_train + pres_train)

    acc_valid = (cf_matrix_valid[0, 0] + cf_matrix_valid[1, 1]) / \
                np.sum(cf_matrix_valid)
    pres_valid = (cf_matrix_valid[1, 1]) / (cf_matrix_valid[1, 1] +
                                            cf_matrix_valid[0, 1])
    rec_valid = (cf_matrix_valid[1, 1]) / (cf_matrix_valid[1, 1] +
                                           cf_matrix_valid[1, 0])
    f1_valid = 2 * rec_valid * pres_valid / (rec_valid + pres_valid)

    acc_test = (cf_matrix_test[0, 0] + cf_matrix_test[1, 1]) / \
               np.sum(cf_matrix_test)
    pres_test = (cf_matrix_test[1, 1]) / (cf_matrix_test[1, 1] +
                                          cf_matrix_test[0, 1])
    rec_test = (cf_matrix_test[1, 1]) / (cf_matrix_test[1, 1] +
                                         cf_matrix_test[1, 0])
    f1_test = 2 * rec_test * pres_test / (rec_test + pres_test)

    update_body = {'doc':
                       {'metrics':
                            {'loss_train': loss_train, 'acc_train': acc_train,
                             'loss_valid': loss_valid, 'acc_valid': acc_valid,
                             'loss_test': loss_test, 'acc_test': acc_test,
                             'cf_matrix_train': cf_matrix_train, 'cf_matrix_valid': cf_matrix_valid,
                             'cf_matrix_test': cf_matrix_test,
                             'pres_train': pres_train, 'rec_train': rec_train, 'f1_train': f1_train,
                             'pres_valid': pres_valid, 'rec_valid': rec_valid, 'f1_valid': f1_valid,
                             'pres_test': pres_test, 'rec_test': rec_test, 'f1_test': f1_test

                             }
                        }
                   }

    try:
        es.update(index=es.get(index='model', id=1)['_source']['name'], id=1,
                  body=update_body)
    except:
        print('Please check the Elasticsearch Server')


if __name__ == '__main__':

    es = Elasticsearch()

    mal_data_path = '../data/malicious_domains.txt'
    benign_data_path = '../data/benign_domains.csv'

    train_set, labels_train_set, valid_set, labels_valid_set, test_set, \
    labels_test_set = data_preprocessing(import_data, 1000,
                                         mal_data_path, benign_data_path)
    while True:
        training_ = es.get(index='model', id=1)['_source']['training']
        if training_:
            body = {'training': {'loss': [], 'val_loss': [], 'acc': [],
                                 'val_acc': [], 'epochs': []},
                    'metrics': {'loss_train': 0, 'acc_train': 0, 'loss_valid': 0,
                                'acc_valid': 0, 'loss_test': 0, 'acc_test': 0,
                                'cf_matrix_train': 0, 'cf_matrix_valid': 0, 'cf_matrix_test': 0,
                                'pres_train': 0, 'rec_train': 0, 'f1_train': 0,
                                'pres_valid': 0, 'rec_valid': 0, 'f1_valid': 0,
                                'pres_test': 0, 'rec_test': 0, 'f1_test': 0}}
            es.index(index=es.get(index='model', id=1)['_source']['name'], id=1, body=body)
            model = model_definition()
            trained_model = training(es, model, es.get(index='model', id=1)['_source']['name'],
                                     es.get(index='model', id=1)['_source']['epochs'],
                                     es.get(index='model', id=1)['_source']['batch'],
                                     train_set, labels_train_set, valid_set, labels_valid_set)
            model_evaluation_metrics(es, trained_model, train_set, labels_train_set,
                                     valid_set, labels_valid_set, test_set, labels_test_set)
            name = es.get(index='model', id=1)['_source']['name']
            model.save('../saved_models/' + name + '.hdf5')

            update_body = {'doc': {'completed': 1, 'training': 0}}

            es.update(index='model', id=1, body=update_body)
