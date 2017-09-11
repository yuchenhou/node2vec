import os
import shutil
import numpy
import pandas
from sklearn import model_selection, metrics, preprocessing
import tensorflow
from tensorflow.contrib import learn


def make_data_files(data_set_name, training_file, testing_file):
    embeddings_file = '../emb/' + data_set_name + '.emb'
    links_file = '../graph/' + data_set_name + '.tsv'
    embeddings = pandas.read_csv(embeddings_file, sep=' ', header=None, index_col=0, skiprows=1)
    embeddings.sort_index(inplace=True)
    links = pandas.read_csv(links_file, sep='\t', header=None)

    def embed(node_id):
        return ', '.join(str(element) for element in embeddings.loc[[node_id]].values.flatten())

    links[0] = links[0].map(embed)
    links[1] = links[1].map(embed)
    links[2] = preprocessing.MaxAbsScaler().fit_transform(numpy.log(links[2].values.reshape(-1, 1)))
    training_links, testing_links = model_selection.train_test_split(links, test_size=0.2)
    training_links.to_csv(training_file, index=False, header=False, quotechar=' ')
    testing_links.to_csv(testing_file, index=False, header=False, quotechar=' ')


def input_fn(data_set, num_epochs, ):
    return tensorflow.estimator.inputs.numpy_input_fn(
        {'x': numpy.array(data_set.data)}, numpy.array(data_set.target), num_epochs=num_epochs, shuffle=False,
    )


def evaluate(data_set_name, training_file, testing_file, dimension, num_epochs, trial):
    model_dir = os.path.join('../log', data_set_name, str(num_epochs), str(dimension), str(trial))
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    training_set = learn.datasets.base.load_csv_without_header(training_file, numpy.float, numpy.float)
    testing_set = learn.datasets.base.load_csv_without_header(testing_file, numpy.float, numpy.float)
    model = tensorflow.estimator.DNNRegressor(
        [dimension, dimension, ], [tensorflow.feature_column.numeric_column('x', [2 * dimension])], model_dir
    )
    model.train(input_fn(training_set, num_epochs, ))
    predictions = list(model.predict(input_fn(testing_set, 1, )))
    actual_target = numpy.concatenate([prediction['predictions'] for prediction in predictions])
    return metrics.mean_squared_error(actual_target, testing_set.target)


def main():
    errors = pandas.DataFrame(columns=['data_set_name', 'num_epochs', 'dimension', 'error', ])
    # for data_set_name in ['airport', 'collaboration', 'congress', 'forum',]:
    for data_set_name in ['airport', ]:
        training_file = os.path.join('../emb', data_set_name + '_training.csv')
        testing_file = os.path.join('../emb', data_set_name + '_testing.csv')
        for num_epochs in range(1, 11):
            for dimension in [4, ]:
                make_data_files(data_set_name, training_file, testing_file)
                error = numpy.mean([
                    evaluate(
                        data_set_name, training_file, testing_file, dimension, num_epochs, trial
                    ) for trial in range(10)
                ])
                errors.loc[len(errors)] = [data_set_name, num_epochs, dimension, error]
    print(errors)


if __name__ == '__main__':
    main()
