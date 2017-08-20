import os
import shutil
import numpy
import pandas
from sklearn import model_selection, metrics, preprocessing
import tensorflow
from tensorflow.contrib import learn


def make_data_files(embeddings_file, links_file, training_file, testing_file):
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


def input_fn(data_set):
    return tensorflow.estimator.inputs.numpy_input_fn(
        {'x': numpy.array(data_set.data)}, numpy.array(data_set.target), shuffle=False
    )


def evaluate(model_dir, training_file, testing_file):
    dimension = 4
    training_set = learn.datasets.base.load_csv_without_header(training_file, numpy.float, numpy.float)
    testing_set = learn.datasets.base.load_csv_without_header(testing_file, numpy.float, numpy.float)
    model = tensorflow.estimator.DNNRegressor(
        [dimension, dimension], [tensorflow.feature_column.numeric_column('x', [2 * dimension])], model_dir
    )
    model.train(input_fn(training_set))
    predictions = list(model.predict(input_fn(testing_set)))
    actual_target = numpy.concatenate([prediction['predictions'] for prediction in predictions])
    return metrics.mean_squared_error(actual_target, testing_set.target)


def main():
    model_dir = '../log'
    training_file = '../emb/training.csv'
    testing_file = '../emb/testing.csv'
    embeddings_file = '../emb/countryLevelCollaboration.emb'
    links_file = '../graph/countryLevelCollaboration.tsv'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    make_data_files(embeddings_file, links_file, training_file, testing_file)
    print evaluate(model_dir, training_file, testing_file)


if __name__ == '__main__':
    main()
