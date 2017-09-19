import os
import shutil
import numpy
import pandas
from sklearn import metrics
import tensorflow
from tensorflow.contrib import learn


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
    feature_columns = [tensorflow.feature_column.numeric_column('x', [2 * 1])]
    model = tensorflow.estimator.DNNRegressor([dimension, dimension], feature_columns, model_dir)
    model.train(input_fn(training_set, num_epochs, ))
    predictions = list(model.predict(input_fn(testing_set, 1, )))
    actual_target = numpy.concatenate([prediction['predictions'] for prediction in predictions])
    return metrics.mean_squared_error(actual_target, testing_set.target)


def main():
    for data_set_name in ['airport', 'collaboration', 'congress', 'forum']:
        errors = pandas.DataFrame(columns=['num_epochs', 'dimension', 'error', ])
        training_file = os.path.join('../emb', data_set_name + '_training.csv')
        testing_file = os.path.join('../emb', data_set_name + '_testing.csv')
        for num_epochs in range(3, 4, 1):
            for dimension in range(10, 100, 10):
                error = numpy.mean([
                    evaluate(
                        data_set_name, training_file, testing_file, dimension, num_epochs, trial
                    ) for trial in range(10)
                ])
                errors.loc[len(errors)] = [num_epochs, dimension, error]
        print(errors)
        errors.to_csv('../log/' + data_set_name + '/errors.csv', index=False)


if __name__ == '__main__':
    main()
