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


def evaluate(data_set_name, num_hidden_layers, units_per_layer, num_epochs, trial):
    dimension = 1
    training_file = os.path.join('../data', data_set_name + '_training.csv')
    testing_file = os.path.join('../data', data_set_name + '_testing.csv')
    model_dir = os.path.join(
        '../log', data_set_name, str(num_epochs), str(num_hidden_layers), str(units_per_layer), str(trial)
    )
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    training_set = learn.datasets.base.load_csv_without_header(training_file, numpy.float, numpy.float)
    testing_set = learn.datasets.base.load_csv_without_header(testing_file, numpy.float, numpy.float)
    feature_columns = [tensorflow.feature_column.numeric_column('x', [2 * dimension])]
    model = tensorflow.estimator.DNNRegressor([units_per_layer] * num_hidden_layers, feature_columns, model_dir)
    model.train(input_fn(training_set, num_epochs, ))
    predictions = list(model.predict(input_fn(testing_set, 1, )))
    actual_target = numpy.concatenate([prediction['predictions'] for prediction in predictions])
    return metrics.mean_squared_error(actual_target, testing_set.target)


def main():
    for data_set_name in ['airport', 'authors', 'collaboration', 'facebook', 'congress', 'forum']:
        errors = pandas.DataFrame(columns=['num_epochs', 'num_hidden_layers', 'units_per_layer', 'error', ])
        for num_epochs in range(1, 2, 1):
            for num_hidden_layers in range(2, 3, 1):
                for units_per_layer in range(10, 100, 10):
                    error = numpy.mean([
                        evaluate(
                            data_set_name, num_hidden_layers, units_per_layer, num_epochs, trial
                        ) for trial in range(30)
                    ])
                    errors.loc[len(errors)] = [num_epochs, num_hidden_layers, units_per_layer, error]
        print(errors)
        errors.to_csv('../log/' + data_set_name + '/errors.csv', index=False)


if __name__ == '__main__':
    main()
