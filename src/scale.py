import pandas
import numpy
from sklearn import preprocessing, model_selection


def make_data_files(data_set_name, training_file, testing_file):
    embeddings_file = '../data/' + data_set_name + '.emb'
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


def main():
    for data_set_name in ['airport', 'authors', 'collaboration', 'facebook', 'congress', 'forum']:
        training_file = '../data/' + data_set_name + '_training.csv'
        testing_file = '../data/' + data_set_name + '_testing.csv'
        make_data_files(data_set_name, training_file, testing_file)


if __name__ == '__main__':
    main()
