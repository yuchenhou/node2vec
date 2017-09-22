import networkx
import numpy
import pandas


def analyze(data_set_name):
    data_set_file = '../graph/' + data_set_name + '.tsv'
    graph = networkx.read_weighted_edgelist(data_set_file, nodetype=int)
    return {
        'number_of_nodes': networkx.number_of_nodes(graph),
        'number_of_edges': networkx.number_of_edges(graph),
        'average_degree': numpy.average(networkx.degree(graph).values()),
    }


def plot(attribute):
    data_sets = ['airport', 'authors', 'collaboration', 'facebook', 'congress', 'forum']
    statistics = pandas.DataFrame(
        [analyze(data_set_name)[attribute] for data_set_name in data_sets],
        data_sets,
        [attribute],
    )
    print statistics
    axes = statistics.plot.bar(rot=0)
    axes.set_xlabel('data_set')
    axes.set_ylabel(attribute)
    axes.get_figure().savefig('../log/' + attribute)


def main():
    for attribute in ['number_of_nodes', 'number_of_edges', 'average_degree']:
        plot(attribute)


if __name__ == '__main__':
    main()
