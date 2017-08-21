import networkx
import numpy
import pandas


def main():
    data_sets = ['airport', 'collaboration', 'congress', 'forum']
    degrees = pandas.DataFrame(
        [analyze(data_set_name)['average_degree'] for data_set_name in data_sets],
        data_sets,
        ['average_degree'],
    )
    print degrees
    axes = degrees.plot.bar(rot=0, figsize=(16, 8,), grid=True, )
    axes.set_xlabel('data_set')
    axes.set_ylabel('average_degree')
    axes.get_figure().savefig('../log/average_degree')


def analyze(data_set_name):
    data_set_file = '../graph/' + data_set_name + '.tsv'
    graph = networkx.read_weighted_edgelist(data_set_file, nodetype=int)
    return {
        'number_of_nodes': networkx.number_of_nodes(graph),
        'number_of_edges': networkx.number_of_edges(graph),
        'average_degree': numpy.average(networkx.degree(graph).values()),
    }


if __name__ == '__main__':
    main()
