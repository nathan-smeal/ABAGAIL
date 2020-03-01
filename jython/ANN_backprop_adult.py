"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
import os
import csv
import time
import sys
sys.path.append("../ABAGAIL.jar")

from func.nn.backprop import BackPropagationNetworkFactory,RPROPUpdateRule, BatchBackPropagationTrainer
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from time import clock # use clock instead of typical time because better resolution
from func.nn.activation import RELU

# from __future__ import with_statement

# INPUT_FILE = os.path.join("..", "data", "opt", "test", "adult.txt")
trainX = os.path.join("..","data", "trainX.csv")
trainY = os.path.join("..","data", "trainY.csv")
testX = os.path.join("..","data", "testX.csv")
testY = os.path.join("..","data", "testY.csv")

INPUT_LAYER = 6
HIDDEN_LAYER = 6
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 3500
# for tweaking
TRAINING_ITERATIONS = 350
num_iterations = 1
set_num = 0

def initialize_instances(fn):
    """Read the X and Y CSV data into a list of instances."""
    instances = []
    labelFn = fn.replace('X.csv','Y.csv')
    print labelFn

    # Read in the adult.txt CSV file
    with open(fn, "r") as adult:
        with open(labelFn, "r") as adulty:
            reader = csv.reader(adult)
            reader_label = csv.reader(adulty)
            
            skip = True
            for row,rowY in zip(reader, reader_label):
                if not skip:
                    instance = Instance([float(value) for value in row])
                    instance.setLabel(Instance(float(rowY[0])))
                    instances.append(instance)
                else:
                    skip = False
    return instances

def eval_instances(net, instances, measure):
    # get the accuracy of the set (training, test, validation)

    set_len = len(instances)
    right,wrong,error = 0,0,0.
    for i in instances:
        net.setInputValues(i.getData())
        net.run()
        # should only need first output binary class
        truth = i.getLabel().getContinuous()
        n_out = net.getOutputValues().get(0)

        if int(truth) == int(n_out):
            right +=1
        else:
            wrong += 1

        output = i.getLabel()
        output_values = net.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    accuracy = float(right)/float(set_len)
    error = error / float(set_len)


    return accuracy,error

def train(oa, network, oaName, train_i,test_i, measure):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] train_i:
    :param list[Instance] test_i:
    :param AbstractErrorMeasure measure:
    """
    print "\nTraining %s\n---------------------------" % (oaName,)
    rep_times = []
    reps = 1
    set_num = 0
    print(set_num)
    if len(sys.argv) > 1: 
        set_num = sys.argv[2]
    print(set_num)
    

    problem_name = "adult_nn"
    headers = ["Alg","Set_Num","Rep_num",
        "Train_acc","Test_acc","Train_err",
        "Test_err","Seconds"] # + list(keys)
    with open("{}-{}-{}.csv".format(problem_name, oaName, set_num), 'w') as out:
        
        out.write(','.join(headers) + os.linesep)
        
        for r in range(reps):
            for iteration in xrange(TRAINING_ITERATIONS):
                start = clock()

                oa.train()
                stop = clock()
                rep_times.append(stop - start)
                # make sampling rate the same as part 1
                if iteration % num_iterations == 0:
                    train_accuracy,train_error = eval_instances(network,train_i, measure)
                    test_accuracy,test_error = eval_instances(network,test_i, measure)
                    line = [oaName, r, iteration, train_accuracy, test_accuracy,train_error,test_error, rep_times[-1] ]
                    line = [str(x) for x in line]

                    out.write(','.join(line) + os.linesep)
                



def main():
    """Run algorithms on the adult dataset."""
    train_instances = initialize_instances(trainX)
    test_instances = initialize_instances(testX)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_instances)


    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["RHC", "SA", "GA", "BP"]
    print(sys.argv)
    if len(sys.argv) > 1:
        oa_names = [sys.argv[1]]
        set_num = sys.argv[2]
    # results = ""
    for name in oa_names:
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER],RELU())
        networks.append(classification_network)
        if name != "BP":
            nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))
        else:
            print("adding backprop")
            rule = RPROPUpdateRule()
            nnop.append(BatchBackPropagationTrainer(data_set, classification_network, measure, rule))


    if "RHC" in oa_names:
        rhc_index = oa_names.index("RHC")
        oa.append(RandomizedHillClimbing(nnop[rhc_index]))
    if "SA" in oa_names:
        sa_index = oa_names.index("SA")
        oa.append(SimulatedAnnealing(1E11, .95, nnop[sa_index]))
    if "GA" in oa_names:
        ga_index = oa_names.index("GA")
        oa.append(StandardGeneticAlgorithm(100, 50, 10, nnop[ga_index]))
    if "BP" in oa_names:
        rule = RPROPUpdateRule()
        bp_index = oa_names.index("BP")
        oa.append(nnop[bp_index])



    for i, name in enumerate(oa_names):
        train(oa[i], networks[i], oa_names[i], train_instances, test_instances, measure)



if __name__ == "__main__":
    main()

