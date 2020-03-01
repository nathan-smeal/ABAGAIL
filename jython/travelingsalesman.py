# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes 
# to a file and plot them in your favorite tool.
import sys
import os
import time
import itertools
sys.path.append("../ABAGAIL.jar")

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array
from time import clock # use clock instead of typical time because better resolution




"""
Commandline parameter(s):
    none
"""

max_iterations = 3500
num_iterations = 10
# set N value.  This is the number of points
N = 50
random = Random()

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()

ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

# # for mimic we use a sort encoding
efm = TravelingSalesmanSortEvaluationFunction(points)
fill = [N] * N
ranges = array('i', fill)
oddm = DiscreteUniformDistribution(ranges)
dfm = DiscreteDependencyTree(.1, ranges); 
pop = GenericProbabilisticOptimizationProblem(efm, oddm, dfm)


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z
def rhc_fac(args = {}):
    constant_params = {'hcp':hcp}
    params = merge_two_dicts(args, constant_params)
    print(params)
    rhc = RandomizedHillClimbing(hcp)
    return FixedIterationTrainer(rhc, num_iterations)

def sa_fac(args = {}):
    constant_params = {'hcp':hcp}
    params = merge_two_dicts(args,constant_params)

    sa = SimulatedAnnealing(args['t'], args['cooling'], hcp)
    # sa = SimulatedAnnealing(**params)

    sfit = FixedIterationTrainer(sa, num_iterations)
    return sfit
def ga_fac(args = {}):
    constant_params = {'hcp':hcp}
    params = merge_two_dicts(args,constant_params)
    ga = StandardGeneticAlgorithm(args['populationSize'], int(args['populationSize'] * args['toMate']), int(args['populationSize'] * args['toMutate']), gap)
    gfit = FixedIterationTrainer(ga, num_iterations)
    return gfit
def mimic_fac(args = {}):
    constant_params = {'op':pop}
    params = merge_two_dicts(args,constant_params)
    mimic = MIMIC(50, 10, pop)
    mimic = MIMIC(args['samples'],int(args['samples'] * args['tokeep']), pop)

    mfit = FixedIterationTrainer(mimic, num_iterations)
    return mfit




trainers = {'rhc':rhc_fac, 'sa':sa_fac, 'ga': ga_fac, 'mimic':mimic_fac }


# hargs = {'rhc':{'na':[0]}, 
#     'sa':{'t': [100, 10E8,10E9,10E10,10E11], 
#     'cooling' : [0.5,0.55,0.6,0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]}, 
#     'ga': {'populationSize':[10,20,100,200,2000],
#     'toMate': [0.5,0.75,1.],
#     'toMutate': [0,0.05,0.1]
#     }, 
#     'mimic':{'samples':range(30,60,5),'tokeep': [0.1,0.2,0.5]} }

hargs = {'rhc':{'na':[0]}, 
    'sa':{'t': [100, 10E8,10E9,10E10,10E11], 
    'cooling' : [0.10,0.4,0.6,0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]}, 
    'ga': {'populationSize':[10,20,100,200],
    'toMate': [0.5,0.75,1.],
    'toMutate': [0,0.05,0.1]
    }, 
    'mimic':{'samples':range(30,60,10),'tokeep': [0.1,0.2,0.5]} }
evaluators = {'ff':ef, 'tlp':ef, 'ga': ga_fac, 'mimic':mimic_fac }
problem_name = "tsp" # todo replace with for loop variable (dict of problem and name as key)

reps = 10 # this is the checkin period for trainer
# todo make this a func that takes in the data


# since this is jython, can't do numpy and pandas
    
for name,factory in trainers.items():
    with open("{}-{}.csv".format(problem_name, name), 'w') as out:
        # get the hyper params ready
        hypers = hargs[name] 
        keys, values = zip(*hypers.items())
        headers = ["Alg","Set_Num","Rep_num","Fitness","Seconds", "Func_Evals"] + list(keys)
        out.write(','.join(headers) + os.linesep)
        print "Examining " + name + " trainer..."

        # if len(hypers.values()) > 0:
        for exper in itertools.product(*values):

            # here implement hyper params
            rep_times = []
            for r in range(reps):
                # create a new trainer each time
                row = {}
                for key, value in zip(keys, exper):
                    row[key] = value
                print(row)
                fit = factory(row)
                # have to reset func eval per run 
                ef.func_evals = 0
                for i in range(0, max_iterations,num_iterations):
                    start = clock()
                    fit.train()
                    stop = clock()
                    rep_times.append(stop - start)
                    func_eval = ef.func_evals
                    fitness = ef.value(fit.trainer.getOptimal())
                    # log
                    line = [name,r,i,fitness,rep_times[-1], func_eval] + list(exper)
                    line = [str(x) for x in line]
                    out.write(','.join(line) + os.linesep)
                

        
        print "Done " + name + " trainer..."



# rhc = RandomizedHillClimbing(hcp)
# fit = FixedIterationTrainer(rhc, 200000)
# fit.train()
# print "RHC Inverse of Distance: " + str(ef.value(rhc.getOptimal()))
# print "Route:"
# path = []
# for x in range(0,N):
#     path.append(rhc.getOptimal().getDiscrete(x))
# print path

# sa = SimulatedAnnealing(1E12, .999, hcp)
# fit = FixedIterationTrainer(sa, 200000)
# fit.train()
# print "SA Inverse of Distance: " + str(ef.value(sa.getOptimal()))
# print "Route:"
# path = []
# for x in range(0,N):
#     path.append(sa.getOptimal().getDiscrete(x))
# print path


# ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
# fit = FixedIterationTrainer(ga, 1000)
# fit.train()
# print "GA Inverse of Distance: " + str(ef.value(ga.getOptimal()))
# print "Route:"
# path = []
# for x in range(0,N):
#     path.append(ga.getOptimal().getDiscrete(x))
# print path


# # for mimic we use a sort encoding
# ef = TravelingSalesmanSortEvaluationFunction(points);
# fill = [N] * N
# ranges = array('i', fill)
# odd = DiscreteUniformDistribution(ranges);
# df = DiscreteDependencyTree(.1, ranges); 
# pop = GenericProbabilisticOptimizationProblem(ef, odd, df);

# mimic = MIMIC(500, 100, pop)
# fit = FixedIterationTrainer(mimic, 1000)
# fit.train()
# print "MIMIC Inverse of Distance: " + str(ef.value(mimic.getOptimal()))
# print "Route:"
# path = []
# optimal = mimic.getOptimal()
# fill = [0] * optimal.size()
# ddata = array('d', fill)
# for i in range(0,len(ddata)):
#     ddata[i] = optimal.getContinuous(i)
# order = ABAGAILArrays.indices(optimal.size())
# ABAGAILArrays.quicksort(ddata, order)
# print order
