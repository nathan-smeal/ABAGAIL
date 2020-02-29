import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
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
import opt.example.CountOnesEvaluationFunction as CountOnesEvaluationFunction
import opt.example.FlipFlopEvaluationFunction as FlipFlopEvaluationFunction
from array import array
from time import clock # use clock instead of typical time because better resolution
import os
import itertools
# import pandas as pd
# import numpy as np
"""
Commandline parameter(s):
   none
"""

N=80
fill = [2] * N
ranges = array('i', fill)

fff = FlipFlopEvaluationFunction()
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(fff, odd, nf)
gap = GenericGeneticAlgorithmProblem(fff, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(fff, odd, df)


max_iterations = 3500
num_iterations = 10

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


hargs = {'rhc':{'na':[0]}, 
    'sa':{'t': [100, 10E8,10E9,10E10,10E11], 
    'cooling' : [0.5,0.55,0.6,0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]}, 
    'ga': {'populationSize':[10,20,100,200,2000],
    'toMate': [0.5,0.75,1.],
    'toMutate': [0,0.05,0.1]
    }, 
    'mimic':{'samples':range(30,60,5),'tokeep': [0.1,0.2,0.5]} }
evaluators = {'ff':fff, 'tlp':fff, 'ga': ga_fac, 'mimic':mimic_fac }
problem_name = "flipflop" # todo replace with for loop variable (dict of problem and name as key)

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
                for i in range(0, max_iterations,num_iterations):
                    start = clock()
                    fit.train()
                    stop = clock()
                    rep_times.append(stop - start)
                    func_eval = fff.func_evals
                    fitness = fff.value(fit.trainer.getOptimal())
                    # log
                    line = [name,r,i,fitness,rep_times[-1], func_eval] + list(exper)
                    line = [str(x) for x in line]
                    out.write(','.join(line) + os.linesep)
                

        
        print "Done " + name + " trainer..."
