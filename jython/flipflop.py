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
from time import time
import os
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


def rhc_fac():
    rhc = RandomizedHillClimbing(hcp)
    return FixedIterationTrainer(rhc, num_iterations)

def sa_fac():
    sa = SimulatedAnnealing(100, .95, hcp)
    sfit = FixedIterationTrainer(sa, num_iterations)
    return sfit
def ga_fac():
    ga = StandardGeneticAlgorithm(20, 20, 0, gap)
    gfit = FixedIterationTrainer(ga, num_iterations)
    return gfit
def mimic_fac():
    mimic = MIMIC(50, 10, pop)
    mfit = FixedIterationTrainer(mimic, num_iterations)
    return mfit


trainers = {'rhc':rhc_fac, 'sa':sa_fac, 'ga': ga_fac, 'mimic':mimic_fac }
args = {'rhc':{}, 'sa':{}, 'ga': {}, 'mimic':{} }
# evaluators = {'ff':fff, 'tlp':fff, 'ga': ga_fac, 'mimic':mimic_fac }

reps = 10 # this is the checkin period for trainer
# todo make this a func that takes in the data


# since this is jython, can't do numpy and pandas
with open("flipflop.csv", 'w') as out:
        
    out.write("{},{},{},{},{}".format("Alg","Set_Num","Rep_num","Fitness","Seconds") + os.linesep)
    for name,factory in trainers.items():
        print "Examining " + name + " trainer..."
        # here implement hyper params
        rep_times = []
        for r in range(reps):
            # create a new trainer each time
            fit = factory()
            for i in range(0, max_iterations,num_iterations):
                start = time()
                fit.train()
                stop = time()
                rep_times.append(stop - start)
                fitness = fff.value(fit.trainer.getOptimal())
                # log
                out.write("{},{},{},{},{}".format(name,r,i,fitness,rep_times[-1]) + os.linesep)
            

        
        print "Done " + name + " trainer..."
