#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#

echo $CLASSPATH

export CLASSPATH=../ABAGAIL.jar:$CLASSPATH
mkdir -p data/plot logs image
export TERM=xterm-color

echo $CLASSPATH

# # four peaks
# echo "four peaks"
# jython fourpeaks.py

# # count ones
# echo "count ones"
# jython countones.py

# # continuous peaks
# echo "continuous peaks"
# jython continuouspeaks.py

# # knapsack
# echo "Running knapsack"
# jython knapsack.py

# # abalone test
# echo "Running abalone test"
# jython abalone_test.py

echo "Running adult"
jython ANN_backprop_adult.py RHC 0 &
jython ANN_backprop_adult.py RHC 1 &
jython ANN_backprop_adult.py RHC 2 &
jython ANN_backprop_adult.py SA 0 &
jython ANN_backprop_adult.py SA 1 &
jython ANN_backprop_adult.py SA 2 &
jython ANN_backprop_adult.py GA 0 &
jython ANN_backprop_adult.py GA 1 &
jython ANN_backprop_adult.py GA 2 &
jython ANN_backprop_adult.py BP 0 &
jython ANN_backprop_adult.py BP 1 &
jython ANN_backprop_adult.py BP 2 &


# # traveling salesman
# echo "Running traveling salesman test"
# jython travelingsalesman.py

# New stuff
# jython flipflop.py