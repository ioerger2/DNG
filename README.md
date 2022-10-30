# DNG

This repo contains scripts and data for a paper on predicting de novo
genes (DNGs) in plant genomes using Machine Learning Algorithms
(MLAs).

Send comments or questions to Tom Ioerger (ioerger@cs.tamu.edu).

---------------------------

Requirements (Python packages that need to be installed using pip):

* numpy
* pandas
* sklearn

---------------------------

The main python script that runs MLAs is MLA2.py.  It can train either
a decision tree (DT) or neural network (NN), as specified by the first
argument.

The second argument is a datafile (tab-separated format) with features
calculated for all genes in a given genome.  The gene name is in
column 1.  The features to use are specified by the third arg in a
comma-separated list (the first column is 0, the last is -1).  Note
that features based on size or distance in nucleotides, as well as
RPKMs are automatically log-transformed.

The datafile must also contain a column of class labels, typically DNG
for known de novo genes and AG for ancestral genes.  This column is
indicated and labels is indicated by the --target flag, as shown
below.

By default, the script runs in one-shot mode.  It divides the data
randomly into 70% for training and 30% for testing.  Additionally, the
negative examples (AG) and sub-sampled to be equal in number to the
positive examples, for 'balanced' training and testing to compute
'balanced accuracies.  The user may provide the '--CV <folds>' flag to
indicate how many folds (iterations) of cross-valiation are desired.
In all cases, the test set is prevented from overlapping with the
training set.  In the summary, confidence intervals for accuracies
will be printed out for CV runs.

For all runs, the accuracies and confusion matrices are printed out
for applying the model(s) to the whole genome as well.  Keep in mind
that there are usually many more AGs the DNGs (highly imbalanced).
The confusion matrices can be used to compute things like false
positive rates.

The '--select A,B' flag can be used to select rows that have a
particular value B in column index A (or to exclude rows that do
not have this).


Here is an example command that illustrate how to run the MLA2.py script:

> python3 MLA2.py DT summary_data_de_novo-Atha_v10g.txt 13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,33,34,-1 --select 1,include --target 2,DNG,AG --CV 10

This trains a decision tree and tests it on independent test data
using 10-fold cross-validation.  The decision trees in and feature
importances are shown in each iteration.
If a neural net is desired, change the first arg from DT to NN.


An important option in MLA2.py is the ability to read or write models
into files (called pickle files, by convention in python).  A trained
model may be write out by giving the '--write-model' flag on the
command line.  It may then be read in and used subsequently by giving
the '--load-model' flag.  This can facilitate training a model on the
genome for one species and testing it on another.  Here is an example
where a model is trained on data for A. thaliana (in one-shot mode),
saved in a file (temp.DT.Atha.pickle), and then evaluated on B. rapa
(using cross-validation):

> python3 MLA2.py DT summary_data_de_novo-Atha_v10g.txt 13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,33,34,-1 --select 1,include --target 2,DNG,AG --write-model temp.DT.Atha.pickle 

> python3 MLA2.py DT summary_data_de_novo-Brapa-v3.0_v4h.txt 10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,-1 --select 1,include --target 2,DNG,AG --load-model temp.DT.Atha.pickle --CV 10 

