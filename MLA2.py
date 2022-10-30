import sys,pandas,sklearn,random,numpy,pickle
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text # requires python3
from sklearn.neural_network import MLPClassifier
numpy.set_printoptions(suppress=True) # don't use scientific notation in confusion matrices

DT = False
NN = False
if len(sys.argv)<4 or sys.argv[1] not in "DT NN".split():
  print("usage: python3 MLA2.py [DT|NN] <summary_data> <columns> [flags]")
  sys.exit(0)
if sys.argv[1]=="DT": DT = True
if sys.argv[1]=="NN": NN = True

FEATURE_COLS = sys.argv[3].split(',')
FEATURE_COLS = [int(x) for x in FEATURE_COLS]

TARGET_COL = None
if "--target" in sys.argv:
  c = sys.argv.index("--target")+1
  w = sys.argv[c].split(',')
  TARGET_COL = int(w[0])
  POS_CLASS = w[1] # "DNG"
  NEG_CLASS = w[2] # "AG"

SELECT_COL = None
if "--select" in sys.argv:
  c = sys.argv.index("--select")+1
  w = sys.argv[c].split(',')
  SELECT_COL = int(w[0])
  SELECT_VAL = w[1]

CV = 1
if "--CV" in sys.argv:
  c = sys.argv.index("--CV")+1
  CV = int(sys.argv[c])

neg_frac = 1.0
if "--neg_frac" in sys.argv:
  c = sys.argv.index("--neg_frac")+1
  neg_frac = float(sys.argv[c])

if "--load-model" in sys.argv:
  i = sys.argv.index("--load-model")
  f = sys.argv[i+1]
  print("** reading in model from file %s **" % f)
  model = pickle.load(open(f,"rb"))

  ### ??? should I set one-pass, CV=1 ???

# see also --write-model, and --classify-genes below...

print("# command: python %s" % (' '.join(sys.argv)))

########################

# read the data file

headers,data = None,[]
skip = 1
for line in open(sys.argv[2]):
  if skip>0: skip -= 1; continue
  w = line.rstrip().split('\t')
  if headers==None: headers = w; continue
  if SELECT_COL!=None and w[SELECT_COL]!=SELECT_VAL: continue # for excluding certain genes
  data.append(w)

# encoding discrete features as integers
# (alternatively, these could be binarized)

CPC_hashtable = {'noncoding': -2, 'noncoding (weak)': -1, 'coding (weak)': 1, 'coding': 2}
# chloroplast-transit peptide, luminal-transit, mitochondrial-transit protein, SP=signal peptide (PMC6769257) (20k/27k in AT are OTHER, 4k are SP)
LOC_hashtable = {'OTHER':0,'cTP':1,'luTP':2,'lTP':2,'mTP':3,'SP':4} 

feats = []
feat_names = [headers[col] for col in FEATURE_COLS]
classes = [w[TARGET_COL] for w in data]
for w in data:
  vals = []
  for col in FEATURE_COLS:
    val,colname = w[col],headers[col]
    if "cpc2" in colname and "label" in colname: val = CPC_hashtable[val.strip()]
    elif "localization" in colname: val = LOC_hashtable[val.strip()]
    else: val = float(val)
    vals.append(val)
  feats.append(vals)

for x in feat_names: print("using feature: %s" % x)

print("randomizing order of examples")
L = list(range(len(classes)))
random.shuffle(L)
classes = [classes[i] for i in L]
feats = [feats[i] for i in L]

X,Y = feats,classes

hashtable = {}
for i,col in enumerate(FEATURE_COLS): hashtable[headers[col]] = [x[i] for x in X]
df = pandas.DataFrame(hashtable)

columns_to_take_log = ["Gene length","CDS length","Distance from TEs","RPKM"]
for colname in columns_to_take_log:
  if colname in df.columns: df[colname] = numpy.log(df[colname])
  print("*** taking log of %s" % colname)

X = df

labels = [POS_CLASS,NEG_CLASS] # for confusion matrices

########################

# split data into training and test sets

def subsample_negs(X,Y,frac):
  indexes = range(len(X)) # X is DF, Y is list
  ipos = list(filter(lambda x: Y[x]==POS_CLASS,indexes))
  ineg = list(filter(lambda x: Y[x]==NEG_CLASS,indexes))
  n = int(len(ipos)*frac)
  if n<1 or n>len(ineg): sys.stderr.write("error: neg_frac out of range"); sys.exit(0)
  subineg = random.sample(ineg,n)
  combined = ipos+subineg
  random.shuffle(combined)
  subX = X.iloc[combined]
  subY = [Y[i] for i in combined]
  return subX,subY

def count_pos_neg(Y):
  npos = Y.count(POS_CLASS)
  nneg = Y.count(NEG_CLASS)
  return npos,nneg

# prepare lists of training and testing examples for each iteration (or fold of CV)...

X_train_list,Y_train_list,X_test_list,Y_test_list = [],[],[],[]

if CV>1:
  kf = StratifiedKFold(n_splits=CV,shuffle=True)
  for train_index, test_index in kf.split(X,Y):
    X_train = X.iloc[train_index] # note: X is a panda DF (not np.array), while Y is a list
    Y_train = [Y[i] for i in train_index]
    X_train,Y_train = subsample_negs(X_train,Y_train,neg_frac)
    (nPosTrain,nNegTrain) = count_pos_neg(Y_train)

    X_test = X.iloc[test_index]
    Y_test = [Y[i] for i in test_index]
    X_test,Y_test = subsample_negs(X_test,Y_test,1.0) # test sets are always balanced
    (nPosTest,nNegTest) = count_pos_neg(Y_test)

    X_train_list.append(X_train)
    Y_train_list.append(Y_train)
    X_test_list.append(X_test)
    Y_test_list.append(Y_test)

    print("training set size=%s, Npos=%s, Nneg=%s" % (len(X_train),nPosTrain,nNegTrain))
    print("testing set size=%s, Npos=%s, Nneg=%s" % (len(X_test),nPosTest,nNegTest))

else: # if it is not CV, use 70/30 split
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.301,stratify=Y) # 70% training and 30% test; stratify: maintain same proportion of + and - examples
  X_train,Y_train = subsample_negs(X_train,Y_train,neg_frac)
  X_test,Y_test = subsample_negs(X_test,Y_test,1.0) # test sets are always balanced
  (nPosTrain,nNegTrain) = count_pos_neg(Y_train)
  (nPosTest,nNegTest) = count_pos_neg(Y_test)
  (nPosGenome,nNegGenome) = count_pos_neg(Y)
  print("training set size=%s, Npos=%s, Nneg=%s" % (len(X_train),nPosTrain,nNegTrain))
  print("testing set size=%s, Npos=%s, Nneg=%s" % (len(X_test),nPosTest,nNegTest))

  #print("whole genome: de_novo=%d, old_genes=%d (without excluded examples)" % (len(posX),len(negX)))
  print("whole genome: de_novo=%d, old_genes=%d" % (nPosGenome,nNegGenome))

  X_train_list = [X_train]
  Y_train_list = [Y_train]
  X_test_list = [X_test]
  Y_test_list = [Y_test]


#################################

class Result:
  def __init__(self): pass

results = []
for i in range(CV):
  print("#########################\niteration %s/%s" % ((i+1),CV))
  X_train = X_train_list[i]
  Y_train = Y_train_list[i]
  X_test = X_test_list[i]
  Y_test = Y_test_list[i]

  res = Result()

  if DT:
    if "--load-model" in sys.argv: clf = model
    else:
      clf = DecisionTreeClassifier()
      clf = clf.fit(X_train,Y_train)
      model = clf
  
    pred = clf.predict(X_test)
    print("dtree acc=%0.4f" % (sklearn.metrics.accuracy_score(Y_test, pred)))
    print("labels: %s" % labels)
    print("rows are actual labels, columns are predicted")
    print(sklearn.metrics.confusion_matrix(Y_test, pred,labels=labels))
  
    SHOWTREE = True
    if SHOWTREE: print(export_text(clf,feature_names=feat_names))
  
    print("Feature Importances...")
    names,importances = X_train.columns,clf.feature_importances_
    for a,b in zip(names,importances): print('\t'.join(["%0.3f" % b,a]))
  
    print()
    print("\n*** DECISION TREE ***")
    print("tree depth=%s, number of leaves=%s" % (clf.get_depth(),clf.get_n_leaves()))
    cm = sklearn.metrics.confusion_matrix(Y_test, pred,labels=labels)
    print("labels: %s" % labels)
    print("rows are actual labels, columns are predicted")
    print(cm)
    acc = sklearn.metrics.accuracy_score(Y_test,pred)
    print("balanced accuracy=%0.4f" % acc)
  
    # apply the DT to the whole genome
    
    print("\napply decision tree to whole genome to classify genes...")
    pred2 = clf.predict(X)
    cm2 = sklearn.metrics.confusion_matrix(classes, pred2,labels=labels)
    #print("(note: the no-synteny genes are not included)")
    print(cm2)
    acc2 = sklearn.metrics.accuracy_score(classes,pred2)
    print("genome accuracy=%0.4f" % acc2)
  
    res.dtree = clf
    res.dtree_bal_acc = acc
    res.dtree_bal_cm = cm
    res.dtree_whole_acc = acc2
    res.dtree_whole_cm = cm2

  #########################

  if NN:
    #activation='relu'
    #hidden_layers=(10,10)
    activation='tanh'
    hidden_layers=(20)
    
    if "--load-model" in sys.argv: nnet = model
    else: 
      nnet = MLPClassifier(hidden_layer_sizes=hidden_layers,activation=activation,solver='lbfgs',max_iter=10000)
      nnet.fit(X_train, Y_train)
      model = nnet
  
    nnpred = nnet.predict(X_test)
    cm3 = sklearn.metrics.confusion_matrix(Y_test,nnpred,labels=labels)
    
    print()
    print("\n*** NEURAL NET ***")
    print("hidden nodes: %s" % (str(hidden_layers)))
    print("activation: %s" % activation)
    print(cm3)
    acc3 = sklearn.metrics.accuracy_score(Y_test,nnpred)
    print("balanced accuracy=%0.4f" % acc3)
  
    # apply the NN to the whole genome
    
    print("\napply neural net to whole genome to classify genes...")
    nnpred_whole = nnet.predict(X)
    cm_whole = sklearn.metrics.confusion_matrix(classes,nnpred_whole,labels=labels)
    print(cm_whole)
    acc_whole = sklearn.metrics.accuracy_score(classes,nnpred_whole)
    print("genome accuracy=%0.4f" % acc_whole)
  
    res.nnet = nnet
    res.nnet_bal_acc = acc3
    res.nnet_bal_cm = cm3
    res.nnet_whole_acc = acc_whole
    res.nnet_whole_cm = cm_whole

  #results.append((clf,cm,acc,cm2,acc2,nnet,cm3,acc3,cm_whole,acc_whole))
  results.append(res)

######################################################

print("\n*** SUMMARY ***")

if DT:
  accs1 = [x.dtree_bal_acc for x in results]
  m1 = numpy.mean(accs1)
  sd1 = numpy.std(accs1)
  ci1 = 1.96*sd1/numpy.sqrt(CV)
  print("mean acc of Dtrees on balanced test set(s): %0.3f, 95CI=[%0.3f,%0.3f]" % (m1,m1-ci1,m1+ci1))
  
  accs2 = [x.dtree_whole_acc for x in results]
  m2 = numpy.mean(accs2)
  sd2 = numpy.std(accs2)
  ci2 = 1.96*sd2/numpy.sqrt(CV)
  print("mean acc of Dtrees on whole genome:         %0.3f, 95CI=[%0.3f,%0.3f]" % (m2,m2-ci2,m2+ci2))

if NN:
  accs3 = [x.nnet_bal_acc for x in results]
  m3 = numpy.mean(accs3)
  sd3 = numpy.std(accs3)
  ci3 = 1.96*sd3/numpy.sqrt(CV)
  print("mean acc of Nnets on balanced test set(s):  %0.3f, 95CI=[%0.3f,%0.3f]" % (m3,m3-ci3,m3+ci3))

  accs4 = [x.nnet_whole_acc for x in results]
  m4 = numpy.mean(accs4)
  sd4 = numpy.std(accs4)
  ci4 = 1.96*sd4/numpy.sqrt(CV)
  print("mean acc of Nnets on whole genome:          %0.3f, 95CI=[%0.3f,%0.3f]" % (m4,m4-ci4,m4+ci4))

# print average confusion matrices and feature importances

if DT:
  DT_confusion_matrices = [x.dtree_bal_cm for x in results]
  avgcm = numpy.round(numpy.mean(DT_confusion_matrices,axis=0),1)
  print("\naverage of confusion matrices for Dtrees on balanced test sets:")
  print("labels: %s" % labels)
  print("rows are actual labels, columns are predicted")
  print(avgcm)
  
  DT_confusion_matrices = [x.dtree_whole_cm for x in results]
  avgcm = numpy.round(numpy.mean(DT_confusion_matrices,axis=0),1)
  print("\naverage of confusion matrices for Dtrees on whole genome:")
  print("labels: %s" % labels)
  print("rows are actual labels, columns are predicted")
  print(avgcm)

if NN:
  NN_confusion_matrices = [x.nnet_bal_cm for x in results]
  avgcm = numpy.round(numpy.mean(NN_confusion_matrices,axis=0),1)
  print("\naverage of confusion matrices for Nnets on balanced test sets:")
  print("labels: %s" % labels)
  print("rows are actual labels, columns are predicted")
  print(avgcm)
  
  NN_confusion_matrices2 = [x.nnet_whole_cm for x in results]
  avgcm = numpy.round(numpy.mean(NN_confusion_matrices2,axis=0),1)
  print("\naverage of confusion matrices for Nnets on whole genome:")
  print("labels: %s" % labels)
  print("rows are actual labels, columns are predicted")
  print(avgcm)


if DT:
  models = [x.dtree for x in results]
  importances = [x.feature_importances_ for x in models]
  avgimp = numpy.mean(importances,axis=0)
  names = X_train.columns
  print("\naverage feature importances over %s iterations (CV):" % CV)
  for a,b in zip(names,avgimp): print('\t'.join(["%0.3f" % b,a]))

#########################

# these tasks run once at the end 
# should only be done in one-shot mode? else write model from last iteration of CV

if "--write-model" in sys.argv:
  fname = sys.argv[sys.argv.index("--write-model")+1]
  print("writing out model to %s" % fname)
  pickle.dump(model,open(fname, "wb"))

if "--classify-genes" in sys.argv:
  fname = sys.argv[sys.argv.index("--classify-genes")+1]
  f = open(fname,"w")
  f.write('\t'.join("predicted_class pred_type".split()+headers)+"\n")
  pred_whole = model.predict(X) # X is features for whole genome
  for i,w in enumerate(data):
    truelab,predlab = Y[i],pred_whole[i]
    if truelab==POS_CLASS and predlab==POS_CLASS: cat = 'TP'
    if truelab==POS_CLASS and predlab!=POS_CLASS: cat = 'FN'
    if truelab!=POS_CLASS and predlab==POS_CLASS: cat = 'FP'
    if truelab!=POS_CLASS and predlab!=POS_CLASS: cat = 'TN'
    v = [predlab,cat]+w
    s = '\t'.join(v)
    f.write("%s\n" % s)
  f.close()

#############################

# show feature importances by Random Forest

print()
print("Feauture Importances by Random Forest:")

forest = RandomForestClassifier(random_state=0)
forest.fit(X,Y)

importances = forest.feature_importances_
std = numpy.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances = pandas.Series(importances, index=feat_names)

print(forest_importances)
print("standard deviations of feature importances")
print(std)
