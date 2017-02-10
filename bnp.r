# Kaggle competition "bnp-paribas-cardif-claims-management"

############## Competition description:
# You are provided with an anonymized dataset containing both categorical and numeric variables 
# available when the claims were received by BNP Paribas Cardif. All string type variables are 
# categorical. There are no ordinal variables.
#
# The "target" column in the train set is the variable to predict. It is equal to 1 for claims
# suitable for an accelerated approval.
#
# The task is to predict a probability ("PredictedProb") for each claim in the test set.
##############

if (.Platform$OS.type == 'windows') {
  Sys.setenv(JAVA_HOME = 'D:\\Program Files\\Java\\jdk1.7.0_79')
}
options(java.parameters = '-Xmx16g')

library (readr)
library (entropy)
library (caret)
library (Hmisc)
library (xgboost)
library (ranger)
library (nnls)
library (glmnet)
library (RANN)
library (randomForest)
library (ComputeBackend)
library (extraTrees)
library (h2o)

source('../utils.r')

# Configuration
# ==================================================================================================

config = list()

config$mode = 'cv' # { single, cv, cv.batch }
config$layer = 0
config$proper.validation = F # if true, will hold-out some data from entire training process, and test on it

config$do.preprocess = T
config$do.train      = T
config$do.xgb.cv     = F
config$do.submit     = T

config$do.generate.base.features  = T
config$do.generate.meta.features  = T # These are L[-1] meta features (i.e., will be used by "L0" models)

config$generate.ccv.yenc.cat          = T
config$generate.ccv.yenc.qnum         = T
config$generate.ccv.yenc.cat.cat      = T
config$generate.ccv.yenc.cat.cat.cat  = T
config$generate.ccv.yenc.cat.qnum     = T
config$generate.ccv.yenc.qnum.qnum    = T
config$generate.ccv.knn.features      = T
config$generate.ccv.glmnet.features   = T
config$generate.ccv.xgb.features      = T

config$submt.id = 23
config$submt.column = 10
config$ref.submt.id = 0 # 0 is the team's best submission so far

#
# Preprocessing parameters
#

config$denoise.numericals            = T
config$drop.numericals               = F
config$drop.categoricals             = F
config$drop.big.categoricals         = F
config$drop.most.important.raw       = F
config$always.prune.categoricals     = F
config$categoricals.max.cardinality  = 30
config$categoricals.min.samplesize   = 200
config$max.levels.to.sms.bayes       = 2 #30 # FIXME try tuning this
config$sms.bayes.count               = 200
config$max.nr.numerical.sms.levels   = 30  # FIXME try tuning this
config$add.manual.features           = F
config$drop.correlated.features      = F
config$count.special.values          = F
config$add.na.indicators             = F
config$add.summary.stats.per.row     = F
config$add.low.card.ints.as.factors  = F
config$add.hexv.encoded.factors      = F
config$add.freq.encoded.factors      = T
config$add.freq.encoded.numericals   = F
config$add.freq.encoded.interactions = F
config$add.numerical.interactions    = F
config$ohe.categoricals              = T
config$remove.factor.reps            = T
config$add.dim.red                   = F
config$add.loo.knn.features          = F # warning: this leaks!
config$add.cv.knn.features           = F # maybe leaks a little
config$add.ccv.yenc.categoricals     = T
config$add.ccv.yenc.numericals       = T
config$add.ccv.yenc.interactions     = T
config$add.ccv.knn.features          = T
config$add.ccv.glmnet.features       = T # actually CV for now
config$add.ccv.xgb.features          = T # actually CV for now
config$drop.low.marginal.assoc       = 0
config$drop.entropy.below            = 3
config$drop.importance.below         = 0
config$drop.importance.above         = 1
config$drop.original.dr.features     = T
config$scale.all                     = F
config$weighting.scheme              = 'none' # { none, partition }
config$weighting.param               = 1
config$weighting.var                 = 'count.na'
config$weighting.level               = 1
config$nr.folds                      = 5
config$data.seed                     = 999

if (config$layer == 0) {
  config$model.tag = 'team_xgb27b'
  config$model.select = 'xgb' # { best, blend, glm, nnls, glmnet, xgb, ranger, et, rf, knn, iso, nnet, nb }
  config$rng.seed = 90210 #999

  config$xgb.params = list(
    booster           = 'gbtree',
   #booster           = 'gblinear',
    predleaf          = F,
    objective         = 'binary:logistic',
   #objective         = 'reg:linear',
   #objective         = 'rank:pairwise',
    eval_metric       = 'logloss',
    reweight          = 1,
    nrounds           = 2500,  # This is the maximum nrounds, and I will not stop early. For prediction I'll use each of ntreelimit.v. This saves the time it would take to retrain separately with different values of nrounds.
    eta               = 0.005, # 0.1 for gblinear
    max_depth         = 7, #8,
   #min_child_weight  = 5,     # even 2 hurts validation
   #gamma             = 5,
   #lambda            = 5,     # even 5 hurts validation
   #alpha             = 0.5,   # even 0.1 hurts validation
   #num_parallel_tree = 5,     # helps, but very little (-0.0002 logloss)
    subsample         = 0.9,
    colsample_bytree  = 0.1
  )

  config$ranger.params = list(
    num.trees = 2000,
    mtry = 200,
    reweight.dummies = T
  )
  
  config$et.params = list(
    ntree = 1000,
    mtry = 100,
    nodesize = 20
  )

  config$rf.params = list(
    ntree = 1000,
    mtry = 10,
    nodesize = 50, # only for randomForest
    max.depth = 50, # only for h2o
    binomial_double_trees = F
  )

  config$knn.params = list(
    k = 200,
    eps = 5
  )

  config$glmnet.params = list(
    lambda = exp((-5):(-15)),
    alpha = 1 # i.e., only L1 regularization
  )

  config$nnet.params = list(
    # TODO add the many, many params this thing needs
  )
  
  config$batch.preprocess = F
  config$batch.auto.xgb.nrounds = F # but yeah.. this will introduce some overfitting. Should I do nested CV?
  
  config$batch.model.specs = data.frame(stringsAsFactors = F,
    model.tag       = c('team_xgb25a', 'team_xgb25b'),
    weighting.var   = c('v110'       , 'v110'       ),
    weighting.level = c(1            , 2            )
  )
  
  # config$batch.model.specs = expand.grid(stringsAsFactors = F,
  #   model.tag                    = 'tag',
  #   model.select                 = 'xgb',
  #   rng.seed                     = 123,
  #   drop.big.categoricals        = F,
  #   add.manual.features          = F,
  #   add.hexv.encoded.factors     = F,
  #   count.special.values         = F,
  #   add.na.indicators            = F,
  #   add.summary.stats.per.row    = F,
  #   add.low.card.ints.as.factors = F,
  #   add.ccv.yenc.categoricals    = F,
  #   add.freq.encoded.factors     = F,
  #   drop.low.marginal.assoc      = 0.1, # 0.1 to 0.3 doesn't hurt, 0.5 hurts
  #   weighting.scheme             = 0,
  #   weighting.param              = 0,
  #   weighting.var                 = 'v50',
  #   weighting.level               = '1',
  #   xgb.reweight                 = c(0.2, 0.4, 0.6, 0.8),
  #   xgb.nrounds                  = 700, 
  #   xgb.objective                = 'binary:logistic',
  #   xgb.eta                      = 0.02,
  #   xgb.max_depth                = 10,
  #   xgb.min_child_weight         = 1,
  #   xgb.subsample                = 1,
  #   xgb.colsample_bytree         = 0.4
  # )
  # 
  # config$batch.model.specs$model.tag = paste0('xgb_rw', 1:nrow(config$batch.model.specs))
  # config$batch.model.specs$rng.seed = 1:nrow(config$batch.model.specs)
} else if (config$layer == 1) {
  config$model.tag = 'team_stack_l1'
  config$model.select = 'xgb'
  config$rng.seed = 9999 #1111
  
  # first bach: config$stack.tags = c('team_xgb1', 'team_xgb1a', 'team_xgb1b', 'team_xgb3a', 'team_xgb5a', 'team_xgb5b', 'team_xgb5c', 'team_xgb6', 'team_xgb7', 'team_xgb7a', 'team_xgb7b', 'team_xgb8', 'team_xgb9', 'team_xgb10', 'team_xgb11', 'team_xgb11b', 'team_xgb11c', 'team_xgb12', 'team_xgb13', 'team_et1', 'team_et2', 'team_ranger1', 'team_ranger2', 'team_ranger3', 'team_ranger4', 'team_rf1', 'team_rf2', 'team_rf3', 'team_glmnet1', 'team_glmnet2', 'team_glmnet3')
  # weaker but also possible: 'team_nnet1', 'team_knn1', 
  config$stack.tags = c('team_xgb15', 'team_xgb16', 'team_xgb17', 'team_xgb18a', 'team_xgb18b', 'team_xgb18c', 'team_xgb19a', 'team_xgb19b', 'team_xgb19c', 'team_xgb19d', 'team_xgb20a', 'team_xgb20b', 'team_xgb20c', 'team_xgb20d', 'team_xgb21', 'team_xgb22', 'team_xgb23', 'team_xgb24a', 'team_xgb24b', 'team_xgb25a', 'team_xgb25b', 'team_xgb26a', 'team_xgb27b')

  config$stack.darragh = T
  config$stack.giba = T
  config$stack.sk = T
  config$add.top.l0.to.stack = F # FIXME dunno why this can be expected to help, but to do it right gotta use CCV to guard against leaks

  config$questionable.models = c('v50139_SVC_bagged', 'xgb7', 'sk_xgb16_23', 'xgb9', 'sk_rf1', 'sk_xgb1b', 'sk_glmnet1_3', 'sk_ranger4', 'sk_xgb15_24', 'v0.488721.glmnet.', 'sk_xgb13_22', 'sk_xgb12_23', 'rgf1')
  config$leaking.models = c('sk_xgb12_23', 'sk_xgb13_22', 'sk_xgb15_24', 'sk_xgb16_23')
  config$partition.models = c('sk_xgb18', 'sk_xgb19', 'sk_xgb20', 'sk_xgb25')
  
  config$xgb.params = list(
    objective         = 'binary:logistic',
    eval_metric       = 'logloss',
    reweight          = 1,
    nrounds           = 1500,
    eta               = 0.01,
    max_depth         = 5,
   #min_child_weight  = 1,
   #gamma             = 1,
   #lambda            = 1,
   #alpha             = 1,
    num_parallel_tree = 5,
    subsample         = 0.7,
    colsample_bytree  = 0.2 #0.7
  )

  config$ranger.params = list(
    num.trees = 1000,
    mtry = 10,
    reweight.dummies = T
  )
  
  config$batch.preprocess = F
  config$batch.auto.xgb.nrounds = F
  
  config$batch.model.specs = expand.grid(stringsAsFactors = F,
    model.tag = 'tag',
    model.select = 'xgb',
    rng.seed = 123,
    weighting.scheme = 0,
    weighting.param = 0,
    xgb.nrounds = 5000, 
    xgb.objective = 'binary:logistic',
    xgb.eta = c(0.02, 0.01, 0.005),
    xgb.max_depth = 2:6,
    xgb.min_child_weight = 1,
    xgb.subsample = c(0.5, 0.7, 0.9),
    xgb.colsample_bytree = c(0.5, 0.7, 0.9)
  )

  config$rf.params = list(
    ntree = 1000,
    mtry = -1,
    nodesize = 20,
    max.depth = 50,
    binomial_double_trees = T
  )
  
  config$batch.model.specs$model.tag = paste0('l1_xgb_stack', 1:nrow(config$batch.model.specs))
} else if (config$layer == 2) {
  config$model.tag = 'team_stack_l2'
  config$model.select = 'best'
  config$rng.seed = 2222
  config$stack.tags = 'team_stack_l1'
  config$stack.darragh = F
  config$stack.giba = F
  config$stack.sk = F
  
  config$xgb.params = list(
    objective         = 'binary:logistic',
    eval_metric       = 'logloss',
    reweight          = 1,
    nrounds           = 800,
    eta               = 0.01,
    max_depth         = 10,
    #min_child_weight  = 1,
    #gamma             = 1,
    #lambda            = 1,
    #alpha             = 1,
    num_parallel_tree = 5,
    subsample         = 0.7,
    colsample_bytree  = 0.7
  )
} else {
  stop('wtf')
}

# Misc training parameters
config$debug.small = F
config$holdout.validation = F
config$measure.importance = F
config$xgb.early.stop.round = 300
config$xgb.params$ntreelimits = seq(100, config$xgb.params$nrounds, by = 100) * ifelse(is.null(config$xgb.params$num_parallel_tree), 1, config$xgb.params$num_parallel_tree)

#
# Compute platform stuff
#

config$compute.backend = 'serial' # {serial, multicore, condor, pbs}
config$nr.cores = ifelse(config$compute.backend %in% c('condor', 'pbs'), 200, 3)
if (.Platform$OS.type == 'windows') {
  config$nr.threads = 8 #7
} else {
  config$nr.threads = detectCores(all.tests = F, logical = F) # for computation on this machine
}

config$package.dependencies = c('ComputeBackend', 'readr', 'xgboost', 'entropy', 'caret', 'Hmisc', 'ranger', 'nnls', 'glmnet')
config$source.dependencies  = NULL
config$cluster.dependencies = NULL
config$cluster.requirements = 'FreeMemoryMB >= 4500'

config$data.dir = '../../Data/Kaggle/bnp-paribas-cardif-claims-management'
if (.Platform$OS.type == 'windows') {
  config$project.dir = getwd()
} else {
  config$project.dir = system('pwd', intern = T)
}
if (config$compute.backend == 'condor') {
  config$tmp.dir = paste0(config$project.dir, '/tmp') # need this on the shared FS
} else if (.Platform$OS.type == 'windows') {
  config$tmp.dir = 'c:/TEMP/kaggle-bnp'
} else {
  config$tmp.dir = '/tmp' # save costs on Azure
}

config$dataset.filename   = paste0(config$tmp.dir, '/pp-data-L', config$layer, '.RData')
config$ancillary.filename = paste0(config$tmp.dir, '/pp-data-ancillary-L', config$layer, '.RData')
config$xgb.trainset.filename  = paste0(config$tmp.dir, '/xgb-trainset-L', config$layer, '.data')
config$xgb.testset.filename   = paste0(config$tmp.dir, '/xgb-testset-L', config$layer, '.data')
config$xgb.validset.filename  = paste0(config$tmp.dir, '/xgb-validset-L', config$layer, '.data')

if (config$debug.small) {
  cat('NOTE: running in small debugging mode\n')
  config$xgb.params$nrounds = 100
}

# ==================================================================================================
# Preprocessing

# This is called manually, once
config$generate.proper.validation = function(config) {
  cat(date(), 'Loading raw data\n')
  train = read_csv(paste0(config$data.dir, '/train.csv'), progress = F)
  cat(date(), 'Generating validation idx list\n')
  set.seed(config$data.seed)
  valid.idxs = data.frame(idx = sample(nrow(train), nrow(train) / 3))
  write_csv(valid.idxs, 'Team/valid-index.csv', col_names = F)
}

# This is called manually, rarely
config$generate.base.features1 = function(config) {
  cat(date(), 'Loading raw data\n')
  train = read_csv(paste0(config$data.dir, '/train.csv'), progress = F)
  test  = read_csv(paste0(config$data.dir, '/test.csv'), progress = F)
  
  if (config$proper.validation) {
    cat(date(), 'NOTE: Data will be generated with proper holdout validation\n')
    valid.idxs = read_csv('Team/valid-index.csv', col_names = F, progress = F)[, 1]
    valid = train[valid.idxs, ]
    train = train[-valid.idxs, ]
    valid.labels = valid$target
    test = rbind(valid[, -which(names(valid) == 'target')], test)
  } else {
    valid.labels = NULL
  }
  
  train.labels = train$target
  train.ids = train$ID
  test.ids = test$ID
  dat = rbind(train[-which(names(train) == 'target')], test)

  set.seed(config$data.seed)
  
  # Due to read_csv: change char to factor
  for (f in names(dat)[unlist(lapply(dat, is.character))]) {
    dat[[f]] = as.factor(dat[[f]])
  }
  
  # This is a little weird... but it doesn't seem too important anyway
  if (config$denoise.numericals) {
    cat(date(), 'Denoising numerical features\n')
    # Maybe we can do better than this, but it seems negligible anyway
    num.features = unlist(lapply(dat, class)) == 'numeric'
    
    denoise = function(x) {
      idx.not.na = !is.na(x)
      order.x = order(x[idx.not.na])
      dx = diff(sort(x[idx.not.na]))
      xc = c(0, cumsum(round(dx, 5))) # this will be an almost perfect monotonic transform of the true clean x, which is enough for tree based methods
      xc[order.x] = xc
      x.clean = x
      x.clean[idx.not.na] = xc
      
      #plot(sort(x)[10000 + (1:1000)])
      #lines(sort(x.clean)[10000 + (1:1000)], col = 2)
      
      #length(unique(x))
      #length(unique(x.clean))
      
      return (x.clean)
    }
    
    dat[, num.features] = lapply(dat[, num.features], denoise)
  }
  
  # Doesn't seem to leak
  dat$ID = NULL
  
  if (config$drop.numericals) {
    cat(date(), 'Dropping numerical features\n')
    dat = dat[, unlist(lapply(dat, class)) == 'factor']
  } else if (config$drop.categoricals) {
    cat(date(), 'Dropping categorical features\n')
    dat = dat[, unlist(lapply(dat, class)) != 'factor']
  }
  
  if (config$drop.big.categoricals) {
    # Do I know how to make these useful? or only overfit to it?
    cat(date(), 'Dropping high cardinality categoricals\n')
    dat$v22 = NULL
    dat$v56 = NULL
  }
  
  if (config$drop.most.important.raw) {
    cat(date(), 'Dropping most important raw features (for diversity)\n')
    dat$v50 = NULL
    # TODO: more?
    #dat$v66 = NULL
    #dat$v31 = NULL
    #dat$v10 = NULL
    #dat$v12 = NULL
  }
  
  if (config$add.manual.features) {
    # These are a little funny, I wonder if I'm recreating information the creators wanted to hide
    # but is useful, or useless raw features they thought are irrelevant...
    cat(date(), 'Using hand engineered features\n')
    
    dat$man.v10.cat = cut(dat$v10, c(0, 0.6, 0.9, 1.2, 1.4, 1.75, 1.9, 2.2, 2.5, 2.75, 3.5, 4.5, 6, 30), labels = 1:13)
    dat$man.v12.lin.v10 = dat$v12 - 0.6036 * dat$v10
    dat$man.v43.m.v116 = dat$v43 - dat$v116
    dat$man.v114.qp.v40 = dat$v114 ^ 2 + 21 * dat$v40 # probably same info as v50
    dat$man.v40.lin.v34 = dat$v40 + 1.6 * dat$v34 # same
    dat$man.v1.lin.v37 = dat$v1 - 1.8 * dat$v37
    dat$man.v97.m.v118 = dat$v97 - dat$v118
    dat$man.v92.m.v95 = dat$v92 - dat$v95
    
    tmp.theta = 0.3454
    tmp.rotm = matrix(c(cos(tmp.theta), sin(tmp.theta), -sin(tmp.theta), cos(tmp.theta)), 2, 2)
    dd = data.matrix(dat[, c('v12', 'v50')]) %*% tmp.rotm
    dat$man.v12.lin.v50.1 = dd[, 1]
    dat$man.v12.lin.v50.2 = dd[, 2]
    
    dd = as.data.frame(randomForest::na.roughfix(dat[, names(dat)[!unlist(lapply(dat, is.factor))][-1]]))
    dat$man.v50x12x14x114 = apply(dd[, c('v50', 'v12', 'v14', 'v114')], 1, prod)
    dat$man.v14x114x10 = apply(dd[, c('v14', 'v114', 'v10')], 1, prod)
    dat$man.v14x11x4 = apply(dd[, c('v14', 'v11', 'v4')], 1, prod)
    dat$man.v12x11x7 = apply(dd[, c('v12', 'v11', 'v7')], 1, prod)
    dat$man.v40x114x34 = apply(dd[, c('v40', 'v114', 'v34')], 1, prod)
    dat$man.v10x14x21x114x20 = apply(dd[, c('v10', 'v14', 'v21', 'v114', 'v20')], 1, prod)
    dat$man.v34x72x21 = apply(dd[, c('v34', 'v72', 'v21')], 1, prod)
    dat$man.v10x12 = dd$v10 * dd$v12 * dd$v72 * dd$v6 / ifelse(dd$v40 == 0, NA, dd$v40)
    rm(dd)
  }
  
  if (config$drop.correlated.features) {
    cat(date(), 'Dropping correlated features\n')
    # This was suggested on the forum:
    removals = c('v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128')
    # These are my guesses:
    #removals = c('v58', 'v75', 'v110', 'v107', 'v106', 'v64', 'v17', 'v76', 'v26', 'v43')
    dat = dat[, -which(names(dat) %in% removals)]
  }
  
  if (0) {
    # Drop features with too much missingness
    cat('Dropping MIA features\n')
    dat = dat[, -which(unlist(lapply(dat, function(x) mean(is.na(x)) > 0.2)))]
  }
  
  # Make a list of the basal features and their properties
  orig.features = data.frame(
    class     = unlist(lapply(dat, class)),
    n.unique  = unlist(lapply(dat, function(x) length(unique(x)))),
    f.missing = unlist(lapply(dat, function(x) mean(is.na(x))))
  )
  orig.feature.names = rownames(orig.features)
  
  #
  # Recode categoricals assuming the levels are meaningful somehow (i.e., that there is fuzzy
  # ordinality at least)
  #
  
  if (config$add.hexv.encoded.factors) { # Has a very limited effect if at all
    cat(date(), 'Adding hexv encoded factors\n')
    cat.features = orig.feature.names[orig.features$class == 'factor']
    for (f in cat.features) {
      lvls = as.character(unique(dat[[f]]))
      lvls = lvls[order(nchar(lvls), tolower(lvls))]
      dat[[paste0('hexv.', f)]] = as.integer(factor(dat[[f]], levels = lvls))
    }
  } else if ('v22' %in% names(dat)) {
    # Just do this for v22, as it is needed by important derived features
    cat(date(), 'Adding hexv for v22\n')
    f = 'v22'
    lvls = as.character(unique(dat[[f]]))
    lvls = lvls[order(nchar(lvls), tolower(lvls))]
    dat[[paste0('hexv.', f)]] = as.integer(factor(dat[[f]], levels = lvls))
  }
  
  #
  # Deal with high cardinality categoricals
  #
  
  if (config$always.prune.categoricals) {
    # Some of the features here have way too many categories. I don't have a magic solution, since
    # the categories themselves are anonymized here, so it might be a good idea to only keep common
    # levels.

    nasty.features = orig.feature.names[orig.features$class == 'factor' & orig.features$n.unique > config$categoricals.max.cardinality]
    cat(date(), 'Reducing categoricals with too many levels\n')
    print(nasty.features)
    
    for (f in nasty.features) {
      tbl = head(sort(table(dat[[f]]), decreasing = T), config$categoricals.max.cardinality)
      levels.to.keep = names(tbl[tbl > config$categoricals.min.samplesize])
      dat[!(dat[[f]] %in% levels.to.keep), f] = NA
      dat[[f]] = factor(dat[[f]])
    }
  }
  
  #
  # Deal with NAs
  #
  
  if (config$count.special.values) { # Does not seem to help
    # Count them per example (in the original features only)
    cat(date(), 'Adding count of special values\n')
    dat$count.na   = rowSums(is.na(dat[, orig.feature.names]))
    dat$count.zero = rowSums(round(dat[, orig.feature.names[orig.features$class %in% c('integer', 'numeric')]], 5) == 0, na.rm = T)
    dat$count.neg  = rowSums(dat[, orig.feature.names[orig.features$class %in% c('integer', 'numeric')]] < 1e-5, na.rm = T)
    dat$count.max  = rowSums(dat[, orig.feature.names[orig.features$class %in% c('integer', 'numeric')]] >= 20 - 1e-5, na.rm = T)
  }
  
  if (config$add.na.indicators) { # Does not seem to help
    cat(date(), 'Adding NA indicators and imputing rare NAs\n')
    
    # For each variable with too much missingness: create a new feature for missigness indicator
    mia.features = orig.feature.names[orig.features$f.missing > 0.1]
    for (f in mia.features) {
      dat[[paste0('na.', f)]] = as.integer(is.na(dat[[f]]))
    }
    
    # For the others that have NAs and are numeric, median-impute
    na.features = orig.feature.names[orig.features$class != 'factor' & orig.features$f.missing > 0 & orig.features$f.missing < 0.1]
    for (f in na.features) {
      dat[[paste0('imp.', f)]] = as.numeric(impute(dat[[f]], median))
    }
  }
  
  #
  # Add dimensionality reduction / clustering features
  #
  
  if (config$add.dim.red) {
    dat.dr = dat[, orig.feature.names[orig.features$class != 'factor']]
    dat.dr = scale(randomForest::na.roughfix(dat.dr))

    # Add k-means cluster labels
    cat(date(), 'Adding kmeans features\n')
    kmns = kmeans(dat.dr, 16)
    cnts = kmns$centers
    kmeans.centroid.distances = NULL
    for (i in 1:nrow(cnts)) {
      kmeans.centroid.distances = cbind(kmeans.centroid.distances, sqrt(colSums((t(dat.dr) - unlist(cnts[i, ])) ^ 2)))
    }
    dat = cbind(dat, dr.kmeans.cluster = as.factor(kmns$cluster), dr.kmeans.dist = kmeans.centroid.distances)
    rm(kmns, cnts, kmeans.centroid.distances)
    
    # Add PCs from all numercial features, imputed perhaps beyond repair...
    cat(date(), 'Adding PCs\n')
    prcomp.res = prcomp(dat.dr, center = T, scale. = T)
    if (1) {
      # EXPERIMENT: very mild use of the target, to pick the top few PCs according to marginal
      # Spearman cor (I don't think this will leak a lot, but could move this to meta-features)
      pc.cor = unlist(apply(prcomp.res$x[1:nrow(train), ], 2, cor, y = train.labels, method = 'spearman'))
      select.pcs = unique(c(1:5, head(order(abs(pc.cor)), 5)))
    } else {
      select.pcs = 1:5
    }
    dat = cbind(dat, dr.pca = prcomp.res$x[, select.pcs])
    rm(prcomp.res)

    cat(date(), 'Adding TSNE 2d mapping\n')
    load('tsne.RData') # => tsne.res (generated in explore.r, it takes a long time)
    dat = cbind(dat, dr.tsne = tsne.res$Y)
    rm(tsne.res)
    
    cat(date(), 'Adding autoencoder 2d mapping\n')
    load(file = 'autoencoder-output.RData') # => deep.fea (generated in explore.r, it takes a long time)
    dat = cbind(dat, dr.autoenc = deep.fea)
    rm(deep.fea)
  }
  
  if (config$add.summary.stats.per.row) { # Doesn't seem to have an effect
    cat(date(), 'Adding summary stats per row\n')
    
    # Simple summary statistics of numericals per row
    numdat = data.matrix(dat[, orig.feature.names[orig.features$class == 'numeric']])
    dat$summary.maxnum = apply(numdat, 1, max   , na.rm = T)
    dat$summary.minnum = apply(numdat, 1, min   , na.rm = T)
    dat$summary.mednum = apply(numdat, 1, median, na.rm = T)
    dat$summary.stdnum = apply(numdat, 1, sd    , na.rm = T)
    
    # This is actually very difficult with so many NAs
    idx = (rowSums(is.na(numdat)) < 5)
    dat$summary.maxnum[idx | !is.finite(dat$summary.maxnum)] = NA
    dat$summary.minnum[idx | !is.finite(dat$summary.minnum)] = NA
    dat$summary.mednum[idx] = NA
    dat$summary.stdnum[idx] = NA
    
    # Count quantile-binned numericals per row
    qtls = t(as.data.frame(apply(numdat, 2, quantile, na.rm = T, probs = c(0.1, 0.25, 0.5, 0.75, 0.9))))
    numdat = t(numdat)
    dat$qcnt.numerics1 = colSums(numdat <= qtls[, 1], na.rm = T)
    dat$qcnt.numerics2 = colSums(numdat > qtls[, 1] & numdat <= qtls[, 2], na.rm = T)
    dat$qcnt.numerics3 = colSums(numdat > qtls[, 2] & numdat <= qtls[, 3], na.rm = T)
    dat$qcnt.numerics4 = colSums(numdat > qtls[, 3] & numdat <= qtls[, 4], na.rm = T)
    dat$qcnt.numerics5 = colSums(numdat > qtls[, 4] & numdat <= qtls[, 5], na.rm = T)
    dat$qcnt.numerics6 = colSums(numdat > qtls[, 5], na.rm = T)
    
    rm(numdat)
  }
  
  if (config$add.low.card.ints.as.factors) {
    # Treat low cardinality continuous variables (i.e., ordinals?) as categorical
    cat(date(), 'Adding categoricals for low cardinality ints\n')
    ordinal.names = orig.feature.names[orig.features$class != 'factor' & orig.features$n.unique <= config$categoricals.max.cardinality]
    for (f in ordinal.names) {
      dat[[paste0('cat.', f)]] = as.factor(dat[[f]]) # should I remove the originals?
    }
  }
  
  if (config$add.numerical.interactions) {
    # I'll only try pairwise for now
    cat(date(), 'Adding select pairwise interactions of numerical features\n')
    if (0) { # old implementation
      load('pairwise-numeric-interactions.RData') # => num.pairs
      num.pairs.to.include = num.pairs[!is.na(num.pairs$pv.mul) & num.pairs$pv.mul < 1e-50, ] # TODO tune the threshold
      for (i in 1:nrow(num.pairs.to.include)) {
        new.f.name = paste0('ia.', num.pairs.to.include$X1[i], '.mul.', num.pairs.to.include$X2[i])
        dat[[new.f.name]] = dat[[num.pairs.to.include$X1[i]]] * dat[[num.pairs.to.include$X2[i]]]
      }
      num.pairs.to.include = num.pairs[!is.na(num.pairs$pv.div) & num.pairs$pv.div < 1e-50, ] # TODO tune the threshold
      for (i in 1:nrow(num.pairs.to.include)) {
        new.f.name = paste0('ia.', num.pairs.to.include$X1[i], '.div.', num.pairs.to.include$X2[i])
        dat[[new.f.name]] = dat[[num.pairs.to.include$X1[i]]] / dat[[num.pairs.to.include$X2[i]]]
        dat[!is.finite(dat[[new.f.name]]), new.f.name] = NA
      }
      num.pairs.to.include = num.pairs[!is.na(num.pairs$pv.vid) & num.pairs$pv.vid < 1e-50, ] # TODO tune the threshold
      for (i in 1:nrow(num.pairs.to.include)) {
        new.f.name = paste0('ia.', num.pairs.to.include$X2[i], '.div.', num.pairs.to.include$X1[i])
        dat[[new.f.name]] = dat[[num.pairs.to.include$X2[i]]] / dat[[num.pairs.to.include$X1[i]]]
        dat[!is.finite(dat[[new.f.name]]), new.f.name] = NA
      }
    } else { # new implementation
      load('muldiv-interactions.RData') # => nn.pairs.to.include
      nn.pairs.to.include.mul = nn.pairs.to.include[abs(nn.pairs.to.include$mul) - pmax(abs(nn.pairs.to.include$cor1), abs(nn.pairs.to.include$cor2)) > 0.004, ] # NOTE: I tried lower thresholds, it did not help
      nn.pairs.to.include.div = nn.pairs.to.include[abs(nn.pairs.to.include$div) - pmax(abs(nn.pairs.to.include$cor1), abs(nn.pairs.to.include$cor2)) > 0.004, ] # NOTE: I tried lower thresholds, it did not help
      for (i in 1:nrow(nn.pairs.to.include.mul)) {
        #new.f.name = paste0('ia.', nn.pairs.to.include.mul$Var1[i], '.add.', nn.pairs.to.include.mul$Var2[i])
        #dat[[new.f.name]] = dat[[nn.pairs.to.include.mul$Var1[i]]] + dat[[nn.pairs.to.include.mul$Var2[i]]]
        new.f.name = paste0('ia.', nn.pairs.to.include.mul$Var1[i], '.mul.', nn.pairs.to.include.mul$Var2[i])
        dat[[new.f.name]] = dat[[nn.pairs.to.include.mul$Var1[i]]] * dat[[nn.pairs.to.include.mul$Var2[i]]]
      }
      # for (i in 1:nrow(nn.pairs.to.include.div)) {
      #   #new.f.name = paste0('ia.', nn.pairs.to.include.div$Var1[i], '.sub.', nn.pairs.to.include.div$Var2[i])
      #   #dat[[new.f.name]] = dat[[nn.pairs.to.include.div$Var1[i]]] - dat[[nn.pairs.to.include.div$Var2[i]]]
      #   new.f.name = paste0('ia.', nn.pairs.to.include.div$Var1[i], '.div.', nn.pairs.to.include.div$Var2[i])
      #   xdenom = dat[[nn.pairs.to.include.div$Var2[i]]]
      #   xdenom = ifelse(xdenom == 0, NA, xdenom) # Inf creates problems with scaling
      #   dat[[new.f.name]] = dat[[nn.pairs.to.include.div$Var1[i]]] / xdenom
      # }
    }
  }
  
  #
  # Recode categoricals as numeric
  #
  
  if (config$add.freq.encoded.factors) {
    cat(date(), 'Frequency coding categoricals\n')
    for (f in orig.feature.names[orig.features$class == 'factor' & orig.features$n.unique > 3]) {
      dat[[paste0('frq.', f)]] = freq.encode(dat[[f]])
    }
  }
  
  if (config$add.freq.encoded.interactions) {
    # I'll only try pairwise for now
    cat(date(), 'Frequency coding interactions\n')
    load('pairwise-factor-interactions.RData') # => cat.pairs
    cat.pairs.to.include = cat.pairs[!is.na(cat.pairs$pv) & cat.pairs$pv < 1e-2, ]
    for (i in 1:nrow(cat.pairs.to.include)) {
      new.f.name = paste0('frq.', cat.pairs.to.include$X1[i], 'xtab', cat.pairs.to.include$X2[i])
      new.f = interaction(dat[[cat.pairs.to.include$X1[i]]], dat[[cat.pairs.to.include$X2[i]]], drop = T)
      dat[[new.f.name]] = freq.encode(new.f)
    }
  }
  
  if (config$add.freq.encoded.numericals) {
    cat(date(), 'Frequency coding numericals\n')
    for (f in orig.feature.names[orig.features$class == 'numeric']) {
      dat[[paste0('frq.', f)]] = freq.encode(dat[[f]])
    }
  }

  if (config$ohe.categoricals) {
    # Replace the original factors with their one-hot encodings
    cat(date(), 'Dummifying categoricals\n')
    dat.cat = dat[, which(unlist(lapply(dat, is.factor)))]
    names(dat.cat)[names(dat.cat) == 'v3'] = 'v3x' # change some names, otherwise dummyVars gets it wrong
    
    if (!config$always.prune.categoricals) {
      # If we haven't done this already, need to prune variables with too many levels
      nasty.features = names(dat.cat)[unlist(lapply(dat.cat, function(x) nlevels(x))) > config$categoricals.max.cardinality]
      
      for (f in nasty.features) {
        tbl = head(sort(table(dat.cat[[f]]), decreasing = T), config$categoricals.max.cardinality)
        levels.to.keep = names(tbl[tbl > config$categoricals.min.samplesize])
        dat.cat[!(dat[[f]] %in% levels.to.keep), f] = NA
        dat.cat[[f]] = factor(dat.cat[[f]])
      }
    }
    
    frmla = as.formula('~ .')
    dat = cbind(dat, as.data.frame(predict(dummyVars(frmla, data = dat.cat, sep = '_'), newdata = dat.cat)))
    rm(dat.cat)
  }
  
  #
  # KNN features computed with LOO
  # NOTE: this leaks badly; Use the CCV variant instead
  #
  
  if (config$add.loo.knn.features) {
    cat(date(), 'Generating LOO NN features\n')

    #
    # Preprocessing
    #
    
    cat(date(), '   preprocessing\n')
    gc()
    
    nn.features = rownames(orig.features)
    nn.features = nn.features[!(nn.features %in% c('v75', 'v110', 'v107'))]
    dat.nn = dat[, nn.features]
    
    # OHE with a lot of levels (because we don't have hamming distance here; the Python sklearn implementation is superior...)
    pp.max.levels = 200
    pp.min.n.level = 20
    names(dat.nn)[names(dat.nn) == 'v3'] = 'v3x' # change some names, otherwise dummyVars gets it wrong
    nasty.features = names(dat.nn)[unlist(lapply(dat.nn, function(x) nlevels(x))) > pp.max.levels]
    for (f in nasty.features) {
      tbl = head(sort(table(dat.nn[[f]]), decreasing = T), pp.max.levels)
      levels.to.keep = names(tbl[tbl > pp.min.n.level])
      dat.nn[!(dat.nn[[f]] %in% levels.to.keep), f] = NA
      dat.nn[[f]] = factor(dat.nn[[f]])
    }
    
    frmla = as.formula('~ .')
    dat.nn = as.data.frame(predict(dummyVars(frmla, data = dat.nn, sep = '_'), newdata = dat.nn))
    
    # Can't have NAs for knn    
    dat.nn = randomForest::na.roughfix(dat.nn)
    # TODO: scaling?
    y.nn = c(train.labels, rep(NA, nrow(test)))

    generate.knn.bf = function(nn.features, eps.search, k.search, prefix) {
      res = list()
      
      # Distance from the '1' class
      idx1 = c(train.labels == 1, rep(F, nrow(test)))
      nn1.res = nn2(dat.nn[idx1, nn.features], query = dat.nn[, nn.features], k = k.search + 1, eps = eps.search)
      res$nn1.dist.1 = ifelse(idx1, nn1.res$nn.dists[, 2], nn1.res$nn.dists[, 1])
      res$nn1.dist.2 = ifelse(idx1, nn1.res$nn.dists[, 3], nn1.res$nn.dists[, 2])
      res$nn1.dist.3 = ifelse(idx1, nn1.res$nn.dists[, 4], nn1.res$nn.dists[, 3])
      res$nn1.dist.k = ifelse(idx1, nn1.res$nn.dists[, k.search + 1], nn1.res$nn.dists[, k.search])
      res$nn1.dist.m = ifelse(idx1, apply(nn1.res$nn.dists[, -1], 1, median), apply(nn1.res$nn.dists[, -(k.search + 1)], 1, median))
      res$nn1.dist.s = ifelse(idx1, apply(nn1.res$nn.dists[, -1], 1, sd    ), apply(nn1.res$nn.dists[, -(k.search + 1)], 1, sd    ))

      # Distance from the '0' class
      idx0 = c(train.labels == 0, rep(F, nrow(test)))
      nn0.res = nn2(dat.nn[idx0, nn.features], query = dat.nn[, nn.features], k = k.search + 1, eps = eps.search)
      res$nn0.dist.1 = ifelse(idx0, nn0.res$nn.dists[, 2], nn0.res$nn.dists[, 1])
      res$nn0.dist.2 = ifelse(idx0, nn0.res$nn.dists[, 3], nn0.res$nn.dists[, 2])
      res$nn0.dist.3 = ifelse(idx0, nn0.res$nn.dists[, 4], nn0.res$nn.dists[, 3])
      res$nn0.dist.k = ifelse(idx0, nn0.res$nn.dists[, k.search + 1], nn0.res$nn.dists[, k.search])
      res$nn0.dist.m = ifelse(idx0, apply(nn0.res$nn.dists[, -1], 1, median), apply(nn0.res$nn.dists[, -(k.search + 1)], 1, median))
      res$nn0.dist.s = ifelse(idx0, apply(nn0.res$nn.dists[, -1], 1, sd    ), apply(nn0.res$nn.dists[, -(k.search + 1)], 1, sd    ))
      
      # Relative distances
      res$nn01d.dist.1 = res$nn1.dist.1 - res$nn0.dist.1
      res$nn01d.dist.m = res$nn1.dist.m - res$nn0.dist.m
      res$nn1d.dist.2m1 = res$nn1.dist.2 - res$nn1.dist.1
      res$nn0d.dist.2m1 = res$nn0.dist.2 - res$nn0.dist.1
      res$nn01d.dist.ss = 0
      for (i in 1:k.search) {
        res$nn01d.dist.ss = res$nn01d.dist.ss + (ifelse(idx0, nn0.res$nn.dists[, i + 1], nn0.res$nn.dists[, i]) > ifelse(idx1, nn1.res$nn.dists[, i + 1], nn1.res$nn.dists[, i]))
      }
      
      res = as.data.frame(res)
      names(res) = paste0(prefix, names(res))
      
      return (res)
    }
    
    cat(date(), '   computing on categoricals\n')
    dat = cbind(dat, generate.knn.bf(nn.features = grep('_', names(dat.nn)), eps.search = 0, k.search = 50, prefix = 'looknn.cat.'))
    cat(date(), '   computing on numericals\n')
    dat = cbind(dat, generate.knn.bf(nn.features = grep('_', names(dat.nn), invert = T), eps.search = 0, k.search = 50, prefix = 'looknn.num.'))
    rm(dat.nn, y.nn)
  }
  
  #
  # CV folds that accompany the data throughtout training stages
  #
  
  if (0) {
    # My own folds based on config seeds
    set.seed(config$data.seed)
    if (1) { # TODO see which kind of partition correlated better with the pub LB?
      cv.folds = createFolds(as.factor(train.labels), k = config$nr.folds) # stratified
    } else {
      cv.folds = createFolds(train.labels, k = config$nr.folds) # unstratified
    }
    #valid.idxs = rep(NA, nrow(train))
    #valid.idxs[cv.folds$Fold1] = 0
    #valid.idxs[cv.folds$Fold2] = 1
    #valid.idxs[cv.folds$Fold3] = 2
    #valid.idxs[cv.folds$Fold4] = 3
    #valid.idxs[cv.folds$Fold5] = 4
    #write_csv(data.frame(idx = valid.idxs), 'Team/train-wo-valid-cv-index.csv', col_names = F)
  } else {
    # Group folds
    if (config$proper.validation) {
      fold.idxs = read_csv('Team/train-wo-valid-cv-index.csv', col_names = F, progress = F)
      cv.folds = list()
      cv.folds$Fold1 = which(fold.idxs == 0)
      cv.folds$Fold2 = which(fold.idxs == 1)
      cv.folds$Fold3 = which(fold.idxs == 2)
      cv.folds$Fold4 = which(fold.idxs == 3)
      cv.folds$Fold5 = which(fold.idxs == 4)
    } else {
      fold.idxs = read_csv('Team/index.csv', col_names = F, progress = F)
      cv.folds = list()
      cv.folds$Fold1 = which(fold.idxs == 0)
      cv.folds$Fold2 = which(fold.idxs == 1)
      cv.folds$Fold3 = which(fold.idxs == 2)
      cv.folds$Fold4 = which(fold.idxs == 3)
      cv.folds$Fold5 = which(fold.idxs == 4)
    }
  }
  
  #
  # Save with data and ancillary information
  #
  
  ancillary = list()
  ancillary$valid.labels = valid.labels
  ancillary$train.labels = train.labels
  ancillary$train.ids = train.ids
  ancillary$test.ids = test.ids
  ancillary$cv.folds = cv.folds
  ancillary$orig.features = orig.features
  
  save(dat, ancillary, file = paste0(config$tmp.dir, '/base-features1.RData'))
}

config$generate.base.features2 = function(config) {
  load(paste0(config$tmp.dir, '/base-features1.RData')) # => dat, ancillary

  #
  # Drop stuff
  #
  
  if (config$remove.factor.reps) {
    cat(date(), 'Dropping factor representation of categoricals\n')
    dat = dat[, which(!unlist(lapply(dat, is.factor)))]
  }
  
  if (config$drop.low.marginal.assoc > 0) {
    # Drop least marginally-associated features (currently using Spearman correlation)
    # NOTE: this is a little bit problematic. Use with care if at all.
    spear.cor = unlist(lapply(dat[1:nrow(train), ], function(x) { if (class(x) == 'factor') return (1); idx = !is.na(x); cor(x[idx], y = train.labels[idx], method = 'spearman') }))
    spear.cor = sort(abs(spear.cor), decreasing = F)
    low.to.drop = intersect(names(dat), names(spear.cor[1:round(config$drop.low.marginal.assoc * length(spear.cor))]))
    cat(date(), 'Dropping', length(low.to.drop), 'lowest marginally associated numerical features:\n')
    print(low.to.drop)
    dat = dat[, -which(names(dat) %in% low.to.drop)]
  }
  
  if (config$drop.entropy.below > 0) {
    ents = lapply(dat, function(x) { if (length(unique(x)) > 30) return (Inf); entropy.empirical(as.numeric(x[!is.na(x)])) })
    ents[is.na(ents)] = Inf
    if (any(ents <= config$drop.entropy.below)) {
      cat(date(), 'Dropping these features for low entropy:\n')
      print(names(ents)[ents <= config$drop.entropy.below])
    }
    dat = dat[, ents > config$drop.entropy.below]
  }
  
  #
  # Summary report/sanity check  
  #
  
  if (0) {
    features = data.frame(
      class     = unlist(lapply(dat, class)),
      n.unique  = unlist(lapply(dat, function(x) length(unique(x)))),
      f.missing = unlist(lapply(dat, function(x) mean(is.na(x)))),
      f.inf     = unlist(lapply(dat, function(x) mean(!is.finite(x))))
    )
    View(features)
    browser()
  }

  #
  # Split the datasets and save
  #
  
  train.bf = dat[1:length(ancillary$train.labels), ]
  test.bf  = dat[(length(ancillary$train.labels) + 1):nrow(dat), ]
  
  save(ancillary, file = config$ancillary.filename)
  
  cat(date(), 'Saving data\n')
  save(train.bf, test.bf, file = paste0(config$tmp.dir, '/base-features2.RData'))
}

#
# Generate meta features
#
# These are features that rely in some way on the response, and can be thought of as an initial 
# level of stacking. Following Gilberto's suggestion, I use nested CV. More specifically, for the
# purpose of generating the trainset meta features at this level, and when intending to use them for
# stacking using a given partition, these features are built more carefully than the usual k-fold
# stacking. Specifically, we generate separate full training sets for each stacking fold in the next
# layer. Let's call this next layer fold the "target fold". When we build the trainset for target
# fold k, we do CV like standard stacking, but also always exclude the target fold. This is supposed
# to avoid overfitting/leakage that crazy L0 models tend to have. (It might be an overkill, I'm not
# sure yet). Note though, that this doens't solve the problems with leaks through CV. In some cases
# it improves the situation enough for this not to be a problem, but in other cases it's still 
# prohibitive.
#

config$generate.meta.features = function(config) {
  # Load base features  
  load(paste0(config$tmp.dir, '/base-features1.RData')) # => dat, ancillary

  if (0) {
    # Reduce data size to debug code here
    cat('NOTE::::::::::::: DBUG MODE\n')
    dat = dat[1:2000, ]
    ancillary$train.labels = ancillary$train.labels[1:1000]
    ancillary$cv.folds = createFolds(ancillary$train.labels, k = config$nr.folds)
  }
  
  ndat = nrow(dat)
  ntrain = length(ancillary$train.labels)
  ntest = nrow(dat) - ntrain # NOTE: validset if exists is treated as part of the testset in this function
  train.idx = 1:ntrain
  test.idx  = ntrain + (1:ntest)
  
  dat.labels = c(ancillary$train.labels, rep(NA, ntest))
  orig.feature.names = rownames(ancillary$orig.features)
  
  if (0) {
    # EXPERIMENT: use 10 folds to generate final trainset
    set.seed(config$data.seed)
    train.cv.folds = createFolds(as.factor(ancillary$train.labels), k = 10)
  } else {
    train.cv.folds = ancillary$cv.folds
  }
  
  # This will take dat.to.enc, a set of columns we want to yencode, with new colnames already, and
  # apply yenc on them in parallel, then cbind them to test.mf, train.mf, and cvalid.mf. Finally, we
  # save to disk to avoid running out of memory.
  generate.yenc = function(dat.to.enc, group.name, nr.threads = config$nr.threads) {
    # Wrapper that takes everything as arguments
    yenc = function(x, train.idx, test.idx, k, cv.folds, y, max.nr.numerical.sms.levels, max.levels.to.sms.bayes, sms.bayes.count) {
      if (k == -1) {
        yenc.automatic(x[train.idx], y, cv.folds, k, x[test.idx], max.nr.numerical.sms.levels, max.levels.to.sms.bayes, sms.bayes.count)
      } else {
        yenc.automatic(x[train.idx], y, cv.folds, k, NULL, max.nr.numerical.sms.levels, max.levels.to.sms.bayes, sms.bayes.count)
      }
    }
    
    # for predicting on the testset (as testset)
    cat(date(), '   test...\n')
    gc(verbose = F)
    test.mf = as.data.frame(mclapply(dat.to.enc, yenc, train.idx = train.idx, test.idx = test.idx, k = -1, cv.folds = ancillary$cv.folds, y = ancillary$train.labels, max.nr.numerical.sms.levels = config$max.nr.numerical.sms.levels, max.levels.to.sms.bayes = config$max.levels.to.sms.bayes, sms.bayes.count = config$sms.bayes.count, mc.preschedule = T, mc.cores = nr.threads))
    
    # for predicting on the testset (as trainset)
    cat(date(), '   train...\n')
    gc(verbose = F)
    train.mf = as.data.frame(mclapply(dat.to.enc, yenc, train.idx = train.idx, test.idx = 'NULL', k = 0, cv.folds = ancillary$cv.folds, y = ancillary$train.labels, max.nr.numerical.sms.levels = config$max.nr.numerical.sms.levels, max.levels.to.sms.bayes = config$max.levels.to.sms.bayes, sms.bayes.count = config$sms.bayes.count, mc.preschedule = T, mc.cores = nr.threads))

    # for predicting on validation folds (as both trainset and testset)
    cvalid.mf = list()                                  
    for (i in 1:config$nr.folds) {
      cat(date(), '   valid', i, '...\n')
      gc(verbose = F)
      cvalid.mf[[i]] = as.data.frame(mclapply(dat.to.enc, yenc, train.idx = train.idx, test.idx = 'NULL', k = i, cv.folds = ancillary$cv.folds, y = ancillary$train.labels, max.nr.numerical.sms.levels = config$max.nr.numerical.sms.levels, max.levels.to.sms.bayes = config$max.levels.to.sms.bayes, sms.bayes.count = config$sms.bayes.count, mc.preschedule = T, mc.cores = nr.threads))
    }
    
    save(test.mf, train.mf, cvalid.mf, file = paste0(config$tmp.dir, '/meta-features-', group.name, '.RData'))
    
    return (NULL)
  }
  
  # =========
  #browser()
  if (0) {
    # DEBUG yenc
    f = 'v50'
    x = train[[f]]
    y = ancillary$train.labels
    cv.folds = ancillary$cv.folds
    xnew = test[[f]]
    k = -1
    m = config$sms.bayes.count

    z0 = yenc.automatic(x, y, cv.folds, k, xnew, config$max.nr.numerical.sms.levels, config$max.levels.to.sms.bayes, config$sms.bayes.count)
    
    z1 = yenc.bayes(x, y, cv.folds, k, m, xnew)
    z2 = yenc.random.intercept(x, y, cv.folds, k, xnew)
    z3 = yenc.glmm(x, y, cv.folds, k, xnew)
    z4a = smm.generic(x, y, smm.bayes.core, cv.folds, k, xnew, y0 = 0)
    z4b = smm.generic(x, y, smm.bayes.core, cv.folds, k, xnew, y0 = 1)
    table(z0, z1, exclude = NULL)
    table(z1, z2, exclude = NULL)
    table(z2, z3, exclude = NULL)
    table(z1, z4a, exclude = NULL) # these really aren't supposed to be similar
    mean(abs(z1 - z2))
    mean(abs(z1 - z3))
    table(z4a, z4b, exclude = NULL)
    
    f = 'v12'
    x = train[[f]]
    y = ancillary$train.labels
    cv.folds = ancillary$cv.folds
    xnew = test[[f]]
    k = -1
    z0 = smm.generic(x, y, smm.bayes.core, cv.folds, k, xnew, y0 = 0)
    z1 = smm.generic(x, y, smm.bayes.core, cv.folds, k, xnew, y0 = 1)
  }
  # =========

  if (config$generate.ccv.yenc.cat) {
    cat(date(), 'Generating CCV yenc categoricals\n')
    cat.features = names(dat)[unlist(lapply(dat, function(x) is.factor(x) && nlevels(x) > 2))]
    dat.to.enc = dat[, cat.features]
    names(dat.to.enc) = paste0('yenc.', names(dat.to.enc))
    generate.yenc(dat.to.enc, 'yenc-cat', 6)
  }
  
  if (config$generate.ccv.yenc.qnum) {
    cat(date(), 'Generating CCV yenc quantized numericals\n')
    load(file = 'quantize-numericals.RData') # => qnums
    qnums.to.include = qnums[complete.cases(qnums), ]
    qnums.to.include = qnums.to.include[abs(qnums.to.include$corz) - abs(qnums.to.include$corx) > 0.005, ]
    all.num.features = names(dat)[!unlist(lapply(dat, is.factor))]
    #num.features = c(orig.feature.names[ancillary$orig.features$class != 'factor'], all.num.features[c(grep('^hexv.', all.num.features), grep('^dr.', all.num.features))])
    num.features = c(qnums.to.include$X, all.num.features[c(grep('^hexv.', all.num.features), grep('^dr.pca', all.num.features))])
    dat.to.enc = dat[, num.features]
    dat.to.enc = as.data.frame(lapply(dat.to.enc, function(x) cut2(x, g = min(config$max.nr.numerical.sms.levels, length(unique(x))))))
    names(dat.to.enc) = paste0('yenc.', names(dat.to.enc))
    generate.yenc(dat.to.enc, 'yenc-qnum', 8)
  }
  
  if (config$generate.ccv.yenc.cat.cat) {
    cat(date(), 'Generating CCV yenc categorical interactions (2-way)\n')
    load('factor-factor-interactions.RData') # => cc.pairs
    cc.pairs.to.include = cc.pairs[complete.cases(cc.pairs), ]
    cc.pairs.to.include = cc.pairs.to.include[cc.pairs.to.include$cor12 - pmax(cc.pairs.to.include$cor1, cc.pairs.to.include$cor2) > 0.005, ]
    dat.to.enc = as.data.frame(matrix(NA, nrow(dat), nrow(cc.pairs.to.include)))
    for (i in 1:nrow(cc.pairs.to.include)) {
      names(dat.to.enc)[i] = paste0('yenc.', cc.pairs.to.include$Var1[i], '.xtab.', cc.pairs.to.include$Var2[i])
      dat.to.enc[, i] = interaction(dat[[cc.pairs.to.include$Var1[i]]], dat[[cc.pairs.to.include$Var2[i]]], drop = T)
    }
    generate.yenc(dat.to.enc, 'yenc-cat-cat')
  }
  
  if (config$generate.ccv.yenc.cat.cat.cat) {
    cat(date(), 'Generating CCV yenc caegorical interactions (3-way)\n')
    load('factor-3way-interactions.RData') # => cc.triplets
    cc.triplets.to.include = cc.triplets[complete.cases(cc.triplets), ]
    cc.triplets.to.include = cc.triplets.to.include[cc.triplets.to.include$cor123 - pmax(cc.triplets.to.include$cor1, cc.triplets.to.include$cor2, cc.triplets.to.include$cor3) > 0.01, ]
    cc.triplets.to.include = cc.triplets.to.include[order(cc.triplets.to.include$cor123, decreasing = T), ]
    chunk.sz = 64
    nr.chunks = ceiling(nrow(cc.triplets.to.include) / chunk.sz)
    for (ci in 1:nr.chunks) {
      cat(date(), 'chunk', ci, 'out of', nr.chunks, '\n')
      idx.this.chunk = (ci - 1) * chunk.sz + (1:chunk.sz)
      idx.this.chunk = idx.this.chunk[idx.this.chunk <= nrow(cc.triplets.to.include)]
      dat.to.enc = as.data.frame(matrix(NA, nrow(dat), length(idx.this.chunk)))
      for (i in 1:length(idx.this.chunk)) {
        ii = idx.this.chunk[i]
        names(dat.to.enc)[i] = paste0('yenc.', cc.triplets.to.include$Var1[ii], '.xtab.', cc.triplets.to.include$Var2[ii], '.xtab.', cc.triplets.to.include$Var3[ii])
        dat.to.enc[, i] = interaction(dat[[cc.triplets.to.include$Var1[ii]]], dat[[cc.triplets.to.include$Var2[ii]]], dat[[cc.triplets.to.include$Var3[ii]]], drop = T)
      }
      generate.yenc(dat.to.enc, paste0('yenc-cat-cat-cat-', ci), 8)
    }
  }
  
  if (config$generate.ccv.yenc.cat.qnum) {
    cat(date(), 'Generating CCV yenc categorical X quantized-numerical interactions\n')
    load('factor-numerical-interactions.RData') # => cn.pairs
    cn.pairs.to.include = cn.pairs[complete.cases(cn.pairs), ]
    cn.pairs.to.include = cn.pairs.to.include[cn.pairs.to.include$cor12 - pmax(cn.pairs.to.include$cor1, cn.pairs.to.include$cor2) > 0.005, ]
    cn.pairs.to.include = cn.pairs.to.include[order(cn.pairs.to.include$cor12, decreasing = T), ]
    chunk.sz = 64
    nr.chunks = ceiling(nrow(cn.pairs.to.include) / chunk.sz)
    for (ci in 1:nr.chunks) {
      cat(date(), 'chunk', ci, 'out of', nr.chunks, '\n')
      idx.this.chunk = (ci - 1) * chunk.sz + (1:chunk.sz)
      idx.this.chunk = idx.this.chunk[idx.this.chunk <= nrow(cn.pairs.to.include)]
      dat.to.enc = as.data.frame(matrix(NA, nrow(dat), length(idx.this.chunk)))
      for (i in 1:length(idx.this.chunk)) {
        ii = idx.this.chunk[i]
        names(dat.to.enc)[i] = paste0('yenc.', cn.pairs.to.include$Var1[ii], '.qxtab.', cn.pairs.to.include$Var2[ii])
        x2.quantized = cut2(dat[[cn.pairs.to.include$Var2[ii]]], g = min(config$max.nr.numerical.sms.levels, length(unique(dat[[cn.pairs.to.include$Var2[ii]]]))))
        dat.to.enc[, i] = interaction(dat[[cn.pairs.to.include$Var1[ii]]], x2.quantized, drop = T)
      }
      generate.yenc(dat.to.enc, paste0('yenc-cat-qnum-', ci), 8)
    }
  }
  
  if (config$generate.ccv.yenc.qnum.qnum) {
    cat(date(), 'Generating CCV yenc quantized-numerical X quantized-numerical interactions\n')
    load('numerical-numerical-interactions.RData') # => qqpairs
    qqpairs.to.include = qqpairs[complete.cases(qqpairs), ]
    qqpairs.to.include = qqpairs.to.include[qqpairs.to.include$cor12 - pmax(qqpairs.to.include$cor1, qqpairs.to.include$cor2) > 0.005, ]
    qqpairs.to.include = qqpairs.to.include[order(qqpairs.to.include$cor12, decreasing = T), ]
    chunk.sz = 64
    nr.chunks = ceiling(nrow(qqpairs.to.include) / chunk.sz)
    for (ci in 1:nr.chunks) {
      cat(date(), 'chunk', ci, 'out of', nr.chunks, '\n')
      idx.this.chunk = (ci - 1) * chunk.sz + (1:chunk.sz)
      idx.this.chunk = idx.this.chunk[idx.this.chunk <= nrow(qqpairs.to.include)]
      dat.to.enc = as.data.frame(matrix(NA, nrow(dat), length(idx.this.chunk)))
      for (i in 1:length(idx.this.chunk)) {
        ii = idx.this.chunk[i]
        names(dat.to.enc)[i] = paste0('yenc.', qqpairs.to.include$Var1[ii], '.qxtab.', qqpairs.to.include$Var2[ii])
        x1.quantized = cut2(dat[[qqpairs.to.include$Var1[ii]]], g = min(config$max.nr.numerical.sms.levels, length(unique(dat[[qqpairs.to.include$Var1[ii]]]))))
        x2.quantized = cut2(dat[[qqpairs.to.include$Var2[ii]]], g = min(config$max.nr.numerical.sms.levels, length(unique(dat[[qqpairs.to.include$Var2[ii]]]))))
        dat.to.enc[, i] = interaction(x1.quantized, x2.quantized, drop = T)
      }
      generate.yenc(dat.to.enc, paste0('yenc-qnum-qnum-', ci), 8)
    }
  }
  
  if (config$generate.ccv.knn.features) {
    cat(date(), 'Generating NN features\n')
    gc()
    
    # FIXME: I don't think there is a point in doing this with CCV (standard CV should be enough, and we can even use LOO)
    
    # See explore.r for experiments that lead me here...
    
    #
    # Preprocessing
    #
    
    cat(date(), '   preprocessing\n')
    
    nn.features = rownames(ancillary$orig.features)
    nn.features = nn.features[!(nn.features %in% c('v75', 'v110', 'v107'))]
    dat.nn = dat[, nn.features]
    
    # Redo the OHE with more levels
    pp.max.levels = 200
    pp.min.n.level = 20
    names(dat.nn)[names(dat.nn) == 'v3'] = 'v3x' # change some names, otherwise dummyVars gets it wrong
    nasty.features = names(dat.nn)[unlist(lapply(dat.nn, function(x) nlevels(x))) > pp.max.levels]
    for (f in nasty.features) {
      tbl = head(sort(table(dat.nn[[f]]), decreasing = T), pp.max.levels)
      levels.to.keep = names(tbl[tbl > pp.min.n.level])
      dat.nn[!(dat.nn[[f]] %in% levels.to.keep), f] = NA
      dat.nn[[f]] = factor(dat.nn[[f]])
    }
    
    frmla = as.formula('~ .')
    dat.nn = as.data.frame(predict(dummyVars(frmla, data = dat.nn, sep = '_'), newdata = dat.nn))
    
    # Can't have NAs for knn    
    dat.nn = randomForest::na.roughfix(dat.nn)
    
    # Split for training
    train.y = ancillary$train.labels
    train.x = dat.nn[1:ntrain          , ]
    test.x  = dat.nn[ntrain + (1:ntest), ]
    rm(dat.nn)
    
    generate.knn.mf = function(x, y, xnew, nn.features, eps.search, k.search, prefix) {
      res = list()
      
      # Distance from the '1' class
      nn1.res = nn2(x[y == 1, nn.features], query = xnew[, nn.features], k = k.search, eps = eps.search)
      res$nn1.dist.1 = nn1.res$nn.dists[, 1]
      res$nn1.dist.2 = nn1.res$nn.dists[, 2]
      res$nn1.dist.3 = nn1.res$nn.dists[, 3]
      res$nn1.dist.k = nn1.res$nn.dists[, k.search]
      res$nn1.dist.m = apply(nn1.res$nn.dists, 1, median)
      res$nn1.dist.s = apply(nn1.res$nn.dists, 1, sd)
      res$nn1.v50.1 = (x$v50[y == 1])[nn1.res$nn.idx[, 1]]
      res$nn1.v50.m = rowMeans(matrix((x$v50[y == 1])[nn1.res$nn.idx], nrow = nrow(xnew)))
      
      # Distance from the '0' class
      nn0.res = nn2(x[y == 0, nn.features], query = xnew[, nn.features], k = k.search, eps = eps.search)
      res$nn0.dist.1 = nn0.res$nn.dists[, 1]
      res$nn0.dist.2 = nn0.res$nn.dists[, 2]
      res$nn0.dist.3 = nn0.res$nn.dists[, 3]
      res$nn0.dist.k = nn0.res$nn.dists[, k.search]
      res$nn0.dist.m = apply(nn0.res$nn.dists, 1, median)
      res$nn0.dist.s = apply(nn0.res$nn.dists, 1, sd)
      res$nn0.v50.1 = (x$v50[y == 0])[nn0.res$nn.idx[, 1]]
      res$nn0.v50.m = rowMeans(matrix((x$v50[y == 0])[nn0.res$nn.idx], nrow = nrow(xnew)))
      
      # Relative distances
      res$nn01d.dist.1 = res$nn1.dist.1 - res$nn0.dist.1
      res$nn01d.dist.m = res$nn1.dist.m - res$nn0.dist.m
      res$nn1d.dist.2m1 = res$nn1.dist.2 - res$nn1.dist.1
      res$nn0d.dist.2m1 = res$nn0.dist.2 - res$nn0.dist.1
      res$nn01d.dist.ss = rowSums(nn0.res$nn.dists > nn1.res$nn.dists)
      
      # FIXME use distances for weights ( I should really switch to python for this)
      nn.res = nn2(x[, nn.features], query = xnew[, nn.features], k = k.search, eps = eps.search)
      res$ypred = rowMeans(matrix(y[nn.res$nn.idx], nrow = nrow(xnew)))
      
      res = as.data.frame(res)
      names(res) = paste0(prefix, names(res))
      
      return (res)
    }
    
    cat(date(), '   computing on categoricals\n')
    eps.search  = 5 # FIXME it didn't return with 2
    k.search    = 50
    nn.features = grep('_', names(train.x))
    test.mf  = smm.generic3(train.x, train.y, generate.knn.mf, ancillary$cv.folds, -1, test.x, nn.features = nn.features, eps.search = eps.search, k.search = k.search, prefix = 'cat.knn.')
    train.mf = smm.generic3(train.x, train.y, generate.knn.mf, train.cv.folds    ,  0, NULL  , nn.features = nn.features, eps.search = eps.search, k.search = k.search, prefix = 'cat.knn.')
    cvalid.mf = list()
    for (ii in 1:config$nr.folds) {
      cvalid.mf[[ii]] = smm.generic3(train.x, train.y, generate.knn.mf, ancillary$cv.folds, ii, NULL, nn.features = nn.features, eps.search = eps.search, k.search = k.search, prefix = 'cat.knn.')
    }
    save(test.mf, train.mf, cvalid.mf, file = paste0(config$tmp.dir, '/meta-features-knn-cat.RData'))
    
    cat(date(), '   computing on numericals\n')
    eps.search  = 3 # FIXME can I do less?
    k.search    = 50
    nn.features = grep('_', names(train.x), invert = T)
    test.mf  = smm.generic3(train.x, train.y, generate.knn.mf, ancillary$cv.folds, -1, test.x, nn.features = nn.features, eps.search = eps.search, k.search = k.search, prefix = 'num.knn.')
    train.mf = smm.generic3(train.x, train.y, generate.knn.mf, train.cv.folds    ,  0, NULL  , nn.features = nn.features, eps.search = eps.search, k.search = k.search, prefix = 'num.knn.')
    cvalid.mf = list()
    for (ii in 1:config$nr.folds) {
      cvalid.mf[[ii]] = smm.generic3(train.x, train.y, generate.knn.mf, ancillary$cv.folds, ii, NULL, nn.features = nn.features, eps.search = eps.search, k.search = k.search, prefix = 'num.knn.')
    }
    save(test.mf, train.mf, cvalid.mf, file = paste0(config$tmp.dir, '/meta-features-knn-num.RData'))
    
    rm(train.x, train.y, test.x)
  }
  
  if (config$generate.ccv.glmnet.features) {
    cat(date(), 'Adding glmnet features\n')
    gc()

    do.ccv = F
    
    generate.glmnet.mf = function(x, y, xnew, x.idx) {
      mdl = glmnet(data.matrix(x[, x.idx]), y, family = 'binomial')
      preds = predict(mdl, data.matrix(xnew[, x.idx]), s = 1e-5, type = 'response')
    }
    
    test.mf    = data.frame(removeme = rep(NA, ntest )) # for predicting on the testset (as testset)
    train.mf   = data.frame(removeme = rep(NA, ntrain)) # for predicting on the testset (as trainset)
    cvalid.mf = list()                                  # for predicting on validation folds (as both trainset and testset)
    if (do.ccv) {
      for (i in 1:config$nr.folds) {
        cvalid.mf[[i]] = data.frame(removeme = rep(NA, ntrain))
      }
    }
    
    cat(date(), '   computing on OHE categoricals\n')
    
    glmnet.features = grep('_', names(dat))
    dat.glmnet = dat[, glmnet.features] # glmnet will standardize
    dat.glmnet[is.na(dat.glmnet)] = 0
    train.glmnet = dat.glmnet[1:ntrain          , ]
    test.glmnet  = dat.glmnet[ntrain + (1:ntest), ]
    rm(dat.glmnet)

    nf = 'cat.ohe.glmnet'
    test.mf [[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, -1, test.glmnet, x.idx = 1:ncol(train.glmnet))
    train.mf[[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, train.cv.folds    ,  0, NULL       , x.idx = 1:ncol(train.glmnet))
    if (do.ccv) {
      for (ii in 1:config$nr.folds) {
        cvalid.mf[[ii]][[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, ii, NULL, x.idx = cat.x.idx)
      }
    }
    
    cat(date(), '   computing on FE categoricals\n')

    glmnet.features = as.vector(which(unlist(lapply(dat, is.factor))))
    dat.glmnet = randomForest::na.roughfix(apply(dat[, glmnet.features], 2, freq.encode)) # glmnet will standardize
    train.glmnet = dat.glmnet[1:ntrain          , ]
    test.glmnet  = dat.glmnet[ntrain + (1:ntest), ]
    rm(dat.glmnet)
    
    nf = 'cat.freq.glmnet'
    test.mf [[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, -1, test.glmnet, x.idx = 1:ncol(train.glmnet))
    train.mf[[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, train.cv.folds    ,  0, NULL       , x.idx = 1:ncol(train.glmnet))
    if (do.ccv) {
      for (ii in 1:config$nr.folds) {
        cvalid.mf[[ii]][[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, ii, NULL, x.idx = 1:ncol(train.glmnet))
      }
    }

    cat(date(), '   computing on HV categoricals\n')
    
    glmnet.features = as.vector(which(unlist(lapply(dat, is.factor))))
    dat.glmnet = randomForest::na.roughfix(apply(dat[, glmnet.features], 2, hexv.encode)) # glmnet will standardize
    train.glmnet = dat.glmnet[1:ntrain          , ]
    test.glmnet  = dat.glmnet[ntrain + (1:ntest), ]
    rm(dat.glmnet)
    
    nf = 'cat.hexv.glmnet'
    test.mf [[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, -1, test.glmnet, x.idx = 1:ncol(train.glmnet))
    train.mf[[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, train.cv.folds    ,  0, NULL       , x.idx = 1:ncol(train.glmnet))
    if (do.ccv) {
      for (ii in 1:config$nr.folds) {
        cvalid.mf[[ii]][[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, ii, NULL, x.idx = 1:ncol(train.glmnet))
      }
    }
    
    cat(date(), '   computing on raw numericals\n')
    
    glmnet.features = as.vector(which(!unlist(lapply(dat, is.factor))))
    glmnet.features = glmnet.features[grep('^v', names(dat)[glmnet.features])]
    glmnet.features = intersect(glmnet.features, grep('_', names(dat), invert = T))
    dat.glmnet = as.data.frame(randomForest::na.roughfix(dat[, glmnet.features])) # glmnet will standardize
    dat.glmnet = dat.glmnet[, which(!unlist(lapply(dat.glmnet, function(x) any(is.na(x)))))] # if there are any Infs or constants
    train.glmnet = dat.glmnet[1:ntrain          , ]
    test.glmnet  = dat.glmnet[ntrain + (1:ntest), ]
    rm(dat.glmnet)
    
    nf = 'num.raw.glmnet'
    test.mf [[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, -1, test.glmnet, x.idx = 1:ncol(train.glmnet))
    train.mf[[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, train.cv.folds    ,  0, NULL       , x.idx = 1:ncol(train.glmnet))
    if (do.ccv) {
      for (ii in 1:config$nr.folds) {
        cvalid.mf[[ii]][[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, ii, NULL, x.idx = 1:ncol(train.glmnet))
      }
    }
    
    cat(date(), '   computing on rank numericals\n')
    
    glmnet.features = as.vector(which(!unlist(lapply(dat, is.factor))))
    glmnet.features = glmnet.features[grep('^v', names(dat)[glmnet.features])]
    glmnet.features = intersect(glmnet.features, grep('_', names(dat), invert = T))
    dat.glmnet = as.data.frame(randomForest::na.roughfix(apply(dat[, glmnet.features], 2, rank))) # glmnet will standardize
    dat.glmnet = dat.glmnet[, which(!unlist(lapply(dat.glmnet, function(x) any(is.na(x)))))] # if there are any Infs or constants
    train.glmnet = dat.glmnet[1:ntrain          , ]
    test.glmnet  = dat.glmnet[ntrain + (1:ntest), ]
    rm(dat.glmnet)
    
    nf = 'num.rank.glmnet'
    test.mf [[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, -1, test.glmnet, x.idx = 1:ncol(train.glmnet))
    train.mf[[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, train.cv.folds    ,  0, NULL       , x.idx = 1:ncol(train.glmnet))
    if (do.ccv) {
      for (ii in 1:config$nr.folds) {
        cvalid.mf[[ii]][[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, ii, NULL, x.idx = 1:ncol(train.glmnet))
      }
    }

    cat(date(), '   computing on YJT numericals\n')
    
    glmnet.features = as.vector(which(!unlist(lapply(dat, is.factor))))
    glmnet.features = glmnet.features[grep('^v', names(dat)[glmnet.features])]
    glmnet.features = intersect(glmnet.features, grep('_', names(dat), invert = T))
    pp = preProcess(dat[, glmnet.features], method = 'YeoJohnson')
    dat.glmnet = as.data.frame(randomForest::na.roughfix(predict(pp, dat[, glmnet.features]))) # glmnet will standardize
    dat.glmnet = dat.glmnet[, which(!unlist(lapply(dat.glmnet, function(x) any(is.na(x)))))] # if there are any Infs or constants
    train.glmnet = dat.glmnet[1:ntrain          , ]
    test.glmnet  = dat.glmnet[ntrain + (1:ntest), ]
    rm(dat.glmnet)
    
    nf = 'num.yj.glmnet'
    test.mf [[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, -1, test.glmnet, x.idx = 1:ncol(train.glmnet))
    train.mf[[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, train.cv.folds    ,  0, NULL       , x.idx = 1:ncol(train.glmnet))
    if (do.ccv) {
      for (ii in 1:config$nr.folds) {
        cvalid.mf[[ii]][[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, ii, NULL, x.idx = 1:ncol(train.glmnet))
      }
    }
    
    cat(date(), '   computing on dimred features\n')
    
    glmnet.features = grep('^dr.', names(dat))
    dat.glmnet = as.data.frame(model.matrix(~ . -1, dat[, glmnet.features])) # glmnet will standardize
    dat.glmnet = dat.glmnet[, which(!unlist(lapply(dat.glmnet, function(x) any(is.na(x)))))] # if there are any Infs or constants
    train.glmnet = dat.glmnet[1:ntrain          , ]
    test.glmnet  = dat.glmnet[ntrain + (1:ntest), ]
    rm(dat.glmnet)
    
    nf = 'num.dr.glmnet'
    test.mf [[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, -1, test.glmnet, x.idx = 1:ncol(train.glmnet))
    train.mf[[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, train.cv.folds    ,  0, NULL       , x.idx = 1:ncol(train.glmnet))
    if (do.ccv) {
      for (ii in 1:config$nr.folds) {
        cvalid.mf[[ii]][[nf]] = smm.generic2(train.glmnet, ancillary$train.labels, generate.glmnet.mf, ancillary$cv.folds, ii, NULL, x.idx = 1:ncol(train.glmnet))
      }
    }
    
    rm(train.glmnet, test.glmnet)
    
    test.mf$removeme = NULL
    train.mf$removeme = NULL
    if (do.ccv) {
      for (ii in 1:config$nr.folds) {
        cvalid.mf[[ii]]$removeme = NULL
      }
    }
    
    save(test.mf, train.mf, cvalid.mf, file = paste0(config$tmp.dir, '/meta-features-glmnet.RData'))
  }
  
  if (config$generate.ccv.xgb.features) {
    do.ccv   = F
    do.part1 = F
    do.part2 = T
    
    cat(date(), 'Generating XGB', ifelse(do.ccv, 'CCV', 'CV'), 'features\n') 
    
    xgb.params = list(
      objective         = 'binary:logistic',
      eval_metric       = 'logloss',
      nrounds           = 100,
      min_child_weight  = 20,
      max_depth         = 4
      # (otherwise using default parameters... TODO tune with nested CV! crazy because there
      # are too many tuning parameters, and it's not fast enough. Maybe use some heuristic given
      # the number of features)
    )
    
    # Because I want to use sparse.data.matrix, NAs have to be treated manually
    nafix = function(x) {
      if (is.factor(x)) {
        z = as.character(x)
        z[is.na(z)] = 'NA'
        return (as.factor(z))
      } else {
        z = x
        z[is.na(z)] = -999
        return (z)
      }
    }
    xgb.features = intersect(grep('^v', names(dat)), grep('_', names(dat), invert = T))
    dat.xgb = as.data.frame(lapply(dat[, xgb.features], nafix))
    ordinal.names = names(dat.xgb)[unlist(lapply(dat.xgb, function(x) !is.factor(x) && length(unique(x)) < 30))]
    for (f in ordinal.names) {
      dat.xgb[[paste0('cat.', f)]] = as.factor(dat.xgb[[f]])
    }

    generate.xgb.mf = function(f, prefix) {
      if (any(unlist(lapply(dat.xgb[, f], is.factor)))) {
        dall = xgb.DMatrix(sparse.model.matrix(~ . - 1, dat.xgb[, f, drop = F]), label = c(ancillary$train.labels, rep(0, length(test.idx))))
      } else {
        dall = xgb.DMatrix(data.matrix(dat.xgb[, f, drop = F]), label = c(ancillary$train.labels, rep(0, length(test.idx))))
      }
      dtrain = slice(dall, train.idx)
      dtest  = slice(dall, test.idx )
      
      nf = paste0(prefix, '.', paste0(f, collapse = '.'))
      
      # FIXME implement CCV
      
      train.mf[[nf]] <<- xgb.cv(params = xgb.params, folds = ancillary$cv.folds, verbose = F, prediction = T,
        nrounds = xgb.params$nrounds, nthread = ifelse(config$compute.backend != 'serial', 1, config$nr.threads),
        data = dtrain)$pred

      test.mf[[nf]] <<- predict(xgb.train(params = xgb.params, verbose = 0, 
        nrounds = xgb.params$nrounds, nthread = ifelse(config$compute.backend != 'serial', 1, config$nr.threads),
        data = dtrain), dtest)
    }

    if (do.part1) {
      test.mf    = data.frame(removeme = rep(NA, ntest )) # for predicting on the testset (as testset)
      train.mf   = data.frame(removeme = rep(NA, ntrain)) # for predicting on the testset (as trainset)
      cvalid.mf = list()                                  # for predicting on validation folds (as both trainset and testset)
      if (do.ccv) {
        for (i in 1:config$nr.folds) {
          cvalid.mf[[i]] = data.frame(removeme = rep(NA, ntrain))
        }
      }
      
      cat(date(), '   1-way categoricals\n')
      cat.features = names(dat.xgb)[unlist(lapply(dat.xgb, is.factor))]
      for (fi in 1:length(cat.features)) {
        cat('  working on', fi, 'of', length(cat.features), '\n')
        generate.xgb.mf(cat.features[fi], 'xgb.cat')
      }

      cat(date(), '   1-way numericals\n')
      load(file = 'quantize-numericals.RData') # => qnums
      qnums.to.include = qnums[complete.cases(qnums), ]
      qnums.to.include = qnums.to.include[abs(qnums.to.include$corz) - abs(qnums.to.include$corx) > 0.005, ]
      qnums.to.include = intersect(qnums.to.include$X, names(dat.xgb))
      for (fi in 1:length(qnums.to.include)) {
        cat('  working on', fi, 'of', length(qnums.to.include), '\n')
        generate.xgb.mf(qnums.to.include[fi], 'xgb.num')
      }
  
      cat(date(), '   2-way categoricals\n')
      load('factor-factor-interactions.RData') # => cc.pairs
      cc.pairs.to.include = cc.pairs[complete.cases(cc.pairs), ]
      cc.pairs.to.include = cc.pairs.to.include[cc.pairs.to.include$cor12 - pmax(cc.pairs.to.include$cor1, cc.pairs.to.include$cor2) > 0.005, ]
      cc.pairs.to.include = cc.pairs.to.include[cc.pairs.to.include$Var1 %in% names(dat.xgb) & cc.pairs.to.include$Var2 %in% names(dat.xgb), ]
      for (fi in 1:nrow(cc.pairs.to.include)) {
        cat('  working on', fi, 'of', nrow(cc.pairs.to.include), '\n')
        generate.xgb.mf(c(cc.pairs.to.include$Var1[fi], cc.pairs.to.include$Var2[fi]), 'xgb.cat2')
      }
  
      cat(date(), '   3-way categoricals\n')
      load('factor-3way-interactions.RData') # => cc.triplets
      cc.triplets.to.include = cc.triplets[complete.cases(cc.triplets), ]
      cc.triplets.to.include = cc.triplets.to.include[cc.triplets.to.include$cor123 - pmax(cc.triplets.to.include$cor1, cc.triplets.to.include$cor2, cc.triplets.to.include$cor3) > 0.01, ]
      cc.triplets.to.include = cc.triplets.to.include[order(cc.triplets.to.include$cor123, decreasing = T), ]
      cc.triplets.to.include = cc.triplets.to.include[cc.triplets.to.include$Var1 %in% names(dat.xgb) & cc.triplets.to.include$Var2 %in% names(dat.xgb) & cc.triplets.to.include$Var3 %in% names(dat.xgb), ]
      for (fi in 1:nrow(cc.triplets.to.include)) {
        cat('  working on', fi, 'of', nrow(cc.triplets.to.include), '\n')
        generate.xgb.mf(c(cc.triplets.to.include$Var1[fi], cc.triplets.to.include$Var2[fi], cc.triplets.to.include$Var3[fi]), 'xgb.cat3')
      }
  
      test.mf$removeme = NULL
      train.mf$removeme = NULL
      if (do.ccv) {
        for (ii in 1:config$nr.folds) {
          cvalid.mf[[ii]]$removeme = NULL
        }
      }
      
      save(test.mf, train.mf, cvalid.mf, file = paste0(config$tmp.dir, '/meta-features-xgb.RData'))
    }
    
    if (do.part2) {
      # TODO let's do this later
      gc()
      test.mf    = data.frame(removeme = rep(NA, ntest )) # for predicting on the testset (as testset)
      train.mf   = data.frame(removeme = rep(NA, ntrain)) # for predicting on the testset (as trainset)
      cvalid.mf = list()                                  # for predicting on validation folds (as both trainset and testset)
      if (do.ccv) {
        for (i in 1:config$nr.folds) {
          cvalid.mf[[i]] = data.frame(removeme = rep(NA, ntrain))
        }
      }
      
      cat(date(), '   2-way categorical X numerical\n')
      load('factor-numerical-interactions.RData') # => cn.pairs
      cn.pairs.to.include = cn.pairs[complete.cases(cn.pairs), ]
      cn.pairs.to.include = cn.pairs.to.include[cn.pairs.to.include$cor12 - pmax(cn.pairs.to.include$cor1, cn.pairs.to.include$cor2) > 0.01, ]
      cn.pairs.to.include = cn.pairs.to.include[order(cn.pairs.to.include$cor12, decreasing = T), ]
      cn.pairs.to.include = cn.pairs.to.include[cn.pairs.to.include$Var1 %in% names(dat.xgb) & cn.pairs.to.include$Var2 %in% names(dat.xgb), ]
      for (fi in 1:nrow(cn.pairs.to.include)) {
        cat('  working on', fi, 'of', nrow(cn.pairs.to.include), '\n')
        generate.xgb.mf(c(cn.pairs.to.include$Var1[fi], cn.pairs.to.include$Var2[fi]), 'xgb.catnum')
      }
      
      cat(date(), '   2-way numerical X numerical\n')
      load('numerical-numerical-interactions.RData') # => qqpairs
      qqpairs.to.include = qqpairs[complete.cases(qqpairs), ]
      qqpairs.to.include = qqpairs.to.include[qqpairs.to.include$cor12 - pmax(qqpairs.to.include$cor1, qqpairs.to.include$cor2) > 0.01, ]
      qqpairs.to.include = qqpairs.to.include[order(qqpairs.to.include$cor12, decreasing = T), ]
      qqpairs.to.include = qqpairs.to.include[qqpairs.to.include$Var1 %in% names(dat.xgb) & qqpairs.to.include$Var2 %in% names(dat.xgb), ]
      for (fi in 1:nrow(qqpairs.to.include)) {
        cat('  working on', fi, 'of', nrow(qqpairs.to.include), '\n')
        generate.xgb.mf(c(qqpairs.to.include$Var1[fi], qqpairs.to.include$Var2[fi]), 'xgb.numnum')
      }
  
      test.mf$removeme = NULL
      train.mf$removeme = NULL
      if (do.ccv) {
        for (ii in 1:config$nr.folds) {
          cvalid.mf[[ii]]$removeme = NULL
        }
      }
      
      save(test.mf, train.mf, cvalid.mf, file = paste0(config$tmp.dir, '/meta-features-xgb2.RData'))
    }
  }
}

# --------------------------------------------------------------------------------------------------

config$preprocess.raw = function(config) {
  if (config$do.generate.base.features) {
    config$generate.base.features1(config)
  }
  if (config$do.generate.meta.features) {
    config$generate.meta.features(config)
  }
  if (config$do.generate.base.features) {
    config$generate.base.features2(config)
  }
}

config$finalize.l0.data = function(config) {
  cat(date(), 'Finalizing data\n')
  
  #
  # Set training weights
  #

  load(paste0(config$tmp.dir, '/base-features1.RData')) # => dat, ancillary

  if (config$weighting.scheme == 'none') {
    # No weights
  } else if (config$weighting.scheme == 'partition') {
    cat('NOTE: Partitioning based on', config$weighting.var, '/', config$weighting.level, '\n')
    if (class(dat[[config$weighting.var]]) == 'factor') {
      # Partition on a categorical feature => use existing levels
      factor.weighting.level = levels(dat[[config$weighting.var]])[config$weighting.level]
      ancillary$w.train = ifelse(dat[1:length(ancillary$train.labels), config$weighting.var] == factor.weighting.level, config$weighting.param, 1 - config$weighting.param)
      ancillary$w.test = ifelse(dat[-(1:length(ancillary$train.labels)), config$weighting.var] == factor.weighting.level, config$weighting.param, 1 - config$weighting.param)
    } else {
      # Partition on a numerical feature => use the four quartiles
      x.part = dat[, config$weighting.var]
      x.part[is.na(x.part)] = median(x.part, na.rm = T)
      x.part = as.numeric(cut2(x.part, g = 4)) # might be better to just cut it on the trainset so it's more balanced...
      ancillary$w.train = ifelse(x.part[1:length(ancillary$train.labels)] == config$weighting.level, config$weighting.param, 1 - config$weighting.param)
      ancillary$w.test = ifelse(x.part[-(1:length(ancillary$train.labels))] == config$weighting.level, config$weighting.param, 1 - config$weighting.param)
    }
  } else {
    stop('wtf')
  }

  rm(dat)
  save(ancillary, file = config$ancillary.filename) # just to overwrite the weights
  gc()
  
  #
  # Load and combine the relevant parts
  #

  load(paste0(config$tmp.dir, '/base-features2.RData')) # => train.bf, test.bf

  train = train.bf
  test = test.bf
  rm(train.bf, test.bf)

  # FIXME this should be in the base features generation function
  add.darraghs.knn = function(genscheme, dmetric) {
    require(data.table)
    
    load(paste0(config$tmp.dir, '/knn_', genscheme, '_dist_target_0_', dmetric, '.RData')) # => d (a data.table)
    nni = c(1:10, 50, 100, 300, ncol(d) - 1)
    knd0 = as.data.frame(d[, 1 + nni, with = F])
    names(knd0) = paste0('d', nni)
    knd0$d2m1 = knd0$d2 - knd0$d1
    knd0$dm = apply(d[, 1:20, with = F], 1, median)
    knd0$ds = apply(d[, 1:20, with = F], 1, sd)
    names(knd0) = paste0('dar.', dmetric, '0.', names(knd0))
    
    load(paste0(config$tmp.dir, '/knn_', genscheme, '_dist_target_1_', dmetric, '.RData')) # => d (a data.table)
    
    knd1 = as.data.frame(d[, 1 + nni, with = F])
    names(knd1) = paste0('d', nni)
    knd1$d2m1 = knd1$d2 - knd1$d1
    knd1$dm = apply(d[, 1:20, with = F], 1, median)
    knd1$ds = apply(d[, 1:20, with = F], 1, sd)
    names(knd1) = paste0('dar.', dmetric, '1.', names(knd1))
    
    knd1$darragh.knn.d01.1 = knd1[, 1] - knd0[, 1]
    knd1$darragh.knn.d01.10 = knd1[, 10] - knd0[, 10]
    knd1$darragh.knn.dss = rowSums(knd1[, 1:10] > knd0[, 1:10])
    
    train <<- cbind(train, knd0[1:nrow(train)            , ], knd1[1:nrow(train)            , ])
    test  <<- cbind(test , knd0[(nrow(train) + 1):nrow(d), ], knd1[(nrow(train) + 1):nrow(d), ])
  }
  
  if (config$add.loo.knn.features) {
    # FIXME drop my LOO ones? maybe not since they are different metric and num/cat separated
    add.darraghs.knn('loo', 'braycurtis')
    add.darraghs.knn('loo', 'hamming')
    add.darraghs.knn('loo', 'canberra')
    add.darraghs.knn('loo', 'manhattan')
  } else if (config$add.cv.knn.features) {
    add.darraghs.knn('cv', 'canberra')
    # TODO: if this doesn't leak, add the other distances
  }
  
  add.mf = function(group.name, standard.cv.override = F) {
    load(paste0(config$tmp.dir, '/meta-features-', group.name, '.RData')) # => test.mf, train.mf, cvalid.mf
    
    if (!standard.cv.override && !is.null(config$in.fold) && config$in.fold != -1) {
      # Cross validation mode
      train <<- cbind(train, cvalid.mf[[config$in.fold]])
    } else {
      # Full training mode
      train <<- cbind(train, train.mf)
    }
    
    test <<- cbind(test, test.mf)
  }
  
  if (config$add.ccv.yenc.categoricals) {
    add.mf('yenc-cat')
  }
  
  if (config$add.ccv.yenc.numericals) {
    add.mf('yenc-qnum')
  }
  
  if (config$add.ccv.yenc.interactions) {
    add.mf('yenc-cat-cat')
    # FIXME figure out the number of chunks automatically
    add.mf('yenc-cat-cat-cat-1')
    add.mf('yenc-cat-cat-cat-2')
    add.mf('yenc-cat-cat-cat-3')
    add.mf('yenc-cat-qnum-1')
    add.mf('yenc-cat-qnum-2')
    add.mf('yenc-cat-qnum-3')
    add.mf('yenc-cat-qnum-4')
    add.mf('yenc-cat-qnum-5')
    add.mf('yenc-cat-qnum-6')
    add.mf('yenc-qnum-qnum-1')
    add.mf('yenc-qnum-qnum-2')
    add.mf('yenc-qnum-qnum-3')
    add.mf('yenc-qnum-qnum-4')
    add.mf('yenc-qnum-qnum-5')
    add.mf('yenc-qnum-qnum-6')
    add.mf('yenc-qnum-qnum-7')
    add.mf('yenc-qnum-qnum-8')
    add.mf('yenc-qnum-qnum-9')
  }
  
  if (config$add.ccv.knn.features) {
    add.mf('knn-cat')
    add.mf('knn-num')
  }
  
  if (config$add.ccv.glmnet.features) {
    add.mf('glmnet', standard.cv.override = T)
  }
  
  if (config$add.ccv.xgb.features) {
    add.mf('xgb', standard.cv.override = T)
    add.mf('xgb2', standard.cv.override = T)
  }
  
  #
  # Final preprocessing
  #

  if (0) {
    cat(date(), 'EXPERIMENT: dropping all even features in order of appearance\n')
    colidx = seq(2, ncol(train) - 1, by = 2)
    train = train[, colidx]
    test = test[, colidx]
  }
  
  # Most models can't handle NAs, and those that can are (usually) better off without this  
  if (!config$model.select %in% c('xgb', 'rf', 'nb')) {
    cat(date(), 'Imputing all missing values\n')
    dat = randomForest::na.roughfix(rbind(train, test))
    still.na = (colSums(is.na(dat)) > 0)
    if (any(still.na)) {
      cat('Some NAs still in', names(dat)[still.na], '? dropping\n')
      dat = dat[, !still.na]
    }
    train = dat[1:nrow(train)              , ]
    test  = dat[(nrow(train) + 1):nrow(dat), ]
    rm(dat)
  }

  if (config$drop.original.dr.features) {
    cat(date(), 'Dropping original dim reduce features\n')
    to.drop = names(train)[c(grep('^hexv.', names(train)), grep('^dr.', names(train)))]
    train = train[, !(names(train) %in% to.drop)]
    test  = test [, !(names(test ) %in% to.drop)]
  }
  
  if (config$drop.importance.below > 0) {
    cat(date(), 'Dropping features with gain lower than the', config$drop.importance.below, 'quantile\n')
    load('xgb-feature-importance.RData') # => impo
    to.drop = which(names(train) %in% impo$Feature[impo$Gain < quantile(impo$Gain, probs = config$drop.importance.below)])
    train = train[, -to.drop]
    test  = test [, -to.drop]
  }
  
  if (config$drop.importance.above < 1) {
    cat(date(), 'Dropping features with gain higher than the', config$drop.importance.above, 'quantile\n')
    load('xgb-feature-importance.RData') # => impo
    to.drop = which(names(train) %in% impo$Feature[impo$Gain > quantile(impo$Gain, probs = config$drop.importance.above)])
    train = train[, -to.drop]
    test  = test [, -to.drop]
  }
  
  if (0) {
    cat('EXPERIMENT: dropping some things\n')
    # Original features:
    #to.drop = grep('_', names(train)[grep('^v', names(train))], invert = T)  
    # OHE:
    to.drop = grep('_', names(train))
    train = train[, -to.drop]
    test  = test [, -to.drop]
  }
  
  # Some models work better on scaled features (e.g., nnet, maybe knn)
  if (config$scale.all) {
    cat(date(), 'Scaling all features\n')
    dat = as.data.frame(scale(rbind(train, test)))
    train = dat[1:nrow(train)              , ]
    test  = dat[(nrow(train) + 1):nrow(dat), ]
    rm(dat)
  }
  
  #
  # Save data
  #

  if (config$proper.validation) {
    valid = test[1:length(ancillary$valid.labels), ]
    test  = test[(length(ancillary$valid.labels) + 1):nrow(test), ]
  } else {
    valid = NULL
  }
  
  if (config$model.select == 'xgb') {
    if (config$weighting.scheme == 'none') {
      dtrain = xgb.DMatrix(dat = data.matrix(train), missing = NA, label = ancillary$train.labels)
    } else {
      dtrain = xgb.DMatrix(dat = data.matrix(train), missing = NA, label = ancillary$train.labels, weight = ancillary$w.train)
    }
    dtest = xgb.DMatrix(dat = data.matrix(test), missing = NA)
    xgb.DMatrix.save(dtrain, config$xgb.trainset.filename)
    xgb.DMatrix.save(dtest , config$xgb.testset.filename)
    if (config$proper.validation) {
      dvalid = xgb.DMatrix(dat = data.matrix(valid), label = ancillary$valid.labels, missing = NA)
      xgb.DMatrix.save(dvalid, config$xgb.validset.filename)
    }
    feature.names = names(train)
    save(feature.names, file = paste0(config$tmp.dir, '/xgb-feature-names.RData'))
  } else {
    train$target = ancillary$train.labels
    save(train, valid, test, file = config$dataset.filename)
  }
}

config$collect.important.l0.features = function(config) {
  load(config$ancillary.filename) # => ancillary
  load(paste0(config$tmp.dir, '/base-features2.RData')) # => train.bf, test.bf
  
  # These leak...
  to.drop = c(grep('^num.nn', names(train.bf)), grep('^cat.nn', names(train.bf)))
  train.bf = train.bf[, -to.drop]
  test.bf  = test.bf [, -to.drop]
  
  train = train.bf
  test = test.bf
  rm(train.bf, test.bf)
  
  add.mf = function(group.name) {
    load(paste0(config$tmp.dir, '/meta-features-', group.name, '.RData')) # => test.mf, train.mf, cvalid.mf
    train <<- cbind(train, train.mf)
    test <<- cbind(test, test.mf)
  }

  add.mf('yenc-cat')
  add.mf('yenc-qnum')
  add.mf('yenc-cat-cat')
  add.mf('yenc-cat-cat-cat-1')
  add.mf('yenc-cat-cat-cat-2')
  add.mf('yenc-cat-cat-cat-3')
  add.mf('yenc-cat-qnum-1')
  add.mf('yenc-cat-qnum-2')
  add.mf('yenc-cat-qnum-3')
  add.mf('yenc-cat-qnum-4')
  add.mf('yenc-cat-qnum-5')
  add.mf('yenc-cat-qnum-6')
  add.mf('yenc-qnum-qnum-1')
  add.mf('yenc-qnum-qnum-2')
  add.mf('yenc-qnum-qnum-3')
  add.mf('yenc-qnum-qnum-4')
  add.mf('yenc-qnum-qnum-5')
  add.mf('yenc-qnum-qnum-6')
  add.mf('yenc-qnum-qnum-7')
  add.mf('yenc-qnum-qnum-8')
  add.mf('yenc-qnum-qnum-9')
  add.mf('knn-cat')
  add.mf('knn-num')
  add.mf('glmnet')
  
  load('xgb-feature-importance.RData') # => impo
  train.impo = train[, which(names(train) %in% impo$Feature[1:100])]
  test.impo  = test [, which(names(test ) %in% impo$Feature[1:100])]
  save(train.impo, test.impo, file = paste0(config$tmp.dir, '/top100-L0-features.RData'))
}

config$preprocess.stack = function(config) {
  load(paste0(config$tmp.dir, '/pp-data-ancillary-L', config$layer - 1, '.RData')) # => ancillary
  
  train = NULL
  test = NULL
  
  cat(date(), 'Loading meta-features\n')
  
  if (config$stack.darragh) {
    darragh = read_csv('Team/darragh-level1.csv', progress = F)
    cat(date(), 'Including Darragh\'s models =>', ncol(darragh) - 1, '\n')
    if (is.null(test)) {
      train = darragh[!(darragh$ID %in% ancillary$test.ids), -1]
      test  = darragh[  darragh$ID %in% ancillary$test.ids , -1]
    } else {
      train = cbind(train, darragh[!(darragh$ID %in% ancillary$test.ids), -1])
      test  = cbind(test , darragh[  darragh$ID %in% ancillary$test.ids , -1])
    }
  }

  if (config$stack.giba) {
    giba = as.data.frame(data.table::fread('Team/giba_models.csv', showProgress = F))
    cat(date(), 'Including giba\'s models =>', ncol(giba) - 1, '\n')
    if (is.null(test)) {
      train = giba[!(giba$ID %in% ancillary$test.ids), -1]
      test  = giba[  giba$ID %in% ancillary$test.ids , -1]
    } else {
      train = cbind(train, giba[!(giba$ID %in% ancillary$test.ids), -1])
      test  = cbind(test , giba[  giba$ID %in% ancillary$test.ids , -1])
    }
  }
  
  if (config$stack.sk) {
    sk = read_csv('Team/SK-level1-meta-features.csv', progress = F)
    cat(date(), 'Including my shared models part I =>', ncol(sk) - 1, '\n')
    train = cbind(train, sk[!(sk$ID %in% ancillary$test.ids), -1])
    test  = cbind(test , sk[  sk$ID %in% ancillary$test.ids , -1])
    sk = read_csv('Team/SK-level1-meta-features2.csv', progress = F)
    cat(date(), 'Including my shared models part II =>', ncol(sk) - 1, '\n')
    train = cbind(train, sk[!(sk$ID %in% ancillary$test.ids), -1])
    test  = cbind(test , sk[  sk$ID %in% ancillary$test.ids , -1])
  }
  
  if (1) {
    cat(date(), 'Removing questionable models\n')
    print(config$questionable.models)
    train = train[, !(names(train) %in% config$questionable.models)]
    test  = test [, !(names(test ) %in% config$questionable.models)]
  } else if (0) {
    cat(date(), 'Removing leaking models\n')
    print(config$questionable.models)
    train = train[, !(names(train) %in% config$leaking.models)]
    test  = test [, !(names(test ) %in% config$leaking.models)]
  } else if (0) {
    cat(date(), 'Using only leaking models\n')
    print(config$questionable.models)
    train = train[, names(train) %in% config$leaking.models, drop = F]
    test  = test [, names(test ) %in% config$leaking.models, drop = F]
  }
  
  if (1) {
    cat(date(), 'Merging partitioned models', config$partition.models, '\n')
    for (m in config$partition.models) {
      part.idxs = grep(paste0('^', m), names(train))
      train[[m]] = rowSums(train[, part.idxs], na.rm = T)
      train[rowMeans(is.na(train[, part.idxs])) == 1, m] = NA
      train = train[, -part.idxs]
      test[[m]] = rowSums(test[, part.idxs], na.rm = T)
      test[rowMeans(is.na(test[, part.idxs])) == 1, m] = NA
      test = test[, -part.idxs]
    }
  }
  
  cvscores = data.frame(cv.logloss = rep(NA, ncol(train)))
  rownames(cvscores) = names(train)
  for (i in 1:ncol(train)) {
    cvscores$cv.logloss[i] = config$eval.logloss.core(train[[i]], ancillary$train.labels)
  }
  d.train = cor(train, method = 'spearman', use = 'complete.obs')
  d.test  = cor(test , method = 'spearman', use = 'complete.obs')
  dd = abs(d.train - d.test)
  #image(dd) # needs to be kind of homogeneous
  if (any(rowMeans(dd) > 0.01)) {
    cat('WARNING: some models seem inconsistent in test:\n')
    print(data.frame(mean.cor.diff = rowMeans(dd)[rowMeans(dd) > 0.01]))
  }
  #browser()

  if (1) {
    cat(date(), 'Averaging extremely correlated features:\n')
    # FIXME do local isotonic? NNLS?
    merge.groups = find.correlated.groups(d.train, 0.99)
    for (g in merge.groups) {
      g.name = paste0('GROUP.', names(train)[g[1]])
      cat('  ', g.name, ': ', paste(names(train)[g], collapse = ', '), '\n', sep = '')
      train[[g.name]] = rowMeans(train[, g], na.rm = T)
      test [[g.name]] = rowMeans(test [, g], na.rm = T)
    }
    train = train[, -unlist(merge.groups), drop = F]
    test  = test [, -unlist(merge.groups), drop = F]
  }

  if (0) {
    # Look at the CV scores and correlation of the models we are going to stack
    # Warning: I probably want to reduce the number of models...
    require(corrplot)
    #pdf('l1-models-corr.pdf')
    corrplot(d.test, is.corr = F, method = 'color', col = colorRampPalette(c('white', 'green', 'red'))(40), addCoef.col = 'black', order = 'AOE', cl.lim = c(0.66, 1), title = 'Score correlations of constituent models')
    #dev.off()
    #View(cvscores)
    #View(d)
  }

  if (config$add.top.l0.to.stack) {
    cat('EXPERIMENT: Adding top 100 raw features from L0\n')
    load(file = paste0(config$tmp.dir, '/top100-L0-features.RData')) # => train.impo, test.impo
    if (config$model.select != 'xgb') {
      dat = as.data.frame(lapply(rbind(train.impo, test.impo), function(x) {xx = x; xx[!is.finite(x)] = NA; xx}))
      dat = scale(randomForest::na.roughfix(dat))
      train.impo = dat[  1:nrow(train.impo) , ]
      test.impo  = dat[-(1:nrow(train.impo)), ]
      rm(dat)
    }
    train = cbind(train, train.impo)
    test  = cbind(test , test.impo )
  }
  
  if (0) {
    # Generate a CV partition that is independent of the one used in L0
    # FIXME maybe it is somehow better to keep the same one? maintain the structure that will be in the testset?
    set.seed(config$rng.seed)
    ancillary$cv.folds = createFolds(as.factor(ancillary$train.labels), k = config$nr.folds)
  } else {
    # Group folds
    fold.idxs = read_csv('Team/index.csv', col_names = F, progress = F)
    cv.folds = list()
    cv.folds$Fold1 = which(fold.idxs == 0)
    cv.folds$Fold2 = which(fold.idxs == 1)
    cv.folds$Fold3 = which(fold.idxs == 2)
    cv.folds$Fold4 = which(fold.idxs == 3)
    cv.folds$Fold5 = which(fold.idxs == 4)
  }

  # Override per-level ancillary info
  ancillary$feature.names = names(train)
  ancillary$cv.folds = cv.folds
  
  cat(date(), 'Saving data\n')

  # Data for everything but XGB
  train$target = ancillary$train.labels
  save(ancillary, file = config$ancillary.filename)
  save(train, test, file = config$dataset.filename)
  train$target = NULL
  
  # Data for XGB  
  dtrain = xgb.DMatrix(dat = data.matrix(train), label = ancillary$train.labels, missing = NA)
  dtest  = xgb.DMatrix(dat = data.matrix(test )                                , missing = NA)
  xgb.DMatrix.save(dtrain, config$xgb.trainset.filename)
  xgb.DMatrix.save(dtest , config$xgb.testset.filename)
}

config$preprocess = function(config) {
  if (config$layer == 0) {
    config$preprocess.raw(config)
  } else {
    config$preprocess.stack(config)
  }
}

# --------------------------------------------------------------------------------------------------
# Loss function

# NOTE: the 1e-15 clamping is how the competition defined it

# Version for XGB
config$eval.logloss = function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  weights = getinfo(dtrain, 'weight')
  preds = pmax(pmin(preds, 1 - 1e-15), 1e-15)
  if (length(weights) == 0) {
    logloss = -mean(log(ifelse(labels == 0, 1 - preds, preds)))
  } else {
    logloss = -sum(weights * log(ifelse(labels == 0, 1 - preds, preds))) / sum(weights)
  }
  return (list(metric = 'logloss', value = logloss))
}

# Version for XGB with multiple limits
config$eval.logloss.ltd = function(preds, dtrain, ntreelimits) {
  labels = getinfo(dtrain, 'label')
  weights = getinfo(dtrain, 'weight')
  preds = pmax(pmin(preds, 1 - 1e-15), 1e-15)
  logloss = NULL
  for (i in 1:ncol(preds)) {
    if (length(weights) == 0) {
      logloss = c(logloss, -mean(log(ifelse(labels == 0, 1 - preds[, i], preds[, i]))))
    } else {
      logloss = c(logloss, -sum(weights * log(ifelse(labels == 0, 1 - preds[, i], preds[, i]))) / sum(weights))
    }
  }
  names(logloss) = ntreelimits
  return (list(metric = 'logloss', value = logloss))
}

# Version for other models (single prediction)
config$eval.logloss.core = function(preds, labels, weights = NULL) {
  preds = pmax(pmin(preds, 1 - 1e-15), 1e-15)
  if (is.null(weights)) {
    return (-mean(log(ifelse(labels == 0, 1 - preds, preds)), na.rm = T))
  } else {
    return (-sum(weights * log(ifelse(labels == 0, 1 - preds, preds))) / sum(weights))
  }
}

# Version for other models (multiple predictions)
config$eval.logloss.core.multi = function(preds, labels, weights = NULL) {
  preds = pmax(pmin(preds, 1 - 1e-15), 1e-15)
  logloss = NULL
  for (i in 1:ncol(preds)) {
    if (is.null(weights)) {
      logloss = c(logloss, -mean(log(ifelse(labels == 0, 1 - preds[, i], preds[, i]))))
    } else {
      logloss = c(logloss, -sum(weights * log(ifelse(labels == 0, 1 - preds[, i], preds[, i]))) / sum(weights))
    }
  }
  return (logloss)
}

# --------------------------------------------------------------------------------------------------
# Model fitting

config$train.xgb = function(config) {
  dtrain = xgb.DMatrix(config$xgb.trainset.filename)
  load(config$ancillary.filename) # => ancillary

  if (config$xgb.params$reweight != 1) {
    y = getinfo(dtrain, 'label')
    w = ifelse(y == 0, 1, config$xgb.params$reweight)
    setinfo(dtrain, 'weight', w)
  }
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    dvalid = slice(dtrain, vidx)
    dtrain = slice(dtrain, (1:nrow(dtrain))[-vidx])
    w = getinfo(dtrain, 'weight')
    if (any(!is.na(w) & w == 0)) {
      dtrain = slice(dtrain, which(w > 0))
    }
    watchlist = list(valid = dvalid, train = dtrain)
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(dtrain), 0.1 * nrow(dtrain))
    dvalid = slice(dtrain, vidx)
    dtrain = slice(dtrain, (1:nrow(dtrain))[-vidx])
    watchlist = list(valid = dvalid, train = dtrain)
  } else {
    w = getinfo(dtrain, 'weight')
    if (any(!is.na(w) & w == 0)) {
      dtrain = slice(dtrain, which(w > 0))
    }
    watchlist = list(train = dtrain)
  }

  if (config$proper.validation) {
    dproper = xgb.DMatrix(config$xgb.validset.filename)
    watchlist = c(watchlist, proper = dproper)
  }
  
  cat(date(), 'Training xgb model\n')
  
  set.seed(config$rng.seed)
  
  xgb.fit = xgb.train(
    params            = config$xgb.params,
    nrounds           = config$xgb.params$nrounds,
    maximize          = (objective == 'rank:pairwise'),
    data              = dtrain,
    watchlist         = watchlist,
   #early.stop.round  = config$xgb.early.stop.round, 
    print.every.n     = min(max(100, 100 * round(ceiling(config$xgb.params$nrounds / 20) / 100)), 250),
    nthread           = ifelse(config$compute.backend != 'serial', 1, config$nr.threads)
  )
  
  if (config$measure.importance) {
    cat(date(), 'Examining importance of features in the single XGB model\n')
    load(paste0(config$tmp.dir, '/xgb-feature-names.RData')) # => feature.names
    impo = xgb.importance(feature.names, model = xgb.fit)
    print(impo[1:50, ])
    save(impo, file = 'xgb-feature-importance.RData')
  }
  
  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    if (is.null(config$xgb.params$booster) || config$xgb.params$booster == 'gbtree') {
      preds = predict.xgb.ltd(xgb.fit, dvalid, config$xgb.params$ntreelimits, config$xgb.params$predleaf)
      logloss = config$eval.logloss.ltd(preds, dvalid, config$xgb.params$ntreelimits)$value
    } else {
      preds = predict(xgb.fit, dvalid)
      logloss = config$eval.logloss(preds, dvalid)$value
    }
    cat(date(), 'Validation score:\n')
    print(logloss)
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    dtest = xgb.DMatrix(config$xgb.testset.filename)
    if (is.null(config$xgb.params$booster) || config$xgb.params$booster == 'gbtree') {
      preds = predict.xgb.ltd(xgb.fit, dtest, config$xgb.params$ntreelimits, config$xgb.params$predleaf)
    } else {
      preds = predict(xgb.fit, dtest)
    }
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)

    if (config$proper.validation) {
      preds = predict.xgb.ltd(xgb.fit, dproper, config$xgb.params$ntreelimits)
      logloss = config$eval.logloss.ltd(preds, dproper, config$xgb.params$ntreelimits)$value
      cat(date(), 'Proper validation score:\n')
      print(logloss)
    }
  }
}

config$train.ranger = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename)
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training ranger model\n')
  
  set.seed(config$rng.seed)

  if (config$ranger.params$reweight.dummies) {
    split.select.weights = rep(1, ncol(train) - 1)
    cat.idx = grep('_', names(train))
    cat.names = names(train)[cat.idx]
    cat.var.ids = as.numeric(as.factor(unlist(lapply(strsplit(cat.names, split = '_'), function(x) x[[1]]))))
    cat.var.weights = 1 / table(cat.var.ids)
    split.select.weights[cat.idx] = as.vector(cat.var.weights[cat.var.ids])
  } else {
    split.select.weights = NULL
  }
  
  ranger.fit = ranger(
    as.factor(target) ~ ., train, 
    num.trees = config$ranger.params$num.trees, 
    mtry = config$ranger.params$mtry, 
    split.select.weights = ,
    verbose = T, 
    write.forest = T, 
    probability = T, 
    num.threads = ifelse(config$compute.backend != 'serial', 1, config$nr.threads))
  
  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = predict(ranger.fit, valid)$predictions[, 2]
    logloss = config$eval.logloss.core(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = predict(ranger.fit, test)$predictions[, 2]
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train.best = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training best model\n')
  
  set.seed(config$rng.seed)

  all.loglosss = rep(Inf, ncol(train))
  for (i in 1:ncol(train)) {
    if (names(train)[i] != 'target') {
      all.loglosss[i] = config$eval.logloss.core(train[[i]], train$target)
    }
  }
  best.tag = names(train)[which.min(all.loglosss)]
  logloss = min(all.loglosss)
  cat(date(), ' Best model: ', best.tag, ' (logloss ', logloss, ')\n', sep = '')

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = valid[[best.tag]]
    logloss = config$eval.logloss.core(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = test[[best.tag]]
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train.blend = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  # Thid model doesn't really need any training
  
  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = rowMeans(valid[, -which(names(valid) == 'target'), drop = F])
    logloss = config$eval.logloss.core(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = rowMeans(test[, -which(names(test) == 'target'), drop = F])
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train.glm = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training glm model\n')
  
  mdl = glm(target ~ ., train, family = binomial)

  # For L0, I tried:
  #mdl = stepAIC(mdl, target ~ ., direction = 'backward')
  # =======> 
  #mdl = glm(target ~ v6 + v12 + v14 + v21 + v31_A + v39 + v40 + v50 + v66_B + 
  #  v66_C + v82 + v90 + v99 + v102 + v114 + v119 + v120 + v131 + 
  #  v22_.hexv + v24_.sms + v30_.sms + v47_.sms + v52_.sms + v56_.sms + 
  #  v71_.sms + v75_.sms + v79_.sms + v107_.sms + v112_.sms + 
  #  v113_.sms + v125_.sms, train, family = binomial)
  
  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = predict(mdl, valid, type = 'response')
    logloss = config$eval.logloss.core(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = predict(mdl, test, type = 'response')
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train.nnls = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training nnls model\n')

  beta = coef(nnls(data.matrix(train[, -which(names(train) == 'target'), drop = F]), train$target))
  beta = beta / sum(beta) # FIXME hmm.. in this binary response case I should implement nnglm with optim or something
  
  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = c(data.matrix(valid[, -which(names(valid) == 'target'), drop = F]) %*% beta)
    logloss = config$eval.logloss.core(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = data.matrix(test) %*% beta
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train.glmnet = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training glmnet model\n')
  
  set.seed(config$rng.seed)

  # One option is to let is tune lambda using cv.glmnet. Another is to use multiple values and stack later.

  #registerDoParallel(config$nr.threads)
  #mdl = cv.glmnet(data.matrix(train[, -which(names(train) == 'target'), drop = F]), train$target, family = 'binomial', lambda = config$glmnet.params$lambda, alpha = config$glmnet.params$alpha, parallel = T)
  #stopImplicitCluster()

  mdl = glmnet(data.matrix(train[, -which(names(train) == 'target'), drop = F]), train$target, family = 'binomial', lambda = config$glmnet.params$lambda, alpha = config$glmnet.params$alpha)

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = predict(mdl, data.matrix(valid[, -which(names(valid) == 'target'), drop = F]), type = 'response')
    logloss = config$eval.logloss.core.multi(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = predict(mdl, data.matrix(test), type = 'response')
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train.et = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training extra-trees model\n')
  
  set.seed(config$rng.seed)

  mdl = extraTrees(
    x = data.matrix(train[, -which(names(train) == 'target'), drop = F]),
    y = as.factor(train$target),
    ntree = config$et.params$ntree,
    mtry = config$et.params$mtry,
    nodesize = config$et.params$nodesize,
    numThreads = ifelse(config$compute.backend != 'serial', 1, config$nr.threads)
  )
  
  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = predict(mdl, data.matrix(valid[, -which(names(valid) == 'target'), drop = F]), probability = T)[, 2]
    logloss = config$eval.logloss.core(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = predict(mdl, data.matrix(test), probability = T)[, 2]
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train.rf = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training random forest model\n')
  
  set.seed(config$rng.seed)
  
  if (0) {
    # Train this model only on the categorical features
    cat.features = names(train)[which(lapply(train, class) == 'factor')]
    frmla = as.formula(paste('as.factor(target) ~', paste(cat.features, collapse = ' + ')))
  } else {
    frmla = (target ~ .)
  }
  
  # mdl = randomForest(frmla, train, do.trace = 100,
  #   ntree = config$rf.params$ntree,
  #   mtry = config$rf.params$mtry,
  #   nodesize = config$rf.params$nodesize
  # )
  
  h2o.init(nthreads = -1, max_mem_size = '16G')
  train$target = as.factor(train$target)
  feature.names = names(train)[names(train) != 'target']
  train.for.h2o = as.h2o(train)
  mdl = h2o.randomForest(feature.names, 'target', train.for.h2o, mtries = config$rf.params$mtry, ntrees = config$rf.params$ntree, max_depth = config$rf.params$max.depth, binomial_double_trees = config$rf.params$binomial_double_trees)
  # TODO: tune nbins_cats (leaving factors in the data) ...

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    #preds = predict(mdl, valid, type = 'prob')[, 2]
    valid.for.h2o = as.h2o(valid)
    preds = as.data.frame(h2o.predict(mdl, valid.for.h2o))[, 3]
    
    logloss = config$eval.logloss.core(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    #preds = predict(mdl, test, type = 'prob')[, 2]
    test.for.h2o = as.h2o(test)
    preds = as.data.frame(h2o.predict(mdl, test.for.h2o))[, 3]
    
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train.knn = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training knn model\n')
  # (actual training is separate for valid and test)

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    knn.res = nn2(train[, -which(names(train) == 'target')], query = valid[, -which(names(valid) == 'target')],  k = config$knn.params$k, eps = config$knn.params$eps)
    knn.preds = matrix(train$target[knn.res$nn.idx], ncol = config$knn.params$k)
    preds = rowMeans(knn.preds)
    #ownn.res = ownn(train[, -which(names(train) == 'target')], valid[, -which(names(valid) == 'target')], cl = train$target, k = config$knn.params$k, prob = T)
    #preds = ifelse(ownn.res$ownnpred == 1, attr(ownn.res$ownnpred, 'prob'), 1 - attr(ownn.res$ownnpred, 'prob'))

    logloss = config$eval.logloss.core(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    knn.res = nn2(train[, -which(names(train) == 'target')], query = test,  k = config$knn.params$k, eps = config$knn.params$eps)
    knn.preds = matrix(train$target[knn.res$nn.idx], ncol = config$knn.params$k)
    preds = rowMeans(knn.preds)
    #ownn.res = ownn(train[, -which(names(train) == 'target')], test, cl = train$target, k = config$knn.params$k, prob = T)
    #preds = ifelse(ownn.res$ownnpred == 1, attr(ownn.res$ownnpred, 'prob'), 1 - attr(ownn.res$ownnpred, 'prob'))

    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train.iso = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training iso model\n')
  
  #stopifnot(ncol(train) == 2) # only supports univariate isoreg for now (maybe use liso.. but heavy for this sample size!)
  #isofun = as.stepfun(isoreg(, train$target))
  fit = iam.train(train[, -which(names(train) == 'target')], train$target)

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    #preds = isofun(data.matrix(valid[, -which(names(valid) == 'target')]))
    preds = iam.predict(fit, valid[, -which(names(valid) == 'target')])
    logloss = config$eval.logloss.core(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    if (ncol(test) == 1) {
      test = c(data.matrix(test))
    }
    #preds = isofun(test)
    preds = iam.predict(fit, test)
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train.nnet = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training nnet model\n')
  
  h2o.init(nthreads = -1, max_mem_size = '16G')
  train$target = as.factor(train$target)
  feature.names = names(train)[names(train) != 'target']
  train.for.h2o = as.h2o(train)
  
  mdl = h2o.deeplearning(feature.names, 'target', train.for.h2o,
    activation = 'TanhWithDropout',
    hidden = c(1024, 1024),
    input_dropout_ratio = 0.05,
    hidden_dropout_ratios = c(0.9, 0.9),
    rate = 0.001,
    epochs = 100,
    #l1 = 1e-7,
    #l2 = 1e-7,
    #train_samples_per_iteration = 2000,
    #max_w2 = 10,
    stopping_metric = 'logloss',
    seed = config$rng.seed
  )

  # Things to try:
  # - Leave factors in the data and let H2O handle them
  # - Other activation functions
  # - Other tolopogies...
  # - Look at logs and try to figure out how many epochs...
  # - Adaptive learning (enable, and tune rho & epsilon...)
  # - Regularization...

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    valid.for.h2o = as.h2o(valid)
    preds = as.data.frame(h2o.predict(mdl, valid.for.h2o))$p1
    
    logloss = config$eval.logloss.core(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    test.for.h2o = as.h2o(test)
    preds = as.data.frame(h2o.predict(mdl, test.for.h2o))$p1
    
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train.nb = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training nnet model\n')
  
  h2o.init(nthreads = -1, max_mem_size = '16G')
  train$target = as.factor(train$target)
  feature.names = names(train)[names(train) != 'target']
  train.for.h2o = as.h2o(train)
  
  mdl = h2o.naiveBayes(feature.names, 'target', train.for.h2o)

  # It's not really useful here at all...
  # Things to try:
  # - Leave factors and NAs in the data and let H2O handle them
  # - Smoothing and thresholding
  
  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    valid.for.h2o = as.h2o(valid)
    preds = as.data.frame(h2o.predict(mdl, valid.for.h2o))$p1
    
    logloss = config$eval.logloss.core(preds, valid$target)
    cat(date(), 'Validation score:', logloss, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      w.train = ancillary$w.train
      save(preds, w.train, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    test.for.h2o = as.h2o(test)
    preds = as.data.frame(h2o.predict(mdl, test.for.h2o))$p1
    
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    w.test = ancillary$w.test
    save(preds, w.test, file = fnm)
  }
}

config$train = function(config) {
  if (config$layer == 0) {
    if (0) {
      cat('!!!!!!!!!!!!!!!!! NOTE skipping finalization for debugging/tuning\n')
    } else {
      config$finalize.l0.data(config)
    }
  }
    
  if (config$model.select == 'best') {
    return (config$train.best(config))
  } else if (config$model.select == 'blend') {
    return (config$train.blend(config))
  } else if (config$model.select == 'glm') {
    return (config$train.glm(config))
  } else if (config$model.select == 'nnls') {
    return (config$train.nnls(config))
  } else if (config$model.select == 'glmnet') {
    return (config$train.glmnet(config))
  } else if (config$model.select == 'xgb') {
    return (config$train.xgb(config))
  } else if (config$model.select == 'ranger') {
    return (config$train.ranger(config))
  } else if (config$model.select == 'et') {
    return (config$train.et(config))
  } else if (config$model.select == 'rf') {
    return (config$train.rf(config))
  } else if (config$model.select == 'knn') {
    return (config$train.knn(config))
  } else if (config$model.select == 'iso') {
    return (config$train.iso(config))
  } else if (config$model.select == 'nnet') {
    return (config$train.nnet(config))
  } else if (config$model.select == 'nb') {
    return (config$train.nb(config))
  } else {
    stop('wtf')
  }
}

# --------------------------------------------------------------------------------------------------
# Cross validation

config$cross.validate.xgb = function(config, nfold = 5) {
  stopifnot(config$layer == 0)
  config$finalize.l0.data(config)
  dtrain = xgb.DMatrix(config$xgb.trainset.filename)
  load(config$ancillary.filename) # => ancillary
  
  if (config$xgb.params$reweight != 1) {
    y = getinfo(dtrain, 'label')
    w = ifelse(y == 0, 1, config$xgb.params$reweight)
    setinfo(dtrain, 'weight', w)
  }
  
  if (config$debug.small) { # For debugging
    cat('NOTE: using a small subset of the data for debugging\n')
    dtrain = slice(dtrain, 1:2000)
  }
  
  set.seed(config$rng.seed)
  
  cat(date(), 'Starting CV run\n')
  
  xgb.cv.res = xgb.cv(
    params = config$xgb.params, 
    nrounds = config$xgb.params$nrounds,
    maximize = F,
    data = dtrain, 
    nfold = config$nr.folds,
    prediction = T,
    early.stop.round = config$xgb.early.stop.round, 
    verbose = T, 
    print.every.n = 100,
    nthread = ifelse(config$compute.backend != 'serial', 1, config$nr.threads)
  )
  
  best.logloss = min(xgb.cv.res$dt$test.logloss.mean)
  best.round = which.min(xgb.cv.res$dt$test.logloss.mean)
  cat(date(), 'Best logloss:', best.logloss, '@', best.round, '\n')
  
  if (0 & config$compute.backend != 'condor') {
    plot(1:nrow(xgb.cv.res$dt), xgb.cv.res$dt$train.logloss.mean, type = 'l', lty = 2, ylab = 'logloss', xlab = 'Boosting round', ylim = c(0.40, 0.50))
    lines(1:nrow(xgb.cv.res$dt), xgb.cv.res$dt$test.logloss.mean + 2 * xgb.cv.res$dt$test.logloss.std, lty = 3)
    lines(1:nrow(xgb.cv.res$dt), xgb.cv.res$dt$test.logloss.mean)
    lines(1:nrow(xgb.cv.res$dt), xgb.cv.res$dt$test.logloss.mean - 2 * xgb.cv.res$dt$test.logloss.std, lty = 3)
    abline(h = 0.46 , col = 2)
    abline(h = 0.455, col = 'orange')
    abline(h = 0.45 , col = 3)
  }
  
  return (list(best.logloss = best.logloss, best.round = best.round))
}

config$cross.validate = function(config) {
  for (i in 1:config$nr.folds) {
    cat(date(), 'Working on fold', i, '\n')
    config$in.fold = i
    config$train(config)
  }
  
  # Examine the overall CV fit  
  load(config$ancillary.filename)
  y = ancillary$train.labels
  for (i in 1:config$nr.folds) {
    load(paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', i, '.RData')) # => preds, w.train
    if (i == 1) {
      preds = as.matrix(preds)
      x = matrix(NA, length(ancillary$train.labels), ncol(preds))
    }
    x[ancillary$cv.folds[[i]], ] = preds
  }
  logloss = config$eval.logloss.core.multi(x, y, w.train)
  cat(date(), 'Final CV logloss:', logloss, '\n')
  cat(date(), 'Best final CV logloss:', min(logloss), 'at slot', which.min(logloss), '\n')
  
  cat(date(), 'Working on final model\n')
  config$in.fold = -1
  config$train(config)
  
  return (logloss)
}

config$cv.batch = function(config) {
  stopifnot(config$compute.backend != 'multicore')
  config$nr.cores = nrow(config$batch.model.specs)
  
  cv.job = function(config, core) {
    model.spec = config$batch.model.specs[core, ]
    cat('Working on model ', core, ':\n', sep = '')
    print(data.frame(value = unlist(model.spec)))
    cat('\n')
    
    config$model.tag                    = model.spec$model.tag
    #config$model.select                 = model.spec$model.select

    if (config$do.preprocess && config$batch.preprocess) {
      config$rng.seed                     = model.spec$rng.seed
      config$drop.big.categoricals        = model.spec$drop.big.categoricals
      config$add.manual.features          = model.spec$add.manual.features
      config$add.hexv.encoded.factors     = model.spec$add.hexv.encoded.factors
      config$count.special.values         = model.spec$count.special.values
      config$add.na.indicators            = model.spec$add.na.indicators
      config$add.summary.stats.per.row    = model.spec$add.summary.stats.per.row
      config$add.low.card.ints.as.factors = model.spec$add.low.card.ints.as.factors
      config$add.ccv.yenc.categoricals    = model.spec$add.ccv.yenc.categoricals
      config$add.freq.encoded.factors     = model.spec$add.freq.encoded.factors
      config$drop.low.marginal.assoc      = model.spec$drop.low.marginal.assoc
      config$weighting.scheme             = model.spec$weighting.scheme
      config$weighting.param              = model.spec$weighting.param
      
      # FIXME: how do I not run out of storage? will I have permissions for /tmp on Condor?
      config$dataset.filename   = tempfile('pp-data', fileext = '.RData')
      config$ancillary.filename = tempfile('pp-data-ancillary', fileext = '.RData')
      config$xgb.trainset.filename  = tempfile('xgb-trainset', fileext = '.data')
      config$xgb.testset.filename   = tempfile('xgb-testset', fileext = '.data')
      
      config$preprocess(config)
    }
    
    # config$xgb.params$reweight          = model.spec$xgb.reweight
    # config$xgb.params$nrounds           = model.spec$xgb.nrounds
    # config$xgb.params$objective         = model.spec$xgb.objective
    # config$xgb.params$eta               = model.spec$xgb.eta
    # config$xgb.params$max_depth         = model.spec$xgb.max_depth
    # config$xgb.params$min_child_weight  = model.spec$xgb.min_child_weight
    # config$xgb.params$subsample         = model.spec$xgb.subsample
    # config$xgb.params$colsample_bytree  = model.spec$xgb.colsample_bytree
    
    config$weighting.var   = model.spec$weighting.var
    config$weighting.level = model.spec$weighting.level
    
    if (config$batch.auto.xgb.nrounds) {
      cat(date(), 'Running a quick 2-CV to select nrounds\n')
      # FIXME but this may be a gross underestimate of the optimal nrounds for the full sample size
      # (2-CV uses half the sample!), and doing this for each model might introduce optimism in the
      # validation scores (and maybe actual overfitting as well in stacking).
      quick.cv.res = config$cross.validate.xgb(config, 2)
      config$xgb.params$nrounds = quick.cv.res$best.round
    }
    
    cv.res = config$cross.validate(config)
    
    if (config$do.preprocess && config$batch.preprocess) {
      file.remove(config$dataset.filename)
      file.remove(config$ancillary.filename)
      file.remove(config$xgb.trainset.filename)
      file.remove(config$xgb.testset.filename)
    }
    
    return (cbind(model.spec, logloss = cv.res))
  }
  
  if (config$do.preprocess && !config$batch.preprocess) {
    config$preprocess(config)
  }
  
  res = compute.backend.run(
    config, cv.job, combine = rbind, 
    package.dependencies = config$package.dependencies,
    source.dependencies  = config$source.dependencies,
    cluster.dependencies = config$cluster.dependencies,
    cluster.requirements = config$cluster.requirements,
    cluster.batch.name = 'dscience'
  )
  
  cat(date(), '\nBatch CV run summary:\n\n')
  res2 = res[order(res$logloss, decreasing = F), ]
  print(res2)
  cat('\n')
  
  save(res2, file = 'batch-cv-summary.RData')
}

# --------------------------------------------------------------------------------------------------
# Submission etc

config$generate.submission = function(config) {
  cat(date(), 'Generating submission\n')
  fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
  cat(date(), 'Loading preds from', fnm, '\n')
  load(fnm) # => preds
  load(config$ancillary.filename)
  
  if (is.matrix(preds)) {
    cat(date(), 'NOTE: submitting column', config$submt.column, '\n')
    preds = preds[, config$submt.column]
  }
  
  submission = data.frame(ID = ancillary$test.ids, PredictedProb = preds)
  write_csv(submission, paste0('sbmt-', config$submt.id, '.csv'))
  zip(paste0('sbmt-', config$submt.id, '.zip'), paste0('sbmt-', config$submt.id, '.csv'))
  
  if (config$compute.backend != 'condor') {
    ref.sbmt = data.table::fread(paste0('sbmt-', config$ref.submt.id, '.csv'), showProgress = F)
    names(ref.sbmt)[2] = 'refPredictedProb'
    cmpr.sbmt = merge(ref.sbmt, submission, by = 'ID')
    plot(cmpr.sbmt$refPredictedProb, cmpr.sbmt$PredictedProb, pch = '.', main = 'Sanity check', xlab = 'Ref pred', ylab = 'New pred')
    abline(0, 1, col = 2)
  }
}

config$backup.stack = function(config) {
  backup.dir = '../../../Junk'
  
  for (model.tag in config$stack.tags) {
    for (i in 1:config$nr.folds) {
      file.copy(paste0(config$tmp.dir, '/cv-preds-', model.tag, '-', i, '.RData'), paste0(backup.dir, '/cv-preds-', model.tag, '-', i, '.RData'))
    }
    file.copy(paste0(config$tmp.dir, '/test-preds-', model.tag, '.RData'), paste0(backup.dir, '/test-preds-', model.tag, '.RData'))
  }
}

config$share.stack = function(config) {
  cat(date(), 'Sharing meta-features from level', config$layer - 1, 'models\n')

  load(paste0(config$tmp.dir, '/pp-data-ancillary-L', config$layer - 1, '.RData')) # => ancillary
  
  train = NULL
  test = NULL
  
  select.best.tuning = T
  set.weight.zero.to.na = T

  for (model.tag in config$stack.tags) {
    cat(date(), 'Including model', model.tag)

    w.test = NULL # older models didn't have this
    load(paste0(config$tmp.dir, '/test-preds-', model.tag, '.RData')) # => preds, w.test
    model.preds = data.frame(preds)
    if (ncol(model.preds) == 1) {
      names(model.preds) = model.tag
      cat(' => 1 model')
    } else {
      names(model.preds) = paste(model.tag, 1:ncol(model.preds), sep = '_')
      cat(' =>', ncol(model.preds), 'models')
    }
    
    cv.preds = as.data.frame(matrix(NA, length(ancillary$train.labels), ncol(model.preds)))
    names(cv.preds) = names(model.preds)
    for (i in 1:config$nr.folds) {
      w.train = NULL # older models didn't have this
      load(paste0(config$tmp.dir, '/cv-preds-', model.tag, '-', i, '.RData')) # => preds, w.train
      cv.preds[ancillary$cv.folds[[i]], ] = preds
    }
    
    if (ncol(cv.preds) > 1 && select.best.tuning) {
      cvscores = rep(NA, ncol(cv.preds))
      for (i in 1:ncol(cv.preds)) {
        cvscores[i] = config$eval.logloss.core(cv.preds[[i]], ancillary$train.labels, w.train)
      }
      best.idx = which.min(cvscores)
      model.preds = model.preds[, best.idx, drop = F]
      cv.preds    = cv.preds   [, best.idx, drop = F]
      cat(' => selecting only column', best.idx, '\n')
    } else {
      cat('\n')
    }
    
    if (!is.null(w.test) & set.weight.zero.to.na) {
      model.preds[w.test  == 0, ] = NA
      cv.preds   [w.train == 0, ] = NA
    }
    
    if (is.null(test)) {
      test = model.preds
      train = cv.preds
    } else {
      test = cbind(test, model.preds)
      train = cbind(train, cv.preds)
    }
  }

  names(train) = gsub('team_', 'sk_', names(train))
  names(test ) = gsub('team_', 'sk_', names(test ))
  
  # NOTE: using Darragh's layer counting from 1 and not 0
  if (0) {
    fname = paste0('Team/SK-level', config$layer, '-meta-features2.RData')
    save(train, test, file = fname)
  } else {
    fname = paste0('Team/SK-level', config$layer, '-meta-features2.csv')
    dat = rbind(cbind(ID = ancillary$train.ids, train), cbind(ID = ancillary$test.ids, test))
    write_csv(dat, path = fname)
  }
  cat(date(), 'Stack saved to', fname, '\n')
}

# Do stuff
# ==================================================================================================

if (config$mode == 'single') {
  cat(date(), 'Starting single mode\n')
  
  if (config$do.preprocess) {
    config$preprocess(config)
  }
  
  if (config$do.xgb.cv) {
    config$cross.validate.xgb(config)
  }
  
  if (config$do.train) {
    ret = config$train(config)
  }
} else if (config$mode == 'cv') {
  cat(date(), 'Starting CV mode\n')
  
  if (config$do.preprocess) {
    config$preprocess(config)
  }
  
  if (config$do.train) {
    config$cross.validate(config)
  }
} else if (config$mode == 'cv.batch') {
  cat(date(), 'Batch CV mode\n')
  
  config$cv.batch(config)
}

if (config$do.submit) {
  config$generate.submission(config)
}

cat(date(), 'Done.\n')
