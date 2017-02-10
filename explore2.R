library (readr)
library (caret)
library (Hmisc) 
library (ComputeBackend)

source('../utils.r')

do.load.data     = F
do.yenc1         = F
do.pairwise      = F
do.interactions  = F
do.v22           = F
do.lmm1          = F
do.lmm2          = F
do.knn.darragh   = F
do.manual.labor  = F
do.interactions2 = F

set.seed(123)

if (do.load.data) {
  data.dir = '../../Data/Kaggle/bnp-paribas-cardif-claims-management'
  train = read_csv(paste0(data.dir, '/train.csv'))
  test  = read_csv(paste0(data.dir, '/test.csv'))

  train.labels = train$target
  train.ids = train$ID
  test.ids = test$ID
  dat = rbind(train[-which(names(train) == 'target')], test)
  train.idx = 1:nrow(train)
  test.idx = nrow(train) + (1:nrow(test))
  rm (train, test)
  
  # eliminate nearly/totally redundant features
  to.drop = c('v75', 'v110', 'v107', 'v63', 'v100', 'v17', 'v39')
  dat = dat[, !(names(dat) %in% to.drop)]

  # Chars are factors here  
  for (f in names(dat)[unlist(lapply(dat, is.character))]) {
    dat[[f]] = as.factor(dat[[f]])
  }
}

# Look into effect of number of folds on yenc
# --------------------------------------------------------------------------------------------------

# Let's see how the correlation with the response (out of sample) changes with the number of 
# folds. If it changes a lot, then it would be worth it to use more folds (as many as it helps and
# is feasible) when generating the final trainset (my "k = 0")

if (do.yenc1) {
  max.nr.numerical.levels = 200
  categoricals.max.cardinality = 30
  bayes.count = 200

  res = expand.grid(feature = c('v3', 'v66', 'v56', 'v22'), nr.folds = c(2, 5, 10), stringsAsFactors = F)
  res$cor = NA
  
  for (i in 1:nrow(res)) {
    f = res$feature[i]
    nr.folds = res$nr.folds[i]
    cat(date(), 'Feautre', f, 'folds', nr.folds, '\n')
    
    # FIXME do I have to repeat drawing the CV partition multiple times and average?
    cv.folds = createFolds(as.factor(train.labels), k = nr.folds)
    
    z = yenc.automatic(dat[train.idx, f], train.labels, cv.folds, 0, dat[test.idx, f], max.nr.numerical.levels, categoricals.max.cardinality, bayes.count)
    idx = !is.na(z)
    res$cor[i] = cor(z[idx], y = train.labels[idx], method = 'spearman')
  }
  
  print(res)
  # => I don't see a big difference (but that's not to say there isn't a small difference...)
}

# Another look at pairwise numerical interactions
# --------------------------------------------------------------------------------------------------

if (do.pairwise) {
  num.features = names(dat)[!unlist(lapply(dat, is.factor))][-1]

  # Ok so unfortunately there are too many pairs to consider. I have to either filter the features
  # I wanna look at, or do this automatically. In any case, I need to start somewhere.
  
  # Let's look at a random subset
  some.features = sample(num.features, 4)
  plot(dat[train.idx, some.features], col = as.factor(train.labels), pch = 15)
  # => honestly, I don't see anything interesting here now...
  
  dd = dat[train.idx, num.features]
  dd = as.data.frame(randomForest::na.roughfix(dd))
  dd$y = train.labels
  
  ccc = function(x) { idx = !is.na(x); cor(x[idx], dd$y[idx], method = 'spearman') }
  
  ccc(apply(dd[, c('v50', 'v12', 'v14', 'v114')], 1, prod))
  ccc(apply(dd[, c('v14', 'v114', 'v10')], 1, prod))
  ccc(apply(dd[, c('v14', 'v11', 'v4')], 1, prod))
  ccc(apply(dd[, c('v12', 'v11', 'v7')], 1, prod))
  ccc(apply(dd[, c('v40', 'v114', 'v34')], 1, prod))
  ccc(apply(dd[, c('v10', 'v14', 'v21', 'v114', 'v20')], 1, prod))
  ccc(apply(dd[, c('v34', 'v72', 'v21')], 1, prod))
  ccc(dd$v10 * dd$v12 * dd$v72 * dd$v6 / dd$v40)
}

# Explore interactions not via yencoding
# --------------------------------------------------------------------------------------------------

if (do.interactions) {
  num.features = names(dat)[!unlist(lapply(dat, is.factor))][-1]
  
  get.cor = function(x) {
    idx = !is.na(x)
    cor(x[idx], train.labels[idx], method = 'spearman')
  }

  cor1 = as.data.frame(mclapply(dat[train.idx, num.features], get.cor, mc.preschedule = F, mc.cores = 16))
  
  nn.pairs = as.data.frame(combn(num.features, 2), stringsAsFactors = F)

  get.cor.mul = function(pair) {
    x = dat[train.idx, pair[1]] * dat[train.idx, pair[2]]
    idx = !is.na(x)
    cor(x[idx], train.labels[idx], method = 'spearman')
  }

  get.cor.div = function(pair) {
    x = dat[train.idx, pair[1]] / dat[train.idx, pair[2]]
    idx = !is.na(x)
    cor(x[idx], train.labels[idx], method = 'spearman')
  }
  
  # well, this is redundant actually when looking at spearman corr
  get.cor.vid = function(pair) {
    x = dat[train.idx, pair[2]] / dat[train.idx, pair[1]]
    idx = !is.na(x)
    cor(x[idx], train.labels[idx], method = 'spearman')
  }
  
  cor2 = as.data.frame(t(nn.pairs), stringsAsFactors = F)
  names(cor2) = c('Var1', 'Var2')
  cor2$mul = unlist(mclapply(nn.pairs, get.cor.mul, mc.preschedule = F, mc.cores = 16))
  cor2$div = unlist(mclapply(nn.pairs, get.cor.div, mc.preschedule = F, mc.cores = 16))
  cor2$vid = unlist(mclapply(nn.pairs, get.cor.vid, mc.preschedule = F, mc.cores = 16))

  # Add marginal correlations so that we examine the added value of interactions
  marginal.cors = data.frame(Var = names(cor1), cor = as.vector(unlist(cor1[1, ])), stringsAsFactors = F)
  names(marginal.cors) = c('Var1', 'cor1')
  cor2 = merge(cor2, marginal.cors, by = 'Var1')
  names(marginal.cors) = c('Var2', 'cor2')
  cor2 = merge(cor2, marginal.cors, by = 'Var2')
  
  nn.pairs.to.include = cor2[complete.cases(cor2), ]
  nn.pairs.to.include.mul = nn.pairs.to.include[abs(nn.pairs.to.include$mul) - pmax(abs(nn.pairs.to.include$cor1), abs(nn.pairs.to.include$cor2)) > 0.02, ]
  nn.pairs.to.include.div = nn.pairs.to.include[abs(nn.pairs.to.include$div) - pmax(abs(nn.pairs.to.include$cor1), abs(nn.pairs.to.include$cor2)) > 0.02, ]
  
  save(nn.pairs.to.include, file = 'muldiv-interactions.RData')
}

# Explore v22, again
# --------------------------------------------------------------------------------------------------

if (do.v22) {
  x = dat$v22
  sum(is.na(x)) # => only about 1000 missing
  tbl = table(table(x))
  sum(tbl[1:25] * 1:25) # => half of the data is in levels with small support (< 25)
  d = sparse.model.matrix(~ x - 1) # => it is easy to create and store this huge sparse matrix, so models like XGB that support training directly on it can actually use this (but will have to deal with overfitting issues)

  tbl = sort(table(x[train.idx]), decreasing = T)
  levels.to.keep = names(tbl[tbl > 10])
  x.pruned = x
  x.pruned[!(x %in% levels.to.keep)] = NA
  x.pruned = factor(x.pruned)
  
  library(MatrixModels)
  glm.fit = glm4(train.labels ~ x.pruned[train.idx])
}

# Explore LMM/GLMM more
# --------------------------------------------------------------------------------------------------

if (do.lmm1) {
  # Look into pairwise LMMs 
  # Examine the case of v50 (fixed effect) and v22 (random effect)
  cv.folds = createFolds(as.factor(train.labels), k = 2)

  d.train = dat[train.idx[cv.folds$Fold1], ]
  d.test  = dat[train.idx[cv.folds$Fold2], ]
  d.train$y = train.labels[cv.folds$Fold1]
  d.test$y = train.labels[cv.folds$Fold2]

  yenc.xg = function(x.train, g.train, y.train, x.test, g.test) {
    dtr = data.frame(x = x.train, g = g.train, y = y.train)
    dte = data.frame(x = x.test, g = g.test)
    lmer.fit = glmer(y ~ 1 + x + (1 | g), dtr, family = binomial)
    glvls = rownames(ranef(lmer.fit)$g)
    g.blup = ranef(lmer.fit)$g[, 1]
    z = fixef(lmer.fit)[1] + fixef(lmer.fit)[2] * dte$x + g.blup[match(dte$g, glvls)]
    z = 1 / (1 + exp(-(z)))
  }

  yenc.xgg = function(x.train, g1.train, g2.train, y.train, x.test, g1.test, g2.test) {
    dtr = data.frame(x = x.train, g1 = g1.train, g2 = g2.train, y = y.train)
    dte = data.frame(x = x.test, g1 = g1.test, g2 = g2.test)
    lmer.fit = glmer(y ~ 1 + x + (1 | g1) + (1 | g2), dtr, family = binomial)
    g1.lvls = rownames(ranef(lmer.fit)$g1)
    g2.lvls = rownames(ranef(lmer.fit)$g2)
    g1.blup = ranef(lmer.fit)$g1[, 1]
    g2.blup = ranef(lmer.fit)$g2[, 1]
    z = fixef(lmer.fit)[1] + fixef(lmer.fit)[2] * dte$x + g1.blup[match(dte$g1, g1.lvls)] + g2.blup[match(dte$g2, g2.lvls)]
    z = 1 / (1 + exp(-(z)))
  }

  yenc.xggg = function(x.train, g1.train, g2.train, g3.train, y.train, x.test, g1.test, g2.test, g3.test) {
    dtr = data.frame(x = x.train, g1 = g1.train, g2 = g2.train, g3 = g3.train, y = y.train)
    dte = data.frame(x = x.test, g1 = g1.test, g2 = g2.test, g3 = g3.test)
    lmer.fit = glmer(y ~ 1 + x + (1 | g1) + (1 | g2) + (1 | g3), dtr, family = binomial)
    g1.lvls = rownames(ranef(lmer.fit)$g1)
    g2.lvls = rownames(ranef(lmer.fit)$g2)
    g3.lvls = rownames(ranef(lmer.fit)$g3)
    g1.blup = ranef(lmer.fit)$g1[, 1]
    g2.blup = ranef(lmer.fit)$g2[, 1]
    g3.blup = ranef(lmer.fit)$g3[, 1]
    z = fixef(lmer.fit)[1] + fixef(lmer.fit)[2] * dte$x + g1.blup[match(dte$g1, g1.lvls)] + g2.blup[match(dte$g2, g2.lvls)] + g2.blup[match(dte$g3, g3.lvls)]
    z = 1 / (1 + exp(-(z)))
  }
  
  idx = !is.na(d.test$v50)
  cor(d.test$v50[idx], y.test[idx], method = 'spearman')
  
  z.test = yenc.xg(d.train$v50, d.train$v66, d.train$y, d.test$v50, d.test$v66)
  idx = !is.na(z.test)
  cor(z.test[idx], y.test[idx], method = 'spearman')

  z.test = yenc.xgg(d.train$v50, d.train$v66, d.train$v56, d.train$y, d.test$v50, d.test$v66, d.test$v56)
  idx = !is.na(z.test)
  cor(z.test[idx], y.test[idx], method = 'spearman')

  # More doesn't seem to help
  z.test = yenc.xggg(d.train$v50, d.train$v22, d.train$v52, d.train$v91, d.train$y, d.test$v50, d.test$v22, d.test$v52, d.test$v91)
  idx = !is.na(z.test)
  cor(z.test[idx], y.test[idx], method = 'spearman')
  
  # So let's try more xg and xgg
  z.test = yenc.xg(d.train$v50, d.train$v79, d.train$y, d.test$v50, d.test$v79)
  idx = !is.na(z.test)
  cor(z.test[idx], y.test[idx], method = 'spearman')
}

if (do.lmm2) {
  # Create a simple situation to explore the leak-in-stacking problem
  # Idea: start with a permuted y and x = v22, then yenc x in L0, and use a tree model in L1 (any signal is a leak)
  library(xgboost)
  
  x = dat[train.idx, 'v56']
  #x = dat[train.idx, 'v22']
  #y = train.labels
  y = sample(train.labels)
  #y = sample(0:1, length(x), replace = T)

  nr.folds = 20
  cv.folds = createFolds(as.factor(y), k = nr.folds)
  
  z = yenc.glmm(x, y, cv.folds, 0)
  #z = yenc.bayes(x, y, cv.folds, 0)
  
  xgb.fit = xgboost(data = as.matrix(as.numeric(x)), label = y, missing = NA, nrounds = 100)
  xgb.fit = xgboost(data = as.matrix(as.numeric(z)), label = y, missing = NA, nrounds = 100)
  
  # => maybe LMM doesn't lead to these sort of leaks? I doubt it but worth trying
  
  # Check this in-sample:
  z = .yenc.glmm.core(x, y, x)
  xgb.fit = xgboost(data = as.matrix(as.numeric(x)), label = y, missing = NA, nrounds = 100)
  xgb.fit = xgboost(data = as.matrix(as.numeric(z)), label = y, missing = NA, nrounds = 100)
}

if (do.knn.darragh) {
  # Can we extract a reasonable number of CCV neighbors from the CV KNN files Darragh produced?
  knn_cv_dist_target_1_canberra.RData
}

if (do.manual.labor) {
  theta = 0.3454
  rotm = matrix(c(cos(theta), sin(theta), -sin(theta), cos(theta)), 2, 2)
  dd = data.matrix(dat[, c('v12', 'v50')]) %*% rotm
  plot(dd[train.idx, 1], dd[train.idx, 2], col = alpha(train.labels + 1, 0.5), pch = 16)
  dd[is.na(dd[, 1]), 1] = median(dd[, 1], na.rm = T)
  dd[is.na(dd[, 2]), 2] = median(dd[, 2], na.rm = T)
  cor(dd[train.idx, 1], train.labels)
  cor(dd[train.idx, 2], train.labels)
}

if (do.interactions2) {
  # Look at interactions using new types of L0 models
  dat.cats = dat[, unlist(lapply(dat, function(x) is.factor(x) || length(unique(x)) < 30))]
  dat.cats = as.data.frame(lapply(dat.cats, function(x) { xx = as.character(x); xx[is.na(xx)] = 'NA'; as.factor(xx) }))
  dat.cats$count.na = cut2(rowSums(is.na(dat)), g = 4)
  
  fold.idxs = read_csv('Team/index.csv')
  cv.folds = list()
  cv.folds$Fold1 = which(fold.idxs == 0)
  cv.folds$Fold2 = which(fold.idxs == 1)
  cv.folds$Fold3 = which(fold.idxs == 2)
  cv.folds$Fold4 = which(fold.idxs == 3)
  cv.folds$Fold5 = which(fold.idxs == 4)
  
  if (0) {
    # Look at Gilberto's VW top interactions
    vw3 = read_csv('Team/vw_feature_interaction_3way.csv')
    vw4 = read_csv('Team/vw_feature_interaction_4way.csv')
    vw5 = read_csv('Team/vw_feature_interaction_5way.csv')
    vw3 = vw3[rowSums(is.na(vw3)) == 0, ] # just to remove coefficients of lower order interactions
    vw4 = vw4[rowSums(is.na(vw4)) == 0, ] # ..
    vw5 = vw5[rowSums(is.na(vw5)) == 0, ] # ..
    vw3$Importance = abs(vw3$Importance)
    vw4$Importance = abs(vw4$Importance)
    vw5$Importance = abs(vw5$Importance)
    vw3 = vw3[order(vw3$Importance, decreasing = T), ]
    vw4 = vw4[order(vw4$Importance, decreasing = T), ]
    vw5 = vw5[order(vw5$Importance, decreasing = T), ]
    vw3$f1[vw3$f1 == 'rowSumsNA'] = 'count.na'
    vw4$f1[vw4$f1 == 'rowSumsNA'] = 'count.na'
    vw5$f1[vw5$f1 == 'rowSumsNA'] = 'count.na'
    # FIXME actually I need to replace these with their matching redundant features that I keep...
    to.drop = c('v75', 'v110', 'v107', 'v63', 'v100', 'v17', 'v39')
    vw3 = vw3[!((vw3$f1 %in% to.drop) | (vw3$f2 %in% to.drop) | (vw3$f3 %in% to.drop)), ]
    vw4 = vw4[!((vw4$f1 %in% to.drop) | (vw4$f2 %in% to.drop) | (vw4$f3 %in% to.drop) | (vw4$f4 %in% to.drop)), ]
    vw5 = vw5[!((vw5$f1 %in% to.drop) | (vw5$f2 %in% to.drop) | (vw5$f3 %in% to.drop) | (vw5$f4 %in% to.drop) | (vw5$f5 %in% to.drop)), ]
    vw3 = vw3[1:min(500, nrow(vw3)), ]
    vw4 = vw4[1:min(500, nrow(vw4)), ]
    vw5 = vw5[1:min(500, nrow(vw5)), ]
  } else {
    # Since this doesn't seem right, use my top interactions for now
    load('factor-3way-interactions.RData') # => cc.triplets
    cc.triplets.to.include = cc.triplets[complete.cases(cc.triplets), ]
    cc.triplets.to.include = cc.triplets.to.include[cc.triplets.to.include$cor123 - pmax(cc.triplets.to.include$cor1, cc.triplets.to.include$cor2, cc.triplets.to.include$cor3) > 0.01, ]
    cc.triplets.to.include = cc.triplets.to.include[order(cc.triplets.to.include$cor123, decreasing = T), ]
    vw3 = cc.triplets.to.include
    names(vw3)[1:3] = c('f1', 'f2', 'f3')
    to.drop = c('v75', 'v110', 'v107', 'v63', 'v100', 'v17', 'v39')
    vw3 = vw3[!((vw3$f1 %in% to.drop) | (vw3$f2 %in% to.drop) | (vw3$f3 %in% to.drop)), ]
  }
  
  #
  # Fit small XGBs on the tuples
  #
  
  library(xgboost)

  # FIXME I probably want to tune this per tuple (with nested CV...)
  xgb.params = list(
    objective         = 'binary:logistic',
    eval_metric       = 'logloss',
    nrounds           = 100
    #eta               = 0.03,
    #max_depth         = 5,
    #subsample         = 0.9,
    #colsample_bytree  = 0.7
  )
  
  vw3$va.logloss = NA
  for (i in 1:nrow(vw3)) {
    cat(date(), 'Working on tuple', i, 'of', nrow(vw3), '\n')
    
    vw3$va.logloss[i] = xgb.cv(
        params = xgb.params, folds = cv.folds, verbose = F, nrounds = xgb.params$nrounds, nthread = 8,
        data = xgb.DMatrix(sparse.model.matrix(~ . - 1, dat.cats[train.idx, c(vw3$f1[i], vw3$f2[i], vw3$f3[i])]), label = train.labels)
      )$test.logloss.mean[xgb.params$nrounds]
  }
  
  # => This shows now corr with the VW importance. Either the latter is useless/buggy, or fitting
  # a blind (untuned) XGB is useless...
  
  #
  # Let's try an LMM
  #

  dat.to.encode = matrix(NA, length(train.labels), nrow(vw3))
  for (i in 1:nrow(vw3)) {
    dat.to.encode[, i] = interaction(dat.cats[train.idx, vw3$f1[i]], dat.cats[train.idx, vw3$f2[i]], dat.cats[train.idx, vw3$f3[i]], drop = T)
  }
  
  config = list()
  config$dat.to.encode = dat.to.encode[, 1:16]
  config$train.labels = train.labels
  config$cv.folds = cv.folds
  config$compute.backend = 'multicore'
  config$nr.cores = 7
  
  procfun = function(config, core) {
    cols.core = ComputeBackend::compute.backend.balance(ncol(config$dat.to.encode), config$nr.cores, core)
    nr.cols.core = length(cols.core)
    if (nr.cols.core < 1) return (NULL)
    res.core = rep(NA, nr.cols.core)
    for (i in 1:nr.cols.core) {
      cat(date(), 'Working on tuple', i, 'of', nr.cols.core, '\n')
      z = yenc.glmm(config$dat.to.encode[, cols.core[i]], config$train.labels, config$cv.folds, 0)
      idx = !is.na(z)
      res.core[i] = cor(z[idx], config$train.labels[idx], method = 'spearman')
    }
    return (res.core)
  }
      
  vw3$va.corr[1:16] = compute.backend.run(config, procfun, source.dependencies = '../utils.r', combine = c)
}

cat(date(), 'Done.\n')