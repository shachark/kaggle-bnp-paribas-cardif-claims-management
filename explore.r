library (readr)
library (caret)
library (Hmisc) 
library (ComputeBackend)

source('../utils.r')

data.dir = '../../Data/Kaggle/bnp-paribas-cardif-claims-management'
train = read_csv(paste0(data.dir, '/train.csv'))
test  = read_csv(paste0(data.dir, '/test.csv'))

# Look at the properties of the features
ftrs = data.frame(
  type      = unlist(lapply(train, class)),
  n.unique  = unlist(lapply(train, function(x) length(unique(x)))),
  f.missing = unlist(lapply(train, function(x) mean(is.na(x)))),
  spear.cor  = unlist(lapply(train, function(x) { idx = !is.na(x); if (is.character(x)) x = as.numeric(as.factor(x)); cor(x[idx], y = train$target[idx], method = 'spearman') }))
)

# First, let's eliminate nearly/totally redundant features (found through exploration below)
to.drop = c('v75', 'v110', 'v107', 'v63', 'v100', 'v17', 'v39')
train = train[, !(names(train) %in% duplicated.x)]
test  = test [, !(names(test ) %in% duplicated.x)]

# --------------------------------------------------------------------------------------------------

# Make sure our beloved NA placeholder is not used in the data
sum(train <= -999, na.rm = T)
sum(test <= -999, na.rm = T)

# What's up with those "numeric" features that take only a small number of unique values?? The
# description says that there aren't any ordinals. Should I treat these as categoricals?
xx = table(train$v62, train$target)
xx / rowSums(xx)

# Some categorical features have too many categories:
# v22, v56, v125, v113
# Let's take one of these as an example: v125
x = as.factor(train$v125)
table(x)
# One way to retain them in the model is to simply treat them as numeric, as if the order of the 
# levels is not arbitrary.
boxplot(as.numeric(x) ~ train$target) 
# This seems pretty useless, and since the anonymization work here seems solid, I'm not surprised.
# Another way is to drop or merge some of the less interesting categories.
dummyfied.x = as.data.frame(predict(dummyVars(~ x, data.frame(x = x)), data.frame(x = x)))
tmp.df = data.frame(level = levels(x), pv = p.adjust(unlist(lapply(dummyfied.x, function(x) cor.test(x, y = train$target, method = 'pearson')$p.value)), method = 'BH'))
tmp.df = tmp.df[order(tmp.df$pv), ]
levels.to.keep = tmp.df$level[tmp.df$pv < 0.1]
# Then do: dat$v125[!(dat$v125 %in% levels.to.keep)] = NA
# Yet another way is to stack a penalized saturated marginal model
# (I don't have enough memory to try this. Could subsample...)
#idx = !is.na(dummyfied.x)
#glm.fit = glmnet(data.matrix(dummyfied.x[idx, ]), train$target[idx], 'binomial', dfmax = 30)

# What's up with the many NAs?
na.ftrs = rownames(ftrs)[ftrs$f.missing > 0.4]
summaryRc(as.formula(paste('target ~ ', paste(na.ftrs[1:4], collapse = '+'), collapse = '')), data = train)
plot(train[, na.ftrs[1:4]], pch = '.')

# Maybe this will be important within deep trees (it is not marginally associated with target)
na.count = rowSums(is.na(train))
na.count.group = cut(na.count, c(-1, 24, 80, 99, Inf), labels = 1:4)
na.count.resid = rowSums(is.na(train[, -which(names(train) %in% na.ftrs)]))

# --------------------------------------------------------------------------------------------------
#
# Look at feature correlations
#

library (corrplot)

dat = rbind(train[, -which(names(train) %in% c('ID', 'target'))], test[, -which(names(test) == 'ID')])
for (f in names(dat)) {
  if (class(dat[[f]]) == 'character') {
    dat[[f]] = as.integer(as.factor(dat[[f]]))
  }
}

#
# Start with categoricals
#

cat.features = rownames(ftrs)[ftrs$type == 'character' & ftrs$n.unique < 30]
cat.pairs = as.data.frame(t(combn(cat.features, 2)), stringsAsFactors = F)
cat.pairs$pv = NA
for (i in 1:nrow(cat.pairs)) {
  cat(date(), 'working on', i, 'out of', nrow(cat.pairs), '\n')
  f1 = cat.pairs$V1[i]
  f2 = cat.pairs$V2[i]
  x1 = train[, f1]
  x2 = train[, f2]
  idx = !is.na(x1) & !is.na(x2)
  tbl = table(x1[idx], x2[idx])
  cat.pairs$pv[i] = chisq.test(tbl)$p.value
  
  if (cat.pairs$pv[i] < 1e-100) {
    cat('Features', f1, 'and', f2, '\n')
    print(tbl)
  }
}

# => 
# Can surely remove: v75, v91/v107
# Probably safe to also remove: v3, v47, v110
# Can merge: (v31, v110), (v79, v110)

#
# Now for numericals
#

num.features = rownames(ftrs)[ftrs$type != 'character'][-(1:2)]
duplicated.x = findCorrelation(cor(randomForest::na.roughfix(train[, num.features])), cutoff = .99, names = T, verbose = T)
# (I've plotted these pairs with resposnse color coding, and found no signal, so probably safe to drop)

d = data.matrix(dat)
d.vars = apply(d, 2, var, na.rm = T)
which(d.vars == 0) 
d = d[, d.vars != 0]
d = cor(d, method = 'spearman', use = 'complete.obs') # TODO use more modern measures like distance correlation, HHG, etc
which(is.na(d))
d[is.na(d)] = 0

# Get a global sense of the covariate structure (go through PDF so that I can zoom interactively)
pdf('corrplot.pdf', 50, 50)
corrplot(d, order = 'hclust', type = 'lower', diag = F, tl.cex = 0.6)
dev.off()

# Sorta linear relationships
plot(train$v10, train$v12, pch = '.', col = train$target + 1, xlim = c(0, 5))
cor(train$v10, train$target, use = 'complete.obs')
cor(train$v12, train$target, use = 'complete.obs')
summary(lm(v12 ~ v10, dat))
cor(train$v12 - 0.6036 * train$v10, train$target, use = 'complete.obs') # maybe useful
plot(train$v39, train$v68, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
cor(train$v39, train$target, use = 'complete.obs')
cor(train$v68, train$target, use = 'complete.obs')
summary(lm(v39 ~ v68, dat))
cor(train$v39 + 0.95 * train$v68, train$target, use = 'complete.obs') # doesn't seem to help
plot(train$v58, train$v100, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
cor(train$v58, train$target, use = 'complete.obs')
cor(train$v100, train$target, use = 'complete.obs')
cor(train$v58 - train$v100, train$target, use = 'complete.obs') # => sto drop one of these?
plot(train$v11, train$v53, pch = '.', col = train$target + 1, xlim = c(13, 18), ylim = c(14, 18))
cor(train$v11, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v53, train$target, method = 'spearman', use = 'complete.obs')
summary(lm(v11 ~ v53, dat))
cor(train$v11 - 0.95 * train$v53, train$target, method = 'spearman', use = 'complete.obs') # doesn't seem promising
plot(train$v27, train$v98, pch = '.', col = train$target + 1, xlim = c(0, 7), ylim = c(0, 20))
cor(train$v27, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v98, train$target, method = 'spearman', use = 'complete.obs')
summary(lm(v27 ~ v98, dat))
cor(train$v27 - 0.25 * train$v98, train$target, method = 'spearman', use = 'complete.obs') # doesn't seem promising
plot(train$v27, train$v92, pch = '.', col = train$target + 1, xlim = c(0, 7), ylim = c(0, 2))
cor(train$v27, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v92, train$target, method = 'spearman', use = 'complete.obs')
summary(lm(v27 ~ v92, dat))
cor(train$v27 + 2 * train$v98, train$target, method = 'spearman', use = 'complete.obs') # doesn't seem promising
plot(train$v64, train$v106, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v17, train$v106, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v76, train$v106, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v48, train$v106, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
cor(train$v106, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v64, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v17, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v76, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v48, train$target, method = 'spearman', use = 'complete.obs')
summary(lm(v106 ~ v48, dat))
cor(train$v106 + 1.32 * train$v48, train$target, method = 'spearman', use = 'complete.obs') # doesn't seem promising
plot(train$v26, train$v60, pch = '.', col = train$target + 1, xlim = c(0, 5), ylim = c(0, 5))
cor(train$v26, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v60, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v26 - train$v60, train$target, method = 'spearman', use = 'complete.obs') # doesn't seem promising
plot(train$v43, train$v116, pch = '.', col = train$target + 1, xlim = c(0, 5), ylim = c(0, 5))
cor(train$v43, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v116, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v43 - train$v116, train$target, method = 'spearman', use = 'complete.obs')
plot(train$v34, train$v40, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
cor(train$v34, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v40, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v40 + 1.6 * train$v34, train$target, method = 'spearman', use = 'complete.obs') # ! though probably also like v50
plot(train$v1, train$v37, pch = '.', col = train$target + 1, xlim = c(0, 5), ylim = c(0, 3))
summary(lm(v1 ~ v37, dat))
cor(train$v1, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v37, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v1 - 1.8 * train$v37, train$target, method = 'spearman', use = 'complete.obs') # maybe
plot(train$v97, train$v118, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
cor(train$v97, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v118, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v97 - train$v118, train$target, method = 'spearman', use = 'complete.obs') # maybe
plot(train$v92, train$v95, pch = '.', col = train$target + 1, xlim = c(0, 1), ylim = c(0, 1))
cor(train$v92, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v95, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v92 - train$v95, train$target, method = 'spearman', use = 'complete.obs') # maybe

# Nonlinear interesting relationships
plot(train$v114, train$v40, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
cor(train$v114, train$target, method = 'spearman', use = 'complete.obs')
cor(train$v40, train$target, method = 'spearman', use = 'complete.obs')
summary(lm(I(v114 ^ 2) ~ v40, dat))
plot(train$v114 ^ 2, train$v40, pch = '.', col = train$target + 1)
cor(train$v114 ^ 2 + 21 * train$v40, train$target, method = 'spearman', use = 'complete.obs') # ! (though this looks like v50)
cor(train$v114 ^ 2 + 21 * train$v40, train$v50, method = 'spearman', use = 'complete.obs') # ... so prabably not gonna help

# Maybe drop one
plot(train$v63, train$v82, pch = '.', col = train$target + 1, xlim = c(0, 3), ylim = c(0, 20))
plot(train$v69, train$v115, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
table(train$v71, train$v75)

# Can't make much sense out of these
plot(train$v39, train$v88, pch = '.', col = train$target + 1, xlim = c(0, 2), ylim = c(0, 5))
plot(train$v4, train$v88, pch = '.', col = train$target + 1, xlim = c(0, 10), ylim = c(0, 5))
plot(train$v61, train$v100, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
table(train$v62, train$v72)
plot(train$v111, train$v33, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v111, train$v83, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v83, train$v121, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v121, train$v33, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v55, train$v33, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v81, train$v5, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v81, train$v128, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v86, train$v32, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v32, train$v15, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v86, train$v73, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v80, train$v64, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v18, train$v28, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v8, train$v105, pch = '.', col = train$target + 1, xlim = c(0, 0.3), ylim = c(0, 0.3))
plot(train$v109, train$v128, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v87, train$v128, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))
plot(train$v96, train$v29, pch = '.', col = train$target + 1, xlim = c(0, 20), ylim = c(0, 20))

if (0) {
  # Look into non-monotonic measures of correlation
  # NOTE: this is prohibitive for the full data, so I gotta subsample
  library(energy)
  dat = train[complete.cases(train), ]
  dcors = unlist(lapply(dat, function(x) { idx = !is.na(x); if (is.character(x)) x = as.numeric(as.factor(x)); dcor(x[idx], y = train$target[idx]) }))
}

# --------------------------------------------------------------------------------------------------
#
# Explore methods of dimentionality reduction
#

#
# The data to reduce
#

# It makes zero sense to me to use the categoricals here (FIXME maybe do OWE?)
dr.features = rownames(ftrs)[ftrs$type != 'character'][-(1:2)]

# Filter out non-normal looking features??
#dr.features = setdiff(dr.features, paste0('v', c(8, 19, 23, 25, 38, 39, 46, 54, 58, 62, 63, 72, 82, 89, 100, 105, 119, 124, 129))))

# Get rid of features with many NAs since imputing them is absurd??
#dr.features = setdiff(dr.features, rownames(ftrs)[ftrs$f.missing > 0.1])

dr.dat = rbind(train[, dr.features], test[, dr.features])

# Only do this for reasonably complete cases
#dr.idx = (rowMeans(is.na(dr.dat)) < 0.1)
#dr.dat = dr.dat[dr.idx, ]

# Impute (for lack of a better idea)
dr.dat = randomForest::na.roughfix(dr.dat)

# Scale (important for some DR algos)
dr.dat.scaled = scale(dr.dat)

#
# K-means
#

# Base R
kmns = kmeans(dr.dat, 16)
cnts = kmns$centers
kmeans.centroid.distances = NULL
for (i in 1:nrow(cnts)) {
  kmeans.centroid.distances = cbind(kmeans.centroid.distances, sqrt(colSums((t(dr.dat) - unlist(cnts[i, ])) ^ 2)))
}
table(kmns$cluster)
plot(table(cut2(kmeans.centroid.distances[1:nrow(train), 1], g = 20), train$target))
plot(table(cut2(kmeans.centroid.distances[1:nrow(train), 2], g = 20), train$target))
plot(table(cut2(kmeans.centroid.distances[1:nrow(train), 3], g = 20), train$target))
plot(table(cut2(kmeans.centroid.distances[1:nrow(train), 4], g = 20), train$target))

# H2O
library (h2o)
h2o.init(nthreads = -1, max_mem_size = '16G')
dr.dat.h2o = as.h2o(dr.dat.scaled)

kmns = h2o.kmeans(dr.dat.h2o, k = 8, nfolds = 5)
kmeans.labels = as.data.frame(h2o.predict(kmns, dr.dat.h2o))

sort(table(kmeans.labels), decreasing = T)
tbl = table(kmeans.labels[1:nrow(train), 1], train$target)
cbind(tbl, ratio = tbl[, 1] / (tbl[, 1] + tbl[, 2]))
chisq.test(tbl[tbl[, 1] > 100])

save(kmeans.labels, file = 'kmeans-labels.RData')

#
# PCA
#

prcomp.res = prcomp(dr.dat, center = T, scale. = T)
plot(prcomp.res, type = 'l', log = 'y', npcs = 80) # => take a few?

pc.cor  = unlist(apply(prcomp.res$x[1:nrow(train), ], 2, cor, y = train$target, method = 'spearman'))
View(data.frame(abs(pc.cor)))
# => do we want the top PCs? or the top correlated PCs?

# One way to look at it:
plot(prcomp.res$x[1:nrow(train), 21], prcomp.res$x[1:nrow(train), 7], col = train$target + 1, pch = 19, cex = 0.3)
plot(prcomp.res$x[1:nrow(train), 21], prcomp.res$x[1:nrow(train), 1], col = train$target + 1, pch = 19, cex = 0.3)
plot(prcomp.res$x[1:nrow(train), 1], prcomp.res$x[1:nrow(train), 2], col = train$target + 1, pch = 19, cex = 0.3)
plot(prcomp.res$x[1:nrow(train), 1], prcomp.res$x[1:nrow(train), 3], col = train$target + 1, pch = 19, cex = 0.3)
plot(prcomp.res$x[1:nrow(train), 1], prcomp.res$x[1:nrow(train), 4], col = train$target + 1, pch = 19, cex = 0.3)
plot(prcomp.res$x[1:nrow(train), 1], prcomp.res$x[1:nrow(train), 5], col = train$target + 1, pch = 19, cex = 0.3)
plot(prcomp.res$x[1:nrow(train), 2], prcomp.res$x[1:nrow(train), 3], col = train$target + 1, pch = 19, cex = 0.3)
plot(prcomp.res$x[1:nrow(train), 2], prcomp.res$x[1:nrow(train), 4], col = train$target + 1, pch = 19, cex = 0.3)
plot(prcomp.res$x[1:nrow(train), 2], prcomp.res$x[1:nrow(train), 5], col = train$target + 1, pch = 19, cex = 0.3)
plot(prcomp.res$x[1:nrow(train), 3], prcomp.res$x[1:nrow(train), 4], col = train$target + 1, pch = 19, cex = 0.3)
plot(prcomp.res$x[1:nrow(train), 3], prcomp.res$x[1:nrow(train), 5], col = train$target + 1, pch = 19, cex = 0.3)
plot(prcomp.res$x[1:nrow(train), 4], prcomp.res$x[1:nrow(train), 5], col = train$target + 1, pch = 19, cex = 0.3)

# Second way:
plot(table(cut2(prcomp.res$x[1:nrow(train), 1], g = 20), train$target))
plot(table(cut2(prcomp.res$x[1:nrow(train), 2], g = 20), train$target))
plot(table(cut2(prcomp.res$x[1:nrow(train), 3], g = 20), train$target))
plot(table(cut2(prcomp.res$x[1:nrow(train), 4], g = 20), train$target))
plot(table(cut2(prcomp.res$x[1:nrow(train), 5], g = 20), train$target))
plot(table(cut2(prcomp.res$x[1:nrow(train), 6], g = 20), train$target))
plot(table(cut2(prcomp.res$x[1:nrow(train), 7], g = 20), train$target))

# Third way:
# That hmisc function, i don't recall it's name now

#
# TSNE
#

# TODO: rerun this, it will take hours
library (Rtsne)
set.seed(1234)

# (use dr.dat from above)

tsne.res = Rtsne(dr.dat, check_duplicates = T, pca = T, max_iter = 700, perplexity = 30, theta = 0.5, dims = 2, verbose = T)

save(tsne.res, file = 'tsne.RData')

plot(tsne.res$Y[1:10000, 1], tsne.res$Y[1:10000, 2], col = train$target[1:10000] + 1, pch = 19, cex = 0.3)
plot(table(cut2(tsne.res$Y[1:nrow(train), 1], g = 20), train$target))
plot(table(cut2(tsne.res$Y[1:nrow(train), 2], g = 20), train$target))

# => this really doesn't seem promising. I also don't know how to deal with the massive amount of
# missing data. I guess I can't do better than generating low-dimensional representation only for 
# the (almost/) complete cases.

#
# Autoencoder
#

# This didn't work, dunno why. Using dr.dat from above
if (0) {
  dr.dat = rbind(train[, -(1:2)], test[, -1])
  
  # apparently h2o's autoencoder supports factors! (but let's not overdo it)
  dr.dat = dr.dat[, !(names(dr.dat) %in% c('v22', 'v56', 'v125'))]
  for (f in names(dr.dat)[unlist(lapply(dr.dat, is.character))]) {
    dr.dat[[f]] = as.factor(dr.dat[[f]])
  }
  dr.dat = randomForest::na.roughfix(dr.dat)
  for (f in names(dr.dat)[unlist(lapply(dr.dat, is.numeric))]) {
    dr.dat[[f]] = scale(dr.dat[[f]])
  }
  for (f in names(dr.dat)[unlist(lapply(dr.dat, is.integer))]) {
    dr.dat[[f]] = scale(dr.dat[[f]])
  }
  
  # do we need this?
  #dr.dat = randomForest::na.roughfix(dr.dat)
  
  if (0) { # DEBUG
    dr.dat.h2o = as.h2o(dr.dat[1:1000, ])
  } else {
    dr.dat.h2o = as.h2o(dr.dat)
  }
}

m.aec = h2o.deeplearning(
  x = names(dr.dat),
  training_frame = dr.dat.h2o,
  autoencoder = T,
  activation = "Tanh",
  hidden = c(4096, 256, 16, 2, 16, 256, 4096), # FIXME how to select these?
  #hidden = c(256, 16, 4, 2, 4, 16, 256),
  epochs = 150 # FIXME how to select these?
  # FIXME there is a mindboggling number of parameters to this thing... I have no idea which are important
)

deep.fea = as.data.frame(h2o.deepfeatures(m.aec, dr.dat.h2o, layer = 4))
save(deep.fea, file = 'autoencoder-output.RData')

# Plot what we get, for the trainset, against the true targets. Does it look informative?
deep.fea.tr = deep.fea[1:nrow(train), ]
z = (train$target == 0)
plot  (deep.fea.tr[!z, 1], deep.fea.tr[!z, 2], pch = 19, cex = 0.2, col = "green", xlab = "DF1", ylab = "DF2")
points(deep.fea.tr[ z, 1], deep.fea.tr[ z, 2], pch = 19, cex = 0.3, col = "blue")
legend('bottomleft', legend = c(0, 1), fill = c('blue', 'green'))

# --------------------------------------------------------------------------------------------------
#
# Explore nearest neighbors
#

# With RANN:

library(RANN)
dat = train[, -(1:2)]
cat.features = rownames(ftrs)[ftrs$type == 'character']
dat[, cat.features] = lapply(dat[, cat.features], as.factor)
dat = randomForest::na.roughfix(dat)
dat[, cat.features] = lapply(dat[, cat.features], as.integer) # that's what it does anyway

sds = unlist(lapply(dat, sd))
hist(log(sds), 30)
View(data.frame(sds)) # => the high cardinality factors dominate the variance

# => Normalize the features, and/or maybe remove the categoricals
dat = scale(dat)

# Let's see what's feasible
system.time(nn2(dat[1:1000, ])) # < 1sec
system.time(nn2(dat[1:5000, ])) # 4 sec
system.time(nn2(dat[1:10000, ])) # 14 sec
system.time(nn2(dat[1:20000, ])) # 45 sec
#system.time(nn2(dat[1:50000, ])) # it will be a while...
#system.time(nn2(dat)) # it might never end

# Let's see if we can approximate
nn2.res.eps0 = nn2(dat[1:20000], eps = 0)
nn2.res.eps1 = nn2(dat[1:20000], eps = 1) # runs in less than a sec
nn2.res.eps5 = nn2(dat[1:20000], eps = 5) # runs in less than a sec
summary(nn2.res.eps1$nn.dists - nn2.res.eps0$nn.dists) # => it's pretty damn accurate
summary(nn2.res.eps5$nn.dists - nn2.res.eps0$nn.dists) # => it's pretty damn accurate too

# How long doe this take?
system.time(nn2(dat[1:50000, ], eps = 1)) # 105 sec
system.time(nn2(dat[1:50000, ], eps = 5)) # 13 sec
system.time(nn2(dat, eps = 5)) # 43 sec

k = 200
nn2.res = nn2(dat, k = k + 1, eps = 5) # since it always finds the same point as the nearest neighbor

#plot(nn2.res$nn.dists[, 2], nn2.res$nn.dists[, 3], pch = '.')

knn.preds = matrix(train$target[nn2.res$nn.idx[, 2:(k + 1)]], ncol = k)

lls = rep(NA, k)
for (i in 1:k) {
  knn.p = rowMeans(knn.preds[, 1:i, drop = F])
  pp = pmax(pmin(knn.p, 1 - 1e-15), 1e-15)
  lls[i] = -mean(log(ifelse(train$target == 0, 1 - pp, pp)))
}
plot(lls, type = 'l', main = 'kNN error', ylab = 'logloss', xlab = 'k', ylim = c(0.4, 0.6))
abline(h = 0.45, col = 2, lty = 2)

# With FNN (has an interesting classification algorithm, but does not supports approximate search, although its based on the same core implementation...)

library(FNN)
# using dat from above code
dat[, cat.features] = lapply(dat[, cat.features], as.integer)
# simple NN
fnn.dists = knnx.dist(dat, dat, k + 1)

# Is this the same as nn2? (should be, since NN is NN, and they even use the same core implementation)
sum(nn2.res$nn.dists != fnn.dists) # => yepp. In particular, this means that RANN simply casts factors to integers

# OWNN: this is a problem computatinoally since no approximations are available
n.try = 10000
k = 250 # it can pick it using CV, but since the best k is pretty high, this takes a lot of time (eventually even will run out of RAM)
ownn.res = ownn(dat[1:n.try, ], dat[n.try + (1:n.try), ], cl = train$target[1:n.try], prob = T, k = k)

ownn.res$k
#preds = ifelse(ownn.res$knnpred == 1, attr(ownn.res$knnpred, 'prob'), 1 - attr(ownn.res$knnpred, 'prob'))
#preds = ifelse(ownn.res$bnnpred == 1, attr(ownn.res$bnnpred, 'prob'), 1 - attr(ownn.res$bnnpred, 'prob'))
preds = ifelse(ownn.res$ownnpred == 1, attr(ownn.res$ownnpred, 'prob'), 1 - attr(ownn.res$ownnpred, 'prob'))
preds = pmax(pmin(preds, 1 - 1e-15), 1e-15)
-mean(log(ifelse(train$target[n.try + (1:n.try)] == 0, 1 - preds, preds)))

#---------------------------------------------------------------------------------------------------
#
# Quantize numericals and SMS encode them
#

# TODO try to optimize the number of levels and which features to encode
# FIXME what about the dim reduce features?

num.features = rownames(ftrs)[ftrs$type != 'character'][-(1:2)]
qnums = data.frame(X = num.features, stringsAsFactors = F)
qnums$corx = NA
qnums$corz = NA
qnr = 30
set.seed(123)
cv.folds = createFolds(train$target, k = 3)
y = train$target[cv.folds$Fold1]
for (i in seq_along(num.features)) {
  cat(date(), '---', i, 'of', length(num.features), '\n')
  x = train[[num.features[i]]]
  z = cut2(x, g = min(qnr, length(unique(x))))
  z = yenc.random.intercept(z, train$target, cv.folds, 1)
  x = x[cv.folds$Fold1]
  z = z[cv.folds$Fold1]
  qnums$corx[i] = cor(x[!is.na(x)], y[!is.na(x)])
  qnums$corz[i] = cor(z[!is.na(z)], y[!is.na(z)])
}
View(qnums)
save(qnums, file = 'quantize-numericals.RData')

#---------------------------------------------------------------------------------------------------
#
# Run association tests for interactions (pairwise, more if I can) and select the top ones
#

#
# Between categoricals
#

if (0) {
  cat.features = rownames(ftrs)[ftrs$type == 'character' & ftrs$n.unique < 30]
  cat.pairs = data.frame(t(combn(cat.features, 2)), stringsAsFactors = F)
  cat.pairs$pv = NA
  for (i in 1:nrow(cat.pairs)) {
    frmla.null = as.formula(paste('target ~', cat.pairs$X1[i], '+', cat.pairs$X2[i]))
    frmla.alte = as.formula(paste('target ~', cat.pairs$X1[i], '*', cat.pairs$X2[i]))
    cat.pairs$pv[i] = anova(glm(frmla.null, train, family = 'binomial'), glm(frmla.alte, train, family = 'binomial'), test = 'Chisq')$`Pr(>Chi)`[2]
  }
} else {
  # Take II: allow high cardinality variables through LMM, guage usefulness out of sample
  cat.features = rownames(ftrs)[ftrs$type == 'character']
  cat.pairs = data.frame(t(combn(cat.features, 2)), stringsAsFactors = F)
  cat.pairs$cor1 = NA
  cat.pairs$cor2 = NA
  cat.pairs$cor12 = NA
  set.seed(123)
  cv.folds = createFolds(train$target, k = 3)
  y = train$target[cv.folds$Fold1]
  for (i in 1:nrow(cat.pairs)) {
    cat(date(), '---', i, 'of', nrow(cat.pairs), '\n')
    x1 = as.factor(train[[cat.pairs$X1[i]]])
    x2 = as.factor(train[[cat.pairs$X2[i]]])
    x12 = interaction(x1, x2)
    z1  = yenc.random.intercept(x1 , train$target, cv.folds, 1)[cv.folds$Fold1]
    z2  = yenc.random.intercept(x2 , train$target, cv.folds, 1)[cv.folds$Fold1]
    z12 = yenc.random.intercept(x12, train$target, cv.folds, 1)[cv.folds$Fold1]
    cat.pairs$cor1 [i] = cor(z1 [!is.na(z1 )], y[!is.na(z1 )]) 
    cat.pairs$cor2 [i] = cor(z2 [!is.na(z2 )], y[!is.na(z2 )]) 
    cat.pairs$cor12[i] = cor(z12[!is.na(z12)], y[!is.na(z12)])
  }
}
View(cat.pairs)
save(cat.pairs, file = 'pairwise-factor-interactions.RData')

# Three-way
cat.features = rownames(ftrs)[ftrs$type == 'character']
cat.triplets = data.frame(t(combn(cat.features, 3)), stringsAsFactors = F)
cat.triplets$cor123 = NA
set.seed(123)
cv.folds = createFolds(train$target, k = 3)
y = train$target[cv.folds$Fold1]
for (i in 1:nrow(cat.triplets)) {
  cat(date(), '---', i, 'of', nrow(cat.triplets), '\n')
  x1 = as.factor(train[[cat.triplets$X1[i]]])
  x2 = as.factor(train[[cat.triplets$X2[i]]])
  x3 = as.factor(train[[cat.triplets$X3[i]]])
  x123 = interaction(x1, x2, x3, drop = T)
  z123 = yenc.random.intercept(x123, train$target, cv.folds, 1)[cv.folds$Fold1]
  cat.triplets$cor123[i] = cor(z123[!is.na(z123)], y[!is.na(z123)])
}

marginal.cors = data.frame(Var = cat.features, stringsAsFactors = F)
marginal.cors$cor = NA
for (i in 1:nrow(marginal.cors)) {
  x = as.factor(train[[marginal.cors$Var[i]]])
  z = yenc.random.intercept(x, train$target, cv.folds, 1)[cv.folds$Fold1]
  marginal.cors$cor[i] = cor(z[!is.na(z)], y[!is.na(z)])
}

names(marginal.cors) = c('X1', 'cor1')
cat.triplets = merge(cat.triplets, marginal.cors, by = 'X1')
names(marginal.cors) = c('X2', 'cor2')
cat.triplets = merge(cat.triplets, marginal.cors, by = 'X2')
names(marginal.cors) = c('X3', 'cor3')
cat.triplets = merge(cat.triplets, marginal.cors, by = 'X3')

View(cat.triplets)
save(cat.triplets, file = 'triple-factor-interactions.RData')

#
# Between numericals
#

num.features = rownames(ftrs)[ftrs$type != 'character'][-(1:2)]

if (0) {
  # Filter highly correlated features (FIXME or maybe look at correlated pairs specifically?)
  x = randomForest::na.roughfix(train[, num.features])
  #findLinearCombos(x)
  fc = findCorrelation(cor(x), verbose = T, names = T)
  num.features = setdiff(num.features, fc)
}

config = list()
config$compute.backend = 'multicore'
config$nr.cores = 16
config$train = train
config$num.pairs = data.frame(t(combn(num.features, 2)), stringsAsFactors = F)

test.job = function(config, core) {
  pair.idxs = compute.backend.balance(nrow(config$num.pairs), config$nr.cores, core)
  nr.pairs.core = length(pair.idxs)
  if (nr.pairs.core == 0) {
    return (NULL)
  }
  
  pv.mul = rep(NA, nr.pairs.core)
  pv.div = rep(NA, nr.pairs.core)
  pv.vid = rep(NA, nr.pairs.core)
  
  for (ic in 1:nr.pairs.core) {
    # (I would use something like ad.test(x0, x1)$ad[1, 3], but it's way too expensive; maybe I
    # could subsample or parallelize, or both - it's doable)
    # Problem: conditional omnibus tests are absurdly expensive... so nevermind lets just use GLM GLRT
    
    i = pair.idxs[ic]
    f1 = config$num.pairs$X1[i]
    f2 = config$num.pairs$X2[i]
    frmla.null = as.formula(paste('target ~', f1, '+', f2))
    frmla.mul = as.formula(paste('target ~ ', f1, '+', f2, '+ I(', f1, '*', f2, ')'))
    frmla.div = as.formula(paste('target ~ ', f1, '+', f2, '+ I(', f1, '/', f2, ')'))
    frmla.vid = as.formula(paste('target ~ ', f1, '+', f2, '+ I(', f2, '/', f1, ')'))
    idx.div = config$train[[f2]] != 0
    idx.vid = config$train[[f1]] != 0
    pv.mul[ic] = anova(glm(frmla.null, config$train, family = 'binomial'), glm(frmla.mul, config$train, family = 'binomial'), test = 'Chisq')$`Pr(>Chi)`[2]
    pv.div[ic] = anova(glm(frmla.null, config$train[idx.div, ], family = 'binomial'), glm(frmla.div, config$train[idx.div, ], family = 'binomial'), test = 'Chisq')$`Pr(>Chi)`[2]
    pv.vid[ic] = anova(glm(frmla.null, config$train[idx.vid, ], family = 'binomial'), glm(frmla.vid, config$train[idx.vid, ], family = 'binomial'), test = 'Chisq')$`Pr(>Chi)`[2]
  }
  
  return (data.frame(pv.mul = pv.mul, pv.div = pv.div, pv.vid = pv.vid))
}

res = compute.backend.run(config, test.job, combine = rbind, package.dependencies = 'ComputeBackend')

num.pairs = cbind(config$num.pairs, res)
num.pairs$pv.mul = p.adjust(num.pairs$pv.mul, method = 'BH')
num.pairs$pv.div = p.adjust(num.pairs$pv.div, method = 'BH')
num.pairs$pv.vid = p.adjust(num.pairs$pv.vid, method = 'BH')

save(num.pairs, file = 'pairwise-numeric-interactions.RData')
View(num.pairs)

######################################################################################
# Starting from here I'm using a new way of computing this in parallel.
######################################################################################

#
# This is shared code needed by below segments
#

num.features = rownames(ftrs)[ftrs$type != 'character'][-(1:2)]
cat.features = rownames(ftrs)[ftrs$type == 'character']

# Build a version of the dataset in which we quantize all numericals
qnr = 30
train2 = train
for (f in num.features) {
  train2[[f]] = cut2(train[[f]], g = min(qnr, length(unique(train[[f]]))))
}

# Split to train and test
set.seed(123)
train.idx = sample(nrow(train), round(nrow(train)/2))
train.y = train$target[train.idx]
test.y = train$target[-train.idx]

# This fits an LMM to each on train, predict on test, and compute correlation with test.y
get.cor = function(x) {
  test.z = .yenc.random.intercept.core(x[train.idx], train.y, x[-train.idx])
  idx = !is.na(test.z)
  cor(test.z[idx], test.y[idx], method = 'spearman')
}

# Compute all marginal yenc test correlations (of quantized numericals, and of categoricals)
marginal.cors = data.frame(Var = c(num.features, cat.features), stringsAsFactors = F)
marginal.cors$cor = NA
for (i in 1:nrow(marginal.cors)) {
  cat(date(), '---', i, 'of', nrow(marginal.cors), '\n')
  x = as.factor(train2[[marginal.cors$Var[i]]])
  test.z = .yenc.random.intercept.core(x[train.idx], train.y, x[-train.idx])
  idx = !is.na(test.z)
  marginal.cors$cor[i] = cor(test.z[idx], test.y[idx])
}

#####
# Two-way interactions
#####

#
# Between categoricals
#

# These are all the pairs
cc.pairs = as.data.frame(t(combn(cat.features, 2)), stringsAsFactors = F)
names(cc.pairs) = c('Var1', 'Var2')

# Construct the interactions
dat.to.enc = as.data.frame(matrix(NA, nrow(train2), nrow(cc.pairs)))
for (i in 1:nrow(cc.pairs)) {
  cat(date(), '---', i, 'of', nrow(cc.pairs), '\n')
  x1 = as.factor(train2[[cc.pairs$Var1[i]]])
  x2 = as.factor(train2[[cc.pairs$Var2[i]]])
  dat.to.enc[, i] = interaction(x1, x2, drop = T)
}

# Compute correlations
cc.pairs$cor12 = unlist(mclapply(dat.to.enc, get.cor, mc.preschedule = F, mc.cores = 16))

# Add marginal correlations so that we examine the added value of interactions
names(marginal.cors) = c('Var1', 'cor1')
cc.pairs = merge(cc.pairs, marginal.cors, by = 'Var1')
names(marginal.cors) = c('Var2', 'cor2')
cc.pairs = merge(cc.pairs, marginal.cors, by = 'Var2')

# And I would use this like:
cc.pairs.to.include = cc.pairs[complete.cases(cc.pairs), ]
cc.pairs.to.include = cc.pairs.to.include[cc.pairs.to.include$cor12 - pmax(cc.pairs.to.include$cor1, cc.pairs.to.include$cor2) > 0.01, ]

# Look at it and save
save(cc.pairs, file = 'factor-factor-interactions.RData')
View(cc.pairs)

#
# Between categoricals and quantized numericals
#

# These are all the pairs
cn.pairs = data.frame(expand.grid(cat.features, num.features, stringsAsFactors = F), stringsAsFactors = F)

# Construct the interactions
dat.to.enc = as.data.frame(matrix(NA, nrow(train2), nrow(cn.pairs)))
for (i in 1:nrow(cn.pairs)) {
  cat(date(), '---', i, 'of', nrow(cn.pairs), '\n')
  x1 = as.factor(train2[[cn.pairs$Var1[i]]])
  x2 = as.factor(train2[[cn.pairs$Var2[i]]])
  dat.to.enc[, i] = interaction(x1, x2, drop = T)
}

# Compute correlations
cn.pairs$cor12 = unlist(mclapply(dat.to.enc, get.cor, mc.preschedule = F, mc.cores = 16))

# Add marginal correlations so that we examine the added value of interactions
names(marginal.cors) = c('Var1', 'cor1')
cn.pairs = merge(cn.pairs, marginal.cors, by = 'Var1')
names(marginal.cors) = c('Var2', 'cor2')
cn.pairs = merge(cn.pairs, marginal.cors, by = 'Var2')

# And I would use this like:
cn.pairs.to.include = cn.pairs[complete.cases(cn.pairs), ]
cn.pairs.to.include = cn.pairs.to.include[cn.pairs.to.include$cor12 - pmax(cn.pairs.to.include$cor1, cn.pairs.to.include$cor2) > 0.01, ]

# Look at it and save
save(cn.pairs, file = 'factor-numerical-interactions.RData')
View(cn.pairs)

#
# Between quantized numericals
#

# These are all the pairs
qqpairs = data.frame(t(combn(num.features, 2)), stringsAsFactors = F)
names(qqpairs) = c('Var1', 'Var2')

# Construct the interactions
dat.to.enc = as.data.frame(matrix(NA, nrow(train2), nrow(qqpairs)))
for (i in 1:nrow(qqpairs)) {
  cat(date(), '---', i, 'of', nrow(qqpairs), '\n')
  dat.to.enc[, i] = interaction(train2[[qqpairs$Var1[i]]], train2[[qqpairs$Var2[i]]], drop = T)
}

# Compute correlations
qqpairs$cor12 = NA
for (i in 1:ceiling(ncol(dat.to.enc) / 1000)) {
  cat(date(), '---', i, 'of', ceiling(ncol(dat.to.enc) / 1000), 'chunks\n')
  idx = (i - 1) * 1000 + (1:1000)
  idx = idx[idx <= ncol(dat.to.enc)]
  qqpairs$cor12[idx] = unlist(mclapply(dat.to.enc[, idx], get.cor, mc.preschedule = F, mc.cores = 16))
}

# Add marginal correlations so that we examine the added value of interactions
names(marginal.cors) = c('Var1', 'cor1')
qqpairs = merge(qqpairs, marginal.cors, by = 'Var1')
names(marginal.cors) = c('Var2', 'cor2')
qqpairs = merge(qqpairs, marginal.cors, by = 'Var2')

# And I would use this like:
qqpairs.to.include = qqpairs[complete.cases(qqpairs), ]
qqpairs.to.include = qqpairs.to.include[qqpairs.to.include$cor12 - pmax(qqpairs.to.include$cor1, qqpairs.to.include$cor2) > 0.01, ]

# Look at it and save
save(qqpairs, file = 'numerical-numerical-interactions.RData')
View(qqpairs)

#####
# Three-way interactions
#####

#
# Between categoricals
#

# These are all the triplets
cc.triplets = as.data.frame(t(combn(cat.features, 3)), stringsAsFactors = F)
names(cc.triplets) = c('Var1', 'Var2', 'Var3')

# Construct the interactions
dat.to.enc = as.data.frame(matrix(NA, nrow(train2), nrow(cc.triplets)))
for (i in 1:nrow(cc.triplets)) {
  cat(date(), '---', i, 'of', nrow(cc.triplets), '\n')
  x1 = as.factor(train2[[cc.triplets$Var1[i]]])
  x2 = as.factor(train2[[cc.triplets$Var2[i]]])
  x3 = as.factor(train2[[cc.triplets$Var3[i]]])
  dat.to.enc[, i] = interaction(x1, x2, x3, drop = T)
}

# Compute correlations
cc.triplets$cor123 = unlist(mclapply(dat.to.enc, get.cor, mc.preschedule = F, mc.cores = 16))

# Add marginal correlations so that we examine the added value of interactions
names(marginal.cors) = c('Var1', 'cor1')
cc.triplets = merge(cc.triplets, marginal.cors, by = 'Var1')
names(marginal.cors) = c('Var2', 'cor2')
cc.triplets = merge(cc.triplets, marginal.cors, by = 'Var2')
names(marginal.cors) = c('Var3', 'cor3')
cc.triplets = merge(cc.triplets, marginal.cors, by = 'Var3')

# TODO: Add pairwise correlations too... (we've already computed them, but it's annoying to add here)

# And I would use this like: (because I didn't add the 2-way correlations, some of these will actually be 2-way..)
cc.triplets.to.include = cc.triplets[complete.cases(cc.triplets), ]
cc.triplets.to.include = cc.triplets.to.include[cc.triplets.to.include$cor123 - pmax(cc.triplets.to.include$cor1, cc.triplets.to.include$cor2, cc.triplets.to.include$cor3) > 0.03, ]

# Look at it and save
save(cc.triplets, file = 'factor-3way-interactions.RData')
View(cc.triplets)

#####
# Four-way interactions
#####

#
# Between categoricals
#

# These are all the quads (except for v22, I think at this point it's absurd, and I don't have patience for the time it takes with this feature included...)
cc.quads = as.data.frame(t(combn(cat.features[cat.features != 'v22'], 4)), stringsAsFactors = F)
names(cc.quads) = c('Var1', 'Var2', 'Var3', 'Var4')

# Construct the interactions
dat.to.enc = as.data.frame(matrix(NA, nrow(train2), nrow(cc.quads)))
for (i in 1:nrow(cc.quads)) {
  cat(date(), '---', i, 'of', nrow(cc.quads), '\n')
  x1 = as.factor(train2[[cc.quads$Var1[i]]])
  x2 = as.factor(train2[[cc.quads$Var2[i]]])
  x3 = as.factor(train2[[cc.quads$Var3[i]]])
  x4 = as.factor(train2[[cc.quads$Var3[i]]])
  dat.to.enc[, i] = interaction(x1, x2, x3, x4, drop = T)
}

# Compute correlations
cc.quads$cor1234 = NA
for (i in 1:ceiling(ncol(dat.to.enc) / 500)) {
  cat(date(), '---', i, 'of', ceiling(ncol(dat.to.enc) / 500), 'chunks\n')
  idx = (i - 1) * 500 + (1:500)
  idx = idx[idx <= ncol(dat.to.enc)]
  cc.quads$cor1234[idx] = unlist(mclapply(dat.to.enc[, idx], get.cor, mc.preschedule = F, mc.cores = 16))
}

# Add marginal correlations so that we examine the added value of interactions
names(marginal.cors) = c('Var1', 'cor1')
cc.quads = merge(cc.quads, marginal.cors, by = 'Var1')
names(marginal.cors) = c('Var2', 'cor2')
cc.quads = merge(cc.quads, marginal.cors, by = 'Var2')
names(marginal.cors) = c('Var3', 'cor3')
cc.quads = merge(cc.quads, marginal.cors, by = 'Var3')
names(marginal.cors) = c('Var4', 'cor4')
cc.quads = merge(cc.quads, marginal.cors, by = 'Var4')

# TODO: Add pairwise and three-way correlations too... (we've already computed them, but it's annoying to add here)

# And I would use this like: (because I didn't add the lower order correlations, some of these will not actually be new)
cc.quads.to.include = cc.quads[complete.cases(cc.quads), ]
cc.quads.to.include = cc.quads.to.include[cc.quads.to.include$cor1234 - pmax(cc.quads.to.include$cor1, cc.quads.to.include$cor2, cc.quads.to.include$cor3) > 0.05, ]
# => Actually, it seems like all of these are obvious 3-ways (or less!)

# Look at it and save
save(cc.quads, file = 'factor-4way-interactions.RData')
View(cc.quads)

#---------------------------------------------------------------------------------------------------
#
# It seems that the features in this dataset have been (at least) scaled and translated, and that
# weak zero mean uniform noise: U(-1e-6, 1e-6) has been added to them for good measure. 
# Can we denoise them? Can we recover more (useful) information?

num.features = rownames(ftrs)[ftrs$type != 'character'][-(1:2)]

denoise = function(x) {
  idx.not.na = !is.na(x)
  order.x = order(x[idx.not.na])
  dx = diff(sort(x[idx.not.na]))
  xc = c(0, cumsum(round(dx, 5))) # this will be an almost perfect monotonic transform of the true clean x, which is enough for tree based memthods
  xc[order.x] = xc
  x.clean = x
  x.clean[idx.not.na] = xc
  
  #plot(sort(x)[10000 + (1:1000)])
  #lines(sort(x.clean)[10000 + (1:1000)], col = 2)
  
  #length(unique(x))
  #length(unique(x.clean))
  
  return (x.clean)
}

#---------------------------------------------------------------------------------------------------
#
# Of everything that I did, what made the biggest difference was modeling the high cardinality
# categoricals. Full fledged models aren't good for this, and the best solution I could come up with
# was to create meta-features using specialized marginal models to make these features useful. 
# It looks like all the competitors at the highest positions on the LB are well versed on this, 
# which corroborates the idea that this is important.
# I want to look into more/better ways of doing this. 

x = c(train$v22, test$v22)
x = factor(x)
hist(log(table(x)))
xnew = x[(nrow(train) + 1):length(x)]
x = x[1:nrow(train)]
y = train$target
cv.folds = createFolds(y, k = 5)

# Is it feasible to build a marginal saturated model on this variable without pruning rare levels?
z = yenc(x, y, cv.folds, -1, m = 30, xnew) # => yes, a few min
zlmm = yenc.random.intercept(x, y, cv.folds, -1, xnew) # => one sec??

plot(z, zlmm, pch = 19) # => no connection between them?!
abline(0, 1, col = 2)
idx = !(is.na(z) | is.na(zlmm))
cor(z[idx], zlmm[idx], method = 'pearson')

#---------------------------------------------------------------------------------------------------
#
# More generally, the XGB model depends on (conditional) marginal monotonic dependences. If some
# of the features "hide" behind a stubborn nonmonotonic marginal association with the target, it can
# be difficult for the model to use them. This is what happens with categoricals, and in that case
# it can sometimes be solved with stacking a marginal almost/saturated models. But this can also
# happen with annoyingly distributed numerical features. In that case one solution is to quantize
# them and then treat them as categorical. But I'm thinking maybe a better approach is to use a
# good 1d clustering algorithm. Also another thing that sometimes helps model with categoricals
# is frequency coding, which we can do with numericals as well using a good 1d density estimator.

# Let's look at density estimation first
x = train[['v72']]
hist(x, 100)
plot(table(cut2(x, g = 20), train$target))
density.est = density(x[!is.na(x)], n = length(x) / 100)
plot(density.est)
xd = approxfun(density.est)(x)
summary(xd)
head(cbind(x, xd))

xd2 = freq.encode(x)

#---------------------------------------------------------------------------------------------------
#
# Look again at the v22 situation
#

f = 'v22'
dat = rbind(train[, -2], test)
lvls = as.character(unique(dat[[f]]))
lvls = lvls[order(nchar(lvls), tolower(lvls))]
lvls = lvls[!is.na(lvls)]
lvls0 = c(LETTERS, apply(expand.grid(LETTERS, LETTERS), 1, paste0, collapse = ''), apply(expand.grid(LETTERS, LETTERS, LETTERS), 1, paste0, collapse = ''))
lvls0 = sapply(strsplit(lvls0, split = ''), function(str) { paste(rev(str), collapse = '') }) 
lvls0 = lvls0[lvls0 != 'NA']
mean(lvls0 == lvls[1:length(lvls0)])

#---------------------------------------------------------------------------------------------------
#
# Examine the useful kNN feature more closely
#

library(RANN)

train.labels = train$target
dat = train[, !(names(train) %in% c('ID', 'target', 'v75', 'v110', 'v107'))]

#
# Preprocessing
#
                          # Current useful feature:
pp.hexv.v22         = F   # T
pp.max.levels       = 200 # 30
pp.min.n.level      = 20  # 200
use.only.selected.x = F   # T

for (f in names(dat)[unlist(lapply(dat, is.character))]) {
  dat[[f]] = as.factor(dat[[f]])
}

# TODO: try with all variables coded hexv

if (pp.hexv.v22) {
  lvls = as.character(unique(dat[['v22']]))
  lvls = lvls[order(nchar(lvls), tolower(lvls))]
  na.idx = which(lvls == 'NB')
  dat[[paste0('hexv.', 'v22')]] = as.integer(factor(dat[['v22']], levels = lvls))
  dat$hexv.v22[!is.na(dat$hexv.v22) & dat$hexv.v22 >= na.idx] = dat$hexv.v22[!is.na(dat$hexv.v22) & dat$hexv.v22 >= na.idx] + 1
  # (it's possible that some/all of the NAs should be 'NA's, but we can't know that...)
}

names(dat)[names(dat) == 'v3'] = 'v3x' # change some names, otherwise dummyVars gets it wrong
nasty.features = names(dat)[unlist(lapply(dat, function(x) nlevels(x))) > pp.max.levels]
for (f in nasty.features) {
  tbl = head(sort(table(dat[[f]]), decreasing = T), pp.max.levels)
  levels.to.keep = names(tbl[tbl > pp.min.n.level])
  dat[!(dat[[f]] %in% levels.to.keep), f] = NA
  dat[[f]] = factor(dat[[f]])
}

frmla = as.formula('~ .')
dat = as.data.frame(predict(dummyVars(frmla, data = dat, sep = '_'), newdata = dat))

if (use.only.selected.x) {
  # These are the features on which a simple euclidean distance gave good predictive power
  load(file = 'nn-features.RData') # => nn.features
  
  cat('These are supposed to be in the data but arent:\n')
  print(setdiff(nn.features, names(dat))) # => it's just a couple of less common leves, nevermind
  nn.features = intersect(names(dat), nn.features)
  
  dat = dat[, nn.features]
}

# We can't compute distance with NAs
# FIXME do smarter imputation?
dat = randomForest::na.roughfix(dat)

#
# Split to a train and test set
#

ntrain = round(nrow(dat) * 2/3)
ntest = nrow(train) - ntrain
train.y = train.labels[1:ntrain]
test.y  = train.labels[ntrain + (1:ntest)]
train.x = dat[1:ntrain, ]
test.x  = dat[ntrain + (1:ntest), ]
rm(dat)

#
# Let's first look at the distance metric itslef through some examples
#

#sort(unlist(lapply(train.x, sd))) # => hexv.v22 has 1000 times the sd of any of the others

#
# Compute the features and look at them out of sample
#

examine.knn.features = function(train.x, train.y, test.x, nn.features, eps, k, scale = F) {
  res = list()

  if (scale) {
    dat = scale(rbind(train.x[, nn.features], test.x[, nn.features]))
    train.x[, nn.features] = dat[1:nrow(train.x), , drop = F]
    test.x[, nn.features] = dat[nrow(train.x) + (1:nrow(test.x)), , drop = F]
    rm(dat)
  }
  
  # # Distance from the '1' class
  # nn1.res = nn2(train.x[train.y == 1, nn.features], query = test.x[, nn.features], k = k, eps = eps)
  # res$nn1.dist.1 = nn1.res$nn.dists[, 1]
  # res$nn1.dist.2 = nn1.res$nn.dists[, 2]
  # res$nn1.dist.3 = nn1.res$nn.dists[, 3]
  # res$nn1.dist.k = nn1.res$nn.dists[, k]
  # res$nn1.dist.m = apply(nn1.res$nn.dists, 1, median)
  # res$nn1.dist.s = apply(nn1.res$nn.dists, 1, sd)
  # res$nn1.v50.1 = (train.x$v50[train.y == 1])[nn1.res$nn.idx[, 1]]
  # res$nn1.v50.m = rowMeans(matrix((train.x$v50[train.y == 1])[nn1.res$nn.idx], nrow = nrow(test.x)))
  # 
  # # Distance from the '0' class
  # nn0.res = nn2(train.x[train.y == 0, nn.features], query = test.x[, nn.features], k = k, eps = eps)
  # res$nn0.dist.1 = nn0.res$nn.dists[, 1]
  # res$nn0.dist.2 = nn0.res$nn.dists[, 2]
  # res$nn0.dist.3 = nn0.res$nn.dists[, 3]
  # res$nn0.dist.k = nn0.res$nn.dists[, k]
  # res$nn0.dist.m = apply(nn0.res$nn.dists, 1, median)
  # res$nn0.dist.s = apply(nn0.res$nn.dists, 1, sd)
  # res$nn0.v50.1 = (train.x$v50[train.y == 0])[nn0.res$nn.idx[, 1]]
  # res$nn0.v50.m = rowMeans(matrix((train.x$v50[train.y == 0])[nn0.res$nn.idx], nrow = nrow(test.x)))
  # 
  # # Relative distances
  # res$nn01d.dist.1 = res$nn1.dist.1 - res$nn0.dist.1
  # res$nn01d.dist.m = res$nn1.dist.m - res$nn0.dist.m
  # res$nn1d.dist.2m1 = res$nn1.dist.2 - res$nn1.dist.1
  # res$nn0d.dist.2m1 = res$nn0.dist.2 - res$nn0.dist.1
  # res$nn01d.dist.ss = rowSums(nn0.res$nn.dists > nn1.res$nn.dists)
  
  # The kNN predictor (FIXME use distances for weights)
  nn.res = nn2(train.x[, nn.features], query = test.x[, nn.features], k = k, eps = eps)
  res$knn.ypred = rowMeans(matrix(train.y[nn.res$nn.idx], nrow = nrow(test.x)))
  
  res = (as.data.frame(res))
  
  cat('kNN feature abs Spearman correlation with target:\n')
  res = t(as.data.frame(lapply(res, function(x) { idx = !is.na(x); cor(x[idx], y = test.y[idx], method = 'spearman') })))
  colnames(res) = 'cor'
  res = as.data.frame(abs(res))
  res[order(res$cor, decreasing = T), , drop = F]
}

eps.search = 5 # NOTE: must approximate unless one feature dominates or I move to a better implementation...
k.search   = 100

all.features = 1:ncol(train.x)
cat.features = grep('_', names(train.x))
num.features = grep('_', names(train.x), invert = T)

# On all features as is
examine.knn.features(train.x, train.y, test.x, all.features, eps.search, k.search)

# On categoricals only
examine.knn.features(train.x, train.y, test.x, cat.features, eps.search, k.search)

# On scaled numericals only
examine.knn.features(train.x, train.y, test.x, num.features, eps.search, k.search, scale = T)

if (0) {
  plot(table(cut2(test.xnn$nn0.dist.1, g = 20), test.y))
  plot(table(cut2(test.xnn$nn01d.dist.1, g = 20), test.y))
  
  test.xnn = generate.knn.features(train.x[, 'hexv.v22', drop = F], train.y, test.x[, 'hexv.v22', drop = F], eps.search, k.search) # Use only hexv.v22
  test.xnn = generate.knn.features(train.x[, names(train.x) != 'hexv.v22'], train.y, test.x[, names(test.x) != 'hexv.v22'], eps.search, k.search) # Use everything but hexv.v22
  test.xnn = generate.knn.features(train.x[, 1:112], train.y, test.x[, 1:112], eps.search, k.search) # on raw numericals only
  test.xnn = generate.knn.features(train.x[, 114:332], train.y, test.x[, 114:332], eps.search, k.search) # on OHE categoricals only
  test.xnn = generate.knn.features(train.x, train.y, test.x, eps.search, k.search)
  
  # On scaled numericals + categoricals
  test.xnn = generate.knn.features(cbind(train.x.ns, train.x[, 114:332]), train.y, cbind(test.x.ns, test.x[, 114:332]), eps.search, k.search)
  
  # On scaled numericals, except correlated ones
  removals = c('v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128')
  idx = !(names(train.x[, 1:112]) %in% removals)
  test.xnn = generate.knn.features(train.x.ns[, idx], train.y, test.x.ns[, idx], eps.search, k.search)
  
  # On all except correlated numericals
  test.xnn = generate.knn.features(train.x[, idx], train.y, test.x[, idx], eps.search, k.search)

  # maybe these should be given higher weight
  good = paste0('v', c(66, 79, 47, 56, 110, 31, 22, 112, 113, 24))
  gidx = which(unlist(lapply(strsplit(names(train.x)[idx], split = '_'), function(x) x[[1]] %in% good)))
  train.xw = train.x[, idx[gidx]]
  test.xw = test.x[, idx[gidx]]
  test.xnn = generate.knn.features(train.xw, train.y, test.xw, eps = 2, k.search) # this works wil eps down to 2
}

#---------------------------------------------------------------------------------------------------
#
# Look at all low-order GLM/GLMM models
#

train.y = train$target
train.x = train[, !(names(train) %in% c('ID', 'target', 'v75', 'v110', 'v107'))]

for (f in names(train.x)[unlist(lapply(train.x, is.character))]) {
  train.x[[f]] = as.factor(train.x[[f]])
}

cat.features = names(train.x)[ unlist(lapply(train.x, is.factor))]
num.features = names(train.x)[!unlist(lapply(train.x, is.factor))]

train.x[, num.features] = scale(train.x[, num.features])
train.x = randomForest::na.roughfix(train.x) # If we don't do this, then way too many examples will be thrown out by na.omit

# Split to train and test
ntrain = round(nrow(train.x) * 2/3)
ntest = nrow(train.x) - ntrain
test = train.x[ntrain + (1:ntest), ]
test.targets = train.y[ntrain + (1:ntest)]
train = train.x[1:ntrain, ]
train$target = train.y[1:ntrain]
rm(train.x, train.y)

# Remove factor levels from test that are not in train
for (f in names(test)[unlist(lapply(test, is.factor))]) {
  train[[f]] = factor(train[[f]])
  test[[f]] = factor(test[[f]], levels = levels(train[[f]]))
}

logloss = function(preds, labels, na.rm = F) {
  preds = pmax(pmin(preds, 1 - 1e-15), 1e-15)
  -mean(log(ifelse(labels == 0, 1 - preds, preds)), na.rm = na.rm)
}

# Fixed effects only (only makes sense for numericals and low-cardinality categoricals)
# FIXME implement for multiple features (currently it's only univariate)
glm.pred = function(f) {
  if (is.factor(train[[f]]) && nlevels(train[[f]]) > 100) return (NA) # yeah I could prune levels etc...
  lm.fit = glm(as.formula(paste0('target ~ ', f)), binomial, train)
  preds = predict(lm.fit, test, 'response')
  logloss(preds, test.targets)
}

# Random intercepts (only makes sense for categoricals with at least some levels that repeat enough times)
# FIXME implement for multiple features (currently it's only univariate)
glmm1.pred = function(f) {
  lmer.fit = glmer(as.formula(paste0('target ~ 1 + (1 | ', f, ')')), train, binomial)
  xlvls = rownames(ranef(lmer.fit)[[1]])
  x.blup = fixef(lmer.fit) + ranef(lmer.fit)[[1]][, 1]
  preds = 1 / (1 + exp(-x.blup[match(test[[f]], xlvls)]))
  logloss(preds, test.targets, na.rm = T)
}

# Fixed effect and random intercept (makes sense for a (numerical, categorical) pair, or perhaps a (low card cat, cat) pair)
# FIXME implement for multiple features of each type (currently it's only univariate)
glmm2.pred = function(f1, f2) {
  lmer.fit = glmer(as.formula(paste0('target ~ ', f1, ' + (1 | ', f2, ')')), train, binomial)
  xlvls = rownames(ranef(lmer.fit)[[1]])
  x.blup = ranef(lmer.fit)[[1]][, 1]
  eta = fixef(lmer.fit)[1] + fixef(lmer.fit)[2] * test[[f1]] + x.blup[match(test[[f2]], xlvls)]
  preds = 1 / (1 + exp(-eta))
  logloss(preds, test.targets, na.rm = T)
}

# Random slopes (makes sense for a (num, cat) pair)
glmm3.pred = function(f1, f2) {
  lmer.fit = glmer(as.formula(paste0('target ~ 1 + (', f1, ' - 1 | ', f2, ')')), train, binomial)
  xlvls = rownames(ranef(lmer.fit)[[1]])
  x.blup = ranef(lmer.fit)[[1]][, 1]
  eta = fixef(lmer.fit) + test[[f1]] * x.blup[match(test[[f2]], xlvls)]
  preds = 1 / (1 + exp(-eta))
  logloss(preds, test.targets, na.rm = T)
}

glm.scores = sapply(names(test), glm.pred)
glmm.scores = sapply(names(test), glmm1.pred)

#---------------------------------------------------------------------------------------------------
#
# A look at top current features
#


