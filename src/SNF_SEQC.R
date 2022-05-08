setwd("~/Documents/Researches/Network_fusion/FYP_WULUE-master")

library(SNFtool)

rnaseq <- read.csv("GSE49710_original/new_adjmat.csv", header = 1, row.names = 1)
miarray <- read.csv("GSE62564_original/new_adjmat.csv", header = 1, row.names = 1)

W1 = as.matrix(rnaseq)
W2 = as.matrix(miarray)

K = 20
t = 20

W = SNF(list(W1, W2), K, alpha)
write.csv(W, file = "GSE49710+62564/new_adjmat.csv", row.names = TRUE)
