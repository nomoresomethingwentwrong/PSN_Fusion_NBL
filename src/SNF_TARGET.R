setwd("~/Documents/Researches/Network_fusion/FYP_WULUE-master")

library(SNFtool)

rnaseq <- read.csv("TARGET_RNAseq/HTSeq_counts/new_adjmat.csv", header = 1, row.names = 1)
dname <- read.csv("TARGET_DNAMethylation/new_adjmat.csv", header = 1, row.names = 1)

W1 = as.matrix(rnaseq)
W2 = as.matrix(dname)

K = 20
t = 20

W = SNF(list(W1, W2), K, alpha)
write.csv(W, file = "TARGET_Fusion/new_adjmat.csv", row.names = TRUE)
