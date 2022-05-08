num_parallel=24

paras[0]='-d TARGET --outpath ../final_result/TARGET/methy+counts/ --feature_type both'
paras[1]='-d TARGET --outpath ../final_result/TARGET/methy+counts/ --feature_type cen'
paras[2]='-d TARGET --outpath ../final_result/TARGET/methy+counts/ --feature_type mod'
paras[3]='-d TARGET --outpath ../final_result/TARGET/methy+counts/ --fusion feature --feature_type both'
paras[4]='-d TARGET --outpath ../final_result/TARGET/methy+counts/ --fusion feature --feature_type cen'
paras[5]='-d TARGET --outpath ../final_result/TARGET/methy+counts/ --fusion feature --feature_type mod'
paras[6]='-d TARGET --inpath ../TARGET_RNAseq/HTSeq_counts/ --outpath ../final_result/TARGET/RNA-seq/HTSeq_counts/ --fusion none --feature_type both'
paras[7]='-d TARGET --inpath ../TARGET_RNAseq/HTSeq_counts/ --outpath ../final_result/TARGET/RNA-seq/HTSeq_counts/ --fusion none --feature_type cen'
paras[8]='-d TARGET --inpath ../TARGET_RNAseq/HTSeq_counts/ --outpath ../final_result/TARGET/RNA-seq/HTSeq_counts/ --fusion none --feature_type mod'
paras[9]='-d TARGET --inpath ../TARGET_Methylation/ --outpath ../final_result/TARGET/Methylation/ --fusion none --feature_type both'
paras[10]='-d TARGET --inpath ../TARGET_Methylation/ --outpath ../final_result/TARGET/Methylation/ --fusion none --feature_type cen'
paras[11]='-d TARGET --inpath ../TARGET_Methylation/ --outpath ../final_result/TARGET/Methylation/ --fusion none --feature_type mod'
paras[12]='-d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Network_level_fusion/weighted_cv/ --feature_type both'
paras[13]='-d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Network_level_fusion/weighted_cv/ --feature_type cen'
paras[14]='-d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Network_level_fusion/weighted_cv/ --feature_type mod'
paras[15]='-d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Feature_level_fusion/weighted_cv/ --fusion feature --feature_type both'
paras[16]='-d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Feature_level_fusion/weighted_cv/ --fusion feature --feature_type cen'
paras[17]='-d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Feature_level_fusion/weighted_cv/ --fusion feature --feature_type mod'
paras[18]='-d SEQC --inpath ../GSE49710_original/ --outpath ../final_result/SEQC/GSE49710/weighted_cv/ --fusion none --feature_type both'
paras[19]='-d SEQC --inpath ../GSE49710_original/ --outpath ../final_result/SEQC/GSE49710/weighted_cv/ --fusion none --feature_type cen'
paras[20]='-d SEQC --inpath ../GSE49710_original/ --outpath ../final_result/SEQC/GSE49710/weighted_cv/ --fusion none --feature_type mod'
paras[21]='-d SEQC --inpath ../GSE62564_original/ --outpath ../final_result/SEQC/GSE62564/weighted_cv/ --fusion none --feature_type both'
paras[22]='-d SEQC --inpath ../GSE62564_original/ --outpath ../final_result/SEQC/GSE62564/weighted_cv/ --fusion none --feature_type cen'
paras[23]='-d SEQC --inpath ../GSE62564_original/ --outpath ../final_result/SEQC/GSE62564/weighted_cv/ --fusion none --feature_type mod'

for (( p = 0; p < num_parallel; p++ )); do
    (
        python3 train.py ${paras[p]} > ../output/${p}.out
    )&
done