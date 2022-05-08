# declare -a inpath = ("../TARGET_Fusion/" "../TARGET_RNAseq/HTSeq_counts/" "")

num_parallel=24

paras[0]='-m -d TARGET --outpath ../final_result/TARGET/methy+counts/mycn/ --feature_type both'
paras[1]='-m -d TARGET --outpath ../final_result/TARGET/methy+counts/mycn/ --feature_type cen'
paras[2]='-m -d TARGET --outpath ../final_result/TARGET/methy+counts/mycn/ --feature_type mod'
paras[3]='-m -d TARGET --outpath ../final_result/TARGET/methy+counts/mycn/ --fusion feature --feature_type both'
paras[4]='-m -d TARGET --outpath ../final_result/TARGET/methy+counts/mycn/ --fusion feature --feature_type cen'
paras[5]='-m -d TARGET --outpath ../final_result/TARGET/methy+counts/mycn/ --fusion feature --feature_type mod'
paras[6]='-m -d TARGET --inpath ../TARGET_RNAseq/HTSeq_counts/ --outpath ../final_result/TARGET/RNA-seq/HTSeq_counts/mycn/ --fusion none --feature_type both'
paras[7]='-m -d TARGET --inpath ../TARGET_RNAseq/HTSeq_counts/ --outpath ../final_result/TARGET/RNA-seq/HTSeq_counts/mycn/ --fusion none --feature_type cen'
paras[8]='-m -d TARGET --inpath ../TARGET_RNAseq/HTSeq_counts/ --outpath ../final_result/TARGET/RNA-seq/HTSeq_counts/mycn/ --fusion none --feature_type mod'
paras[9]='-m -d TARGET --inpath ../TARGET_Methylation/ --outpath ../final_result/TARGET/Methylation/mycn/ --fusion none --feature_type both'
paras[10]='-m -d TARGET --inpath ../TARGET_Methylation/ --outpath ../final_result/TARGET/Methylation/mycn/ --fusion none --feature_type cen'
paras[11]='-m -d TARGET --inpath ../TARGET_Methylation/ --outpath ../final_result/TARGET/Methylation/mycn/ --fusion none --feature_type mod'
paras[12]='-m -d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Network_level_fusion/mycn/ --feature_type both'
paras[13]='-m -d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Network_level_fusion/mycn/ --feature_type cen'
paras[14]='-m -d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Network_level_fusion/mycn/ --feature_type mod'
paras[15]='-m -d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Feature_level_fusion/mycn/ --fusion feature --feature_type both'
paras[16]='-m -d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Feature_level_fusion/mycn/ --fusion feature --feature_type cen'
paras[17]='-m -d SEQC --inpath ../GSE49710+62564/ --outpath ../final_result/SEQC/Feature_level_fusion/mycn/ --fusion feature --feature_type mod'
paras[18]='-m -d SEQC --inpath ../GSE49710_original/ --outpath ../final_result/SEQC/GSE49710/mycn/ --fusion none --feature_type both'
paras[19]='-m -d SEQC --inpath ../GSE49710_original/ --outpath ../final_result/SEQC/GSE49710/mycn/ --fusion none --feature_type cen'
paras[20]='-m -d SEQC --inpath ../GSE49710_original/ --outpath ../final_result/SEQC/GSE49710/mycn/ --fusion none --feature_type mod'
paras[21]='-m -d SEQC --inpath ../GSE62564_original/ --outpath ../final_result/SEQC/GSE62564/mycn/ --fusion none --feature_type both'
paras[22]='-m -d SEQC --inpath ../GSE62564_original/ --outpath ../final_result/SEQC/GSE62564/mycn/ --fusion none --feature_type cen'
paras[23]='-m -d SEQC --inpath ../GSE62564_original/ --outpath ../final_result/SEQC/GSE62564/mycn/ --fusion none --feature_type mod'

for (( p = 12; p < 18; p++ )); do
    (
        python3 train.py ${paras[p]} > ../output/mycn/${p}.out
    )&
done