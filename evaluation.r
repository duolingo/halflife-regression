# Copyright (c) 2016 Duolingo Inc. MIT Licence.

library('pROC')

sr_evaluate <- function(preds_file) {
    cat(paste('%%%%%%%%%%%%%%%%', preds_file, '%%%%%%%%%%%%%%%%\n'));
    data <- read.csv(preds_file, sep='\t');
    cat('==== mean absolute error ====\n')
    print(t.test(abs(data$p - data$pp), abs(data$p - mean(data$p)), alternative='l'));
    cat('==== area under the ROC curve ====\n')
    print(roc(round(p) ~ pp, data=data));
    print(wilcox.test(round(data$p), data$pp, alternative='g'));
    cat('==== half-life correlation ====\n')
    print(cor.test(data$h, data$hh, method='s'));
}
