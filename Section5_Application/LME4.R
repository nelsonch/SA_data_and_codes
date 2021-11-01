#######
#library(lme4)
library(lme4)
library(sjstats)
#library(unmatrix)
#dat = sleepstudy

####
dat = read.csv('try668.csv')
dat = dat[order(dat$store_clus),]
dat = dat[, c('store_clus', 'disc', 'logistic_qty')]
unique(dat$store_clus)

fm1 <- lmer(logistic_qty~disc + (1 + disc || store_clus), dat, REML = T) 
summary(fm1)

###### make predictions
gh <- predict(fm1)
y <- dat$logistic_qty
sqrt(sum((gh -y)^2)/length(y))

