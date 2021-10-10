#Doc: https://cran.r-project.org/web/packages/plm/vignettes/A_plmPackage.html

#Load packages
pacman::p_load(stargazer, plm, lmtest)
#Load data
dat <- read.csv("firms.csv")


###----------MODELS-------###
### reg1: FEs
fe = plm(ldsa ~ lcap + lemp,
         data = dat, index = c("firmid","year"), 
          model="within", effect="individual")

### reg2: FD
fd = plm(ldsa ~ -1+ lcap + lemp,
            data = dat, index = c("firmid","year"), model="fd")

#Compute heteroscedasticity-robust cov matrix
fe_se <- coeftest(fe,vcov=vcovHC(fe,type="HC0"))
fd_se <- coeftest(fd,vcov=vcovHC(fd,type="HC0"))


###-----------Table-------------###

stargazer(fe, fd,  
          se=list(fe_se[,2], fd_se[,2]),     
          title="Results", type="latex", 
          column.labels=c("FE", "FD"),
          align=TRUE, dep.var.labels=c("Log deflated sales"),
          covariate.labels=c("Log adjusted capital","Log employment"),
          omit.stat=c("LL","ser","f"),
          df = FALSE)



### Test lead and lag


lead <- plm(ldsa ~ lcap + lemp + lead(lcap) +  lead(lemp),
         data = dat, index = c("firmid","year"), 
         model="within", effect="individual") 

lag <- plm(ldsa ~ lcap + lemp + lag(lcap) +  lag(lemp),
            data = dat, index = c("firmid","year"), 
            model="within", effect="individual") 



lead_se <- coeftest(lead1,vcov=vcovHC(lead1,type="HC0"))
lag_se <- coeftest(lag1,vcov=vcovHC(lag1,type="HC0"))

stargazer(lead,  lag, 
          se=list(lead1_se[,2],lag1_se[,2]),     
          title="Results", type="latex", 
          align=TRUE, dep.var.labels=c("Log deflated sales"),
          omit.stat=c("LL","ser","f"),
          df = FALSE)
