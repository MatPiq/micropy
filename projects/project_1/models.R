#Doc: https://cran.r-project.org/web/packages/plm/vignettes/A_plmPackage.html

#Load packages
pacman::p_load(stargazer, plm, lmtest, sandwich)
#Load data
dat <- read.csv("firms.csv")


###----------MODELING and TESTS-------###
### reg1: OLS (pooled)
pools = plm(ldsa ~ lcap + lemp,
           data = dat, index = c("firmid","year"), model="pooling")

### reg3: RE
re = plm(ldsa ~ lcap + lemp,
         data = dat, index = c("firmid","year"), model="random")

### reg2: FEs
fe = plm(ldsa ~ lcap + lemp,
         data = dat, index = c("firmid","year"), 
          model="within", effect="individual")
### reg3: FD
fd = plm(ldsa ~ -1+ lcap + lemp,
            data = dat, index = c("firmid","year"), model="fd")


### Hausman test

phtest(fe, re)

### Serial correlation test
pbgtest(fd)
pbgtest(fe)

pwfdtest(fd)

###-----------Tables-------------###

stargazer(pools, re, fe, fd,  
          #se=list(clse(pools), clse(re), clse(fe),clse(fd)),     
          title="Results", type="latex", 
          column.labels=c("POOLS", "RE", "FE", "FD"),
          align=TRUE, dep.var.labels=c("Log deflated sales"),
          covariate.labels=c("Log adjusted capital","Log employment"),
          omit.stat=c("LL","ser","f"), ci=TRUE, ci.level=0.95,
          df = FALSE)


