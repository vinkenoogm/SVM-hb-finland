library(dplyr)
library(tidyr)
library(ggplot2)
library(ggpubr)

setwd('C:/Users/mvink/OneDrive/PhD/Research/202205 SVM Finnish data/results/distributions')

df <- read.csv('hb_distrs_by_donor.csv', row.names=1) %>%
  mutate(sex = factor(sex),
         across(.cols = snp_1:snp_17, 
                .fns = ~factor(., levels=c(0,1,2), labels=c(0,1,2)))) %>%
  drop_na() %>%
  filter(hbvar != 0)

plot_hbmeans <- function(data, snp, column, save=FALSE) {
  title <- ifelse(column == 'hbmean', 'Mean Hb', 'Hb variance')
  ylab <- ifelse(column == 'hbmean', 'Mean Hb value', 'Hb variance')
  ggplot(data, aes_string(x = snp, y = column)) +
    geom_boxplot() +
    ggtitle(paste(title, 'per donor by', snp)) +
    xlab(paste('Value for', snp)) +
    ylab(ylab) +
    theme_minimal()
  if (save) {
    ggsave(paste0(column, '_', snp, '.png'))
  }
}


plot_hbmeans(df, 'snp_15', 'hbvar', save=TRUE)

df %>%
  group_by(snp_17) %>%
  summarise(mean_hbmean = mean(hbmean),
            mean_hbvar = mean(hbvar),
            n = n())
