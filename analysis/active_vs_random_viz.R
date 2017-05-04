library(ggplot2)
library(dplyr)
library(reshape2)

source('plot_theme.R')

dat <- read.csv('../data/active_random_sgd_log_all_data.csv') %>%
    mutate(iter_clf = paste0(clf, '_', iteration)) %>% tbl_df()
df <- melt(dat, id.vars = c("clf", "iteration", "iter_clf", "train_size"))

# Precision recall and f1 score
p <- ggplot(df, aes(x = train_size, y = value, color = clf,
               fill = clf)) + 
    geom_line(aes(group = factor(iter_clf)), 
                se = FALSE, alpha = 0.02,
                size = 0.3, stat='smooth', method='loess') +
    geom_smooth() +
    facet_wrap(~ variable) +
    theme_bw() +
    theme(panel.border = element_blank())
ggsave(p, filename = '../paper/figures/active_vs_random.png', 
       width = p_width, height = 0.6*p_width)
