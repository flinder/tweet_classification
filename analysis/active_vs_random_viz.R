library(ggplot2)
library(dplyr)
library(reshape2)

source('plot_theme.R')

dat <- read.csv('../data/active_random_w_sparsity_new_data.csv') %>%
    mutate(iter_clf = paste0(clf, '_', iteration),
           train_size_perc = 100 * train_size / max_samples,
           sparsity_label = paste0("sparsity=", sparsity)) %>% tbl_df()


# Precision recall and f1 score
p <- ggplot(dat, aes(x = train_size_perc, y = f1, color = clf,
               fill = clf, linetype = clf)) + 
    geom_point(alpha = 0.1, size = 0.3) +
    geom_smooth() +
    facet_wrap(~ as.factor(sparsity_label)) +
    theme_bw() +
    scale_color_manual(values = cbPalette) +
    scale_fill_manual(values = cbPalette) +
    theme(panel.border = element_blank())
ggsave(p, filename = '../paper/figures/active_vs_random.png', 
       width = p_width, height = 0.6*p_width)
