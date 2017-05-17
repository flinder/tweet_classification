library(dplyr)
library(ggplot2)
library(reshape2)


source('plot_theme.R')

dat <- read.csv('../data/keyword_baseline_res.csv') %>%
    mutate(n_keywords = 1:200)

df <-  dat %>%
    select(-f1, -n_clf_pos) %>% 
    melt(id.vars = c("n_keywords")) %>%
    tbl_df()
colnames(df) <- c("n_keywords", "Measure", "Score")

n_pos <- select(dat, n_keywords, n_clf_pos) %>%
    mutate(prop_positive = n_clf_pos / 19420) # THis is the total ds size

p <- ggplot(df) +
    geom_area(data = n_pos, aes(x = n_keywords, y = prop_positive), 
              fill = cbPalette[3], alpha = 0.3) +
    geom_line(aes(y = Score, x = n_keywords, color = Measure, 
              linetype = Measure), size = 1) +
    scale_color_manual(values = cbPalette) +
    ylim(0,1) + ylab("Precision / Recall") + xlab("Number of Keywords") +
    plot_theme
ggsave(p, filename = '../paper/figures/baseline_precrec.png', 
       width = p_width, height = 0.6*p_width)
