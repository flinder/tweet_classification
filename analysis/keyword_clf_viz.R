library(reshape2)
library(tidyverse)

source('plot_theme.R')

n <- 19420
df <- read_csv('../data/keyword_clf.csv') %>%
    mutate(n_keywords = 1:199,
           "Proportion Annotated" = n_annotated / n,
           "Proportion Selected" = n_selected / n) %>%
    select(-f1, -n_selected, -n_annotated) %>%
    melt(id.vars = c("n_keywords")) %>%
    tbl_df()
colnames(df) <- c("n_keywords", "Measure", "Score")

p <- ggplot(df) +
    geom_line(aes(y = Score, x = n_keywords, color = Measure, 
              linetype = Measure), size = 1) +
    scale_color_manual(values = cbPalette) +
    ylim(0,1) + ylab("Precision / Recall") + xlab("Number of Keywords") +
    plot_theme
ggsave(p, filename = '../paper/figures/keyword_clf_precrec.png', 
       width = p_width, height = 0.6*p_width)


# Baseline
df <- read_csv('../data/keyword_baseline_res.csv') %>%
    mutate(n_keywords = 1:200,
           "Proportion Selected" = n_clf_pos / n) %>%
    select(-f1, -n_clf_pos) %>%
    melt(id.vars = c("n_keywords")) %>%
    tbl_df()
colnames(df) <- c("n_keywords", "Measure", "Score")

p <- ggplot(df) +
    geom_line(aes(y = Score, x = n_keywords, color = Measure, 
              linetype = Measure), size = 1) +
    scale_color_manual(values = cbPalette) +
    ylim(0,1) + ylab("Precision / Recall") + xlab("Number of Keywords") +
    plot_theme
ggsave(p, filename = '../paper/figures/baseline_precrec.png', 
       width = p_width, height = 0.6*p_width)
