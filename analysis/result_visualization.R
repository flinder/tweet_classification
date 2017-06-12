library(reshape2)
library(tidyverse)


source('plot_theme.R')

# Randomized increasing number

dat <- read_csv('../data/experiment_results.csv') #%>%

df <- select(dat, -n_selected, -replication, -clf_precision, -clf_recall, 
             -kw_precision, -kw_recall) %>% 
    melt(id.vars = c("n_keywords")) %>%
    tbl_df() 
colnames(df) <- c("n_keywords", "Measure", "Score")

ggplot(df, aes(x = n_keywords, y = Score, color = Measure, linetype = Measure)) +
    geom_point(size = 0.3, alpha = 0.2, position = "jitter") +
    stat_smooth() +
    scale_color_manual(values = cbPalette[-1]) + 
    plot_theme


# System with manual query expansion 
dat <- read_csv('../data/full_system_scores.csv')

df <- dat %>%
    mutate(n_keywords = iteration + 5) %>%
    select(-iteration, -replication, -recall_clf, -recall_kw, -precision_clf,
           -precision_kw) %>%
    melt(id.vars = c("n_keywords")) %>%
    tbl_df() 
colnames(df) <- c("n_keywords", "Measure", "Score")

ggplot(df, aes(x = n_keywords, y = Score, color = Measure, linetype = Measure)) +
    geom_point(size = 1, alpha = 0.3, position = "jitter") +
    stat_smooth() +
    scale_color_manual(values = cbPalette[-1]) + 
    plot_theme

ggplot(dat) + geom_point(aes(x = recall_clf, y = recall_kw)) + geom_abline(slope = 1)





keyword <- read_csv('../data/experiment_results.csv') %>%
    filter(n_keywords > 4 & n_keywords < 31) %>%
    mutate(keyword_precision = kw_precision,
           keyword_recall = kw_recall,
           keyword_f1 = kw_f1) %>%
    select(keyword_precision, keyword_recall, keyword_f1, n_keywords) %>%
    melt(id.vars = "n_keywords") %>% 
    tbl_df() %>%
    mutate(measure = sapply(variable, 
                            function(x) unlist(strsplit(as.character(x), '_'))[2]),
           variable = sapply(variable, 
                             function(x) unlist(strsplit(as.character(x), '_'))[1]))

expansion <- read_csv('../data/full_system_scores_1.csv')  %>%
    filter(f1_clf > 0) %>%
    mutate(n_keywords = iteration + 5,
           expansion_precision = precision_kw,
           expansion_recall = recall_kw,
           expansion_f1 = f1_kw,
           full_precision = precision_clf,
           full_recall = recall_clf,
           full_f1 = f1_clf) %>%
    select(expansion_precision, expansion_recall, expansion_f1, full_precision, 
           full_recall, full_f1, n_keywords) %>%
    melt(id.vars = "n_keywords") %>% 
    tbl_df() %>%
    mutate(measure = sapply(variable, 
                            function(x) unlist(strsplit(as.character(x), '_'))[2]),
           variable = sapply(variable, 
                             function(x) unlist(strsplit(as.character(x), '_'))[1]))

df <- rbind(keyword, expansion)

ggplot(df, aes(x = n_keywords, y = value, color = variable, linetype = variable)) +
    #geom_point(alpha = 0.6, size = 0.2, position = "jitter") +
    geom_smooth() +
    facet_wrap(~ measure) +
    ylab("") + xlab("# Keywords") +
    guides(color=guide_legend(title=""), linetype=guide_legend(title="")) +
    scale_color_manual(values = cbPalette[-1]) +
    ylim(0,1) +
    plot_theme
ggsave(filename = '../paper/figures/evaluation.png', width = p_width, 
       height = 0.5 * p_width, dpi = 300)
