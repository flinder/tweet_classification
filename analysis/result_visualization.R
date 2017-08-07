library(reshape2)
library(tidyverse)
library(xtable)

source('plot_theme.R')


# Crowdflower word table
kw <- read_csv('../data/crowdflower_keywords.csv')
kw$translation <- NA
tab <- xtable(kw[1:10, ], digits = 2, 
              caption = "List of keywords suggested by survey participants.",
              label = "tab:cf_keywords")
print(tab, file = '../paper/tables/cf_keywords.tex', include.rownames = FALSE)


# Full experiment results
df <- read_csv('../data/experiment_results.csv') %>%
    filter(method != "clf_random")
 
# Relabel
df$method[df$method == "keyword"] <- "Keyword"
df$method[df$method == "search"] <- "Expansion"
#df$method[df$method == "clf_random"] <- "Expansion + Random ML"
df$method[df$method == "clf_active"] <- "Expansion + Active ML"

ggplot(filter(df, is.element(measure, c('precision', 'recall', 'f1'))), 
              aes(x = iteration, y = value, color = method, linetype = method)) +
    #geom_point(alpha = 0.6, isze = 0.2, position = "jitter") +
    geom_smooth() +
    facet_wrap(~ measure) +
    ylab("") + xlab("# Keywords") +
    guides(color=guide_legend(title=""), linetype=guide_legend(title="")) +
    scale_color_manual(values = cbPalette[-1]) +
    ylim(0,1) +
    plot_theme
ggsave(filename = '../paper/figures/evaluation_prec_rec.png', width = p_width, 
       height = 0.5 * p_width, dpi = 300)
ggsave(filename = '../presentation/figures/evaluation_prec_rec.png', width = p_width, 
       height = 0.5 * p_width, dpi = 300)


ggplot(filter(df, !is.element(measure, c('precision', 'recall', 'f1'))), 
              aes(x = iteration, y = value, color = method, linetype = method)) +
    #geom_point(alpha = 0.6, isze = 0.2, position = "jitter") +
    geom_smooth() +
    facet_wrap(~ measure) +
    ylab("") + xlab("# Keywords") +
    guides(color=guide_legend(title=""), linetype=guide_legend(title="")) +
    scale_color_manual(values = cbPalette[-1]) +
    ylim(0,1) +
    plot_theme
ggsave(filename = '../paper/figures/evaluation_similarity.png', width = p_width, 
       height = 0.5 * p_width, dpi = 300)
ggsave(filename = '../presentation/figures/evaluation_similarity.png', width = p_width, 
       height = 0.5 * p_width, dpi = 300)


ggplot(df, aes(x = iteration, y = value, color = method)) +
    geom_line(aes(group = replication), alpha = 0.6, size = 0.2, position = "jitter") +
    geom_smooth() +
    facet_wrap(~ method + measure, nrow = 3) +
    ylab("") + xlab("# Keywords") +
    guides(color=FALSE) +
    scale_color_manual(values = cbPalette[-1]) +
    ylim(0,1) +
    scale_y_continuous(breaks=c(0, 0.5, 1)) +
    theme(strip.text = element_text(size=2)) +
    plot_theme
ggsave(filename = '../paper/figures/evaluation_detail.png', width = p_width, 
       height = p_width, dpi = 300)
ggsave(filename = '../presentation/figures/evaluation_detail.png', width = p_width, 
       height = p_width, dpi = 150)


# Some stats for the discussion
stats <- 
    group_by(df, variable, measure) %>%
    summarize(min = min(value),
              lo = quantile(value, 0.025),
              avg = mean(value), 
              median = median(value),
              hi = quantile(value, 0.975),
              max = max(value)) %>%
    print()




######################################################################
keyword <- read_csv('../data/keyword_only_results.csv') %>%
    #filter(n_keywords > 4 & n_keywords < 300) %>%
    mutate(keyword_precision = kw_precision,
           keyword_recall = kw_recall,
           keyword_f1 = kw_f1) %>%
    select(keyword_precision, keyword_recall, keyword_f1, n_keywords, replication) %>%
    melt(id.vars = c("n_keywords", "replication")) %>% 
    tbl_df() %>%
    mutate(measure = sapply(variable, 
                            function(x) unlist(strsplit(as.character(x), '_'))[2]),
           variable = sapply(variable, 
                             function(x) unlist(strsplit(as.character(x), '_'))[1]))

expansion <- read_csv('../data/full_system_scores_2.csv')  %>%
    mutate(n_keywords = iteration+1,
           expansion_precision = precision_kw,
           expansion_recall = recall_kw,
           expansion_f1 = f1_kw,
           full_precision = precision_clf,
           full_recall = recall_clf,
           full_f1 = f1_clf) %>%
    select(expansion_precision, expansion_recall, expansion_f1, full_precision, 
           full_recall, full_f1, n_keywords, replication) %>%
    melt(id.vars = c("n_keywords", "replication")) %>% 
    tbl_df() %>%
    mutate(measure = sapply(variable, 
                            function(x) unlist(strsplit(as.character(x), '_'))[2]),
           variable = sapply(variable, 
                             function(x) unlist(strsplit(as.character(x), '_'))[1]))

df <- rbind(keyword, expansion)
