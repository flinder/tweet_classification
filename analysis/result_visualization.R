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
df <- read_csv('../data/experiment_results.csv')
    #filter(method != "clf_random")
 
# Relabel
df$method[df$method == "keyword"] <- "Keyword"
df$method[df$method == "search"] <- "Expansion"
#df$method[df$method == "clf_random"] <- "Expansion + Random ML"
df$method[df$method == "clf_active"] <- "Expansion + Active ML"
df$method[df$method == "clf_random"] <- "Expansion + Random ML"
df <- filter(df, replication <= 16)

ggplot(filter(df, is.element(measure, c('precision', 'recall', 'f1'))), 
              aes(x = iteration, y = value, color = method, linetype = method)) +
    #geom_point(alpha = 0.6, isze = 0.2, position = "jitter") +
    geom_smooth() +
    facet_wrap(~ measure) +
    ylab("") + xlab("# Keywords") +
    guides(color=guide_legend(title=""), linetype=guide_legend(title="")) +
    scale_color_manual(values = cbPalette) +
    ylim(0,1) +
    plot_theme
ggsave(filename = '../paper/figures/evaluation_prec_rec.png', width = p_width, 
       height = 0.5 * p_width, dpi = 300)
ggsave(filename = '../presentation/figures/evaluation_prec_rec.png', width = p_width, 
       height = 0.5 * p_width, dpi = 300)


ggplot(filter(df, !is.element(measure, c('precision', 'recall', 'f1', 'timeline_similarity'))), 
              aes(x = iteration, y = value, color = method, linetype = method)) +
    #geom_point(alpha = 0.6, isze = 0.2, position = "jitter") +
    geom_smooth() +
    facet_wrap(~ measure) +
    ylab("") + xlab("# Keywords") +
    guides(color=guide_legend(title=""), linetype=guide_legend(title="")) +
    scale_color_manual(values = cbPalette) +
    ylim(0,1) +
    plot_theme
ggsave(filename = '../paper/figures/evaluation_similarity.png', width = p_width, 
       height = 0.5 * p_width, dpi = 300)
ggsave(filename = '../presentation/figures/evaluation_similarity.png', width = p_width, 
       height = 0.5 * p_width, dpi = 300)


ggplot(filter(df, !is.element(measure, c('timeline_similarity'))), 
              aes(x = iteration, y = value, color = method)) +
    geom_line(aes(group = replication), alpha = 0.6, size = 0.2, position = "jitter") +
    geom_smooth() +
    facet_wrap(~ method + measure, nrow = 4) +
    ylab("") + xlab("# Keywords") +
    guides(color=FALSE) +
    scale_color_manual(values = cbPalette) +
    ylim(0,1) +
    scale_y_continuous(breaks=c(0, 0.5, 1)) +
    theme(strip.text = element_text(size=2)) +
    plot_theme
ggsave(filename = '../paper/figures/evaluation_detail.png', width = p_width, 
       height = p_width, dpi = 150)
ggsave(filename = '../presentation/figures/evaluation_detail.png', width = p_width, 
       height = p_width, dpi = 150)


# Some stats for the discussion
stats <- 
    group_by(filter(df, measure == "recall", (iteration == 0 | iteration == 99)), method, measure, iteration) %>%
    summarize(min = min(value),
              lo = quantile(value, 0.025),
              avg = mean(value), 
              median = median(value),
              hi = quantile(value, 0.975),
              max = max(value)) %>%
    as.data.frame() %>%
    print()


# Boolean vs ML

df <- read_csv('../data/boolean_vs_clf.csv') 
df$benchmark <- 0
df$benchmark[df$measure == "precision"] <- 0.81
df$benchmark[df$measure == "recall"] <- 0.5
df$benchmark[df$measure == "f1"] <- 0.62

ggplot(filter(df, measure == 'f1')) + 
    geom_point(aes(x = iteration, y = value, group = replication), 
               size = 0.5, alpha = 0.3) +
    geom_smooth(aes(x = iteration, y = value)) +
    geom_hline(aes(yintercept = benchmark), linetype = 2) +
    ylim(0,1) + ylab("F1 Score") + xlab("Number of Keywords") +
    geom_text(aes(x = 10, y = 0.65), label = "Classifier Score", color="grey40") +
    plot_theme
 ggsave(filename = '../paper/figures/bool_vs_clf.png', width = p_width, 
       height = p_width, dpi = 301)
 