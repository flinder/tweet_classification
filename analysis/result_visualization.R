# This script produces all visualizations for the paper and presentation

library(reshape2)
library(tidyverse)
library(xtable)

source('plot_theme.R')

PRES_FIGURES = '../presentation/figures/'
PAPER_FIGURES = '../paper/figures/'
RESULT_DIR = '../data/results/'

# ==============================================================================
# Boolean vs ML
# ==============================================================================

df <- read_csv(paste0(RESULT_DIR, 'boolean_vs_clf.csv'))

benchmark <- data_frame('value' = c(0.81, 0.5),
                        'measure' = c('precision', 'recall'),
                         'label' = rep('Classifier Score', 2))
                        
# Bool vs clf for presentation
ggplot(filter(df, measure != 'f1')) + 
    geom_line(aes(x = iteration, y = value, group = replication), 
               size = 0.5, alpha = 0.3) +
    geom_smooth(aes(x = iteration, y = value)) +
    geom_hline(data = filter(benchmark, measure != 'f1'), 
               aes(yintercept = value), linetype = 2) +
    facet_wrap(~measure) +
    ylim(0,1) + ylab("F1 Score") + xlab("Number of Keywords") +
    plot_theme
ggsave(filename = paste0(PRES_FIGURES, 'bool_vs_clf.png'), width = p_width, 
       height = 0.5*p_width, dpi = 301)

# Bool vs clf for presentation
ggplot(df) + 
    geom_line(aes(x = iteration, y = value, group = replication), 
               size = 0.5, alpha = 0.3) +
    geom_smooth(aes(x = iteration, y = value)) +
    geom_hline(data=benchmark, aes(yintercept = value), linetype = 2) +
    facet_wrap(~measure) +
    ylim(0,1) + ylab("F1 Score") + xlab("Number of Keywords") +
    #geom_text(aes(x = 10, y = 0.65), label = "Classifier Score", color="grey40") +
    plot_theme
ggsave(filename = paste0(PAPER_FIGURES, 'bool_vs_clf.png'), width = p_width, 
       height = 0.5*p_width, dpi = 301)
 

# ==============================================================================
# Main experiment results
# ==============================================================================

df <- read_csv(paste0(RESULT_DIR, 'experiment_results_lasso_automatic.csv'))
 
# Relabel
df$method[df$method == "baseline"] <- "Keyword"
df$method[df$method == "expansion"] <- "Expansion"
df$method[df$method == "active"] <- "Expansion + Active ML"
df$method[df$method == "random"] <- "Expansion + Random ML"


# For paper
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
ggsave(filename = paste0(PAPER_FIGURES, 'evaluation_prec_rec.png'), width = p_width, 
       height = 0.5 * p_width, dpi = 300)

# For presentation
ggplot(filter(df, is.element(measure, c('precision', 'recall'))), 
              aes(x = iteration, y = value, color = method, linetype = method)) +
    #geom_point(alpha = 0.6, isze = 0.2, position = "jitter") +
    geom_smooth() +
    facet_wrap(~ measure) +
    ylab("") + xlab("# Keywords") +
    guides(color=guide_legend(title=""), linetype=guide_legend(title="")) +
    scale_color_manual(values = cbPalette) +
    ylim(0,1) +
    plot_theme
ggsave(filename = paste0(PRES_FIGURES, 'evaluation_prec_rec.png'), width = p_width, 
       height = 0.5 * p_width, dpi = 300)

ggplot(filter(df, !is.element(measure, c('precision', 'recall', 'f1'))), 
              aes(x = iteration, y = value, color = method, linetype = method)) +
    #geom_point(alpha = 0.6, isze = 0.2, position = "jitter") +
    geom_smooth() +
    facet_wrap(~ measure) +
    ylab("") + xlab("# Keywords") +
    guides(color=guide_legend(title=""), linetype=guide_legend(title="")) +
    scale_color_manual(values = cbPalette) +
    ylim(0,1) +
    plot_theme
ggsave(filename = paste0(PAPER_FIGURES, 'evaluation_similarity.png'), width = p_width, 
       height = 0.5 * p_width, dpi = 300)
ggsave(filename = paste0(PRES_FIGURES, 'evaluation_similarity.png'), width = p_width, 
       height = 0.5 * p_width, dpi = 300)


ggplot(filter(df, !is.element(measure, c('timeline_similarity'))), 
              aes(x = iteration, y = value, color = method)) +
    geom_line(aes(group = replication), alpha = 0.2, size = 0.2, position = "jitter") +
    facet_wrap(~ method + measure, nrow = 4) +
    ylab("") + xlab("# Keywords") +
    guides(color=FALSE) +
    scale_color_manual(values = cbPalette) +
    ylim(0,1) +
    scale_y_continuous(breaks=c(0, 0.5, 1)) +
    theme(strip.text = element_text(size=2)) +
    plot_theme
ggsave(filename = paste0(PAPER_FIGURES, 'evaluation_detail.png'), width = p_width, 
       height = p_width, dpi = 150)
ggsave(filename = paste0(PRES_FIGURES, 'evaluation_detail.png'), width = p_width, 
       height = p_width, dpi = 150)


# ==============================================================================
# Comparing different expansion methods
# ==============================================================================

df_lasso <- read_csv(paste0(RESULT_DIR, 'experiment_results_lasso_automatic.csv'))
df_monroe <- read_csv(paste0(RESULT_DIR, 'experiment_results_monroe_automatic.csv'))
df_king <- read_csv(paste0(RESULT_DIR, 'experiment_results_king_automatic.csv'))

df <- inner_join(df_lasso, df_monroe, by = c('replication', 'iteration', 
                                             'measure', 'method'),
                 suffix = c(".lasso", ".monroe")) %>%
    inner_join(df_king, by = c('replication', 'iteration', 'measure', 'method')) %>%
    rename(value.king = value) %>%
    filter(method == "expansion") %>%
    melt(id = c('replication', 'iteration', 'measure', 'method')) %>%
    mutate(method = sapply(strsplit(as.character(variable), '\\.'), 
                           function(x) x[2])) %>%
    select(-variable) %>%
    tbl_df()
    
ggplot(df, aes(x = iteration, y = value, color = method, linetype = method)) +
    geom_smooth() +
    xlab("# Keywords") + ylab("") + 
    scale_color_manual(values = cbPalette) +
    guides(color = guide_legend(title = NULL), 
           linetype = guide_legend(title = NULL)) +
    facet_wrap(~measure) +
    plot_theme
ggsave(filename = paste0(PAPER_FIGURES, 'qe_method_comparison.png'), width = p_width, 
       height = 2/3 * p_width, dpi = 150)
ggsave(filename = paste0(PRES_FIGURES, 'qe_method_comparison.png'), width = p_width, 
       height = 2/3 * p_width, dpi = 150)
   
# ==============================================================================
# Timeline Plots for presentation
# ==============================================================================

timelines <- read_csv(paste0(RESULT_DIR, 'timelines.csv'))


# Build up plots
selected_tls <- c(146, 382, 254, 469, 103, 0:24)
max_alpha <- 0.3
min_alpha <- 0.1
coef <- (max_alpha - min_alpha) / length(selected_tls)
for(i in 1:length(selected_tls)) {
    print(i)
    alpha <- max_alpha -coef * i
    selected <- selected_tls[1:i]
    highlighted <- selected[length(selected)]
    pdat <- filter(timelines, iteration %in% selected) %>%
        mutate(hl = factor(ifelse(iteration == highlighted, 'a', 'b'), levels = c('a', 'b')))
    p <- ggplot(pdat, aes(x = date, y = proportion, color = hl, alpha = hl,
                     group = iteration)) +
        scale_y_continuous(labels = scales::percent, limits = c(0, 0.05)) +
        geom_line() +
        ylab('Percent Relevant') + xlab("Date") +
        scale_color_manual(values = c(cbPalette[2], cbPalette[1]), guide = F) +
        scale_alpha_manual(values = c(1, alpha), guide = F) +
        plot_theme
    ggsave(p, filename = paste0(PRES_FIGURES, 'timeline_', i, '.png'), 
           width = p_width, height = 0.5 * p_width)
       
}

# Full Plot
p <- ggplot(timelines, aes(x = date, y = proportion, group = iteration)) +
    scale_y_continuous(labels = scales::percent, limits = c(0, 0.05)) +
    geom_line(alpha = 0.01) +
    ylab('Percent Relevant') + xlab("Date") +
    scale_color_manual(values = c(cbPalette[2], cbPalette[1]), guide = F) +
    scale_alpha_manual(values = c(1, min_alpha), guide = F) +
    plot_theme
ggsave(p, filename = paste0(PRES_FIGURES, 'timeline_full.png'), 
       width = p_width, height = 0.5 * p_width)


# Crowdflower word table
kw <- read_csv('../data/survey_data/crowdflower_keywords.csv')
kw$translation <- NA

tab <- xtable(kw, digits = 2, 
              caption = "List of keywords suggested by survey participants.",
              label = "tab:cf_keywords")
print(tab, file = '../paper/tables/cf_keywords_long.tex', include.rownames = FALSE)

tab <- xtable(kw[1:20,], digits = 2, 
              caption = "List of keywords suggested by survey participants.",
              label = "tab:cf_keywords")
print(tab, file = '../paper/tables/cf_keywords_short.tex', include.rownames = FALSE)


stop()
# ==============================================================================
# Miscelaneous Stuff
# ==============================================================================

# Some stats for the discussion of the results
stats <- 
    group_by(filter(df, measure == "recall", 
                    (iteration == 0 | iteration == 99)), 
             method, measure, iteration) %>%
    summarize(min = min(value),
              lo = quantile(value, 0.025),
              avg = mean(value), 
              median = median(value),
              hi = quantile(value, 0.975),
              max = max(value)) %>%
    as.data.frame() %>%
    print()

