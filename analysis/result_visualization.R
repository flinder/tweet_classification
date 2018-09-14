# This script produces all visualizations for the paper and presentation

library(reshape2)
library(tidyverse)
library(xtable)
pe = flindR::plot_elements()

PAPER_FIGURES = '../paper/figures/'
RESULT_DIR = '../data/results/'


# ==============================================================================
# Main experiment results
# ==============================================================================

df <- read_csv(paste0(RESULT_DIR, 'v3_aci_experiment_results_lasso_automatic.csv'))
#df <- read_csv(paste0(RESULT_DIR, 'v3_experiment_results_lasso_automatic.csv'))
 
# Relabel
df$method[df$method == "baseline"] <- "Survey Keywords Only"
df$method[df$method == "expansion"] <- "Lasso Expansion Only"
df$method[df$method == "active"] <- "Expansion + Active ML"
df$method[df$method == "klr"] <- "KLR Expansion Only"

df$measure[df$measure == 'f1'] = 'F-1 Score'
df$measure[df$measure == 'hashtag_similarity'] = 'Hashtag Similarity'
df$measure[df$measure == 'precision'] = 'Precision'
df$measure[df$measure == 'recall'] = 'Recall'
df$measure[df$measure == 'timeline_similarity'] = 'Timeline Similarity'
df$measure[df$measure == 'user_similarity'] = 'User Similarity'
df$measure = factor(df$measure, levels = c('F-1 Score', 'Precision', 'Recall',
                                           'Hashtag Similarity', 
                                           'User Similarity', 
                                           'Timeline Similarity'))

# Main resultsfigure for the paper
filter(df, measure %in% c('Precision', 'Recall')) %>%
ggplot(aes(x = iteration, y = value, color = method, linetype = method)) +
    #geom_point(alpha = 0.1, size = 0.2, position = "jitter") +
    geom_smooth() +
    facet_wrap(~ measure, nrow = 3) +
    ylab("") + xlab("# Keywords") +
    guides(color=guide_legend(title=""), linetype=guide_legend(title="")) +
    scale_color_manual(values = pe$colors) +
    ylim(0,1) +
    pe$theme
ggsave(filename = paste0(PAPER_FIGURES, 'evaluation_prec_rec.png'), width = pe$p_width, 
       height = pe$p_width, dpi = 300)

# Detailed evaluation (all measures) for appendix
ggplot(filter(df, method != "Expansion + Random ML"), 
              aes(x = iteration, y = value, color = method)) +
    geom_line(aes(group = replication), alpha = 0.5, size = 0.2, position = "jitter") +
    facet_wrap(~ method + measure, nrow = 4) +
    ylab("") + xlab("# Keywords") +
    guides(color=FALSE) +
    scale_color_manual(values = cbPalette) +
    scale_y_continuous(breaks=c(0, 0.5, 1), limits = c(0,1)) +
    theme(strip.text = element_text(size=2)) +
    plot_theme
ggsave(filename = paste0(PAPER_FIGURES, 'evaluation_detail.png'), width = p_width, 
       height = p_width, dpi = 150)

  
# ==============================================================================
# Timeline Plots for presentation
# ==============================================================================

#timelines <- read_csv(paste0(RESULT_DIR, 'timelines.csv'))
#
#
## Build up plots
#selected_tls <- c(146, 382, 254, 469, 103, 0:10)
#max_alpha <- 0.3
#min_alpha <- 0.1
#coef <- (max_alpha - min_alpha) / length(selected_tls)
#for(i in 1:length(selected_tls)) {
#    print(i)
#    alpha <- max_alpha -coef * i
#    selected <- selected_tls[1:i]
#    highlighted <- selected[length(selected)]
#    pdat <- filter(timelines, iteration %in% selected) %>%
#        mutate(hl = factor(ifelse(iteration == highlighted, 'a', 'b'), levels = c('a', 'b')))
#    p <- ggplot(pdat, aes(x = date, y = proportion, color = hl, alpha = hl,
#                     group = iteration)) +
#        scale_y_continuous(labels = scales::percent, limits = c(0, 0.05)) +
#        geom_line() +
#        ylab('Percent Relevant') + xlab("Date") +
#        scale_color_manual(values = c(cbPalette[2], cbPalette[1]), guide = F) +
#        scale_alpha_manual(values = c(1, alpha), guide = F) +
#        plot_theme
#    ggsave(p, filename = paste0(PRES_FIGURES, 'timeline_', i, '.png'), 
#           width = p_width, height = 0.5 * p_width)
#}
#
## Full Plot
#p <- ggplot(timelines, aes(x = date, y = proportion, group = iteration)) +
#    scale_y_continuous(labels = scales::percent, limits = c(0, 0.05)) +
#    geom_line(alpha = 0.01) +
#    ylab('Percent Relevant') + xlab("Date") +
#    scale_color_manual(values = c(cbPalette[2], cbPalette[1]), guide = F) +
#    scale_alpha_manual(values = c(1, min_alpha), guide = F) +
#    plot_theme
#ggsave(p, filename = paste0(PRES_FIGURES, 'timeline_full.png'), 
#       width = p_width, height = 0.5 * p_width)
#
## Full Plot with classifier ground truth
#
### Get the counts of from the database (this requires code from the second
### chapter)
#con <- src_postgres(dbname = 'dissertation')
#tweets_db <- tbl(con, 'tweets')
#users_db <- tbl(con, 'users')
#
## Plot of over time distributin of tweets
#q_res <- filter(tweets_db, data_group %in% c("de_panel", "de_panel_extended")) %>%
#    left_join(users_db, by = c("user_id" = "id"), suffix = c(".tweets", 
#                                                             ".users")) %>%
#    mutate(relevant = ifelse(classification_keyword == 'relevant', 1, 0)) %>%
#    select(created_at.tweets, relevant) %>%
#    tbl_df() %>%
#    filter(created_at.tweets >= as.Date('2015/01/01'), 
#           created_at.tweets < as.Date('2016/01/01'))
#
#pdat <- mutate(q_res, day = yday(created_at.tweets)) %>%
#    group_by(day) %>%
#    summarize(n_relevant = sum(relevant), n = n(), 
#              percentage_relevant = sum(relevant) / n()) %>% 
#    mutate(date = seq(as.Date("2015/01/01"), as.Date("2015/12/31"), 
#                      by = "day"))
#
#dat <- rbind(timelines, data_frame(date = pdat$date, iteration = 500,
#                                   proportion = pdat$percentage_relevant))
#dat$highlight <- dat$iteration == 500
#
#p <- ggplot(dat, aes(x = date, y = proportion, group = iteration,
#                color = highlight, alpha = highlight)) +
#    scale_y_continuous(labels = scales::percent, limits = c(0, 0.05)) +
#    geom_line() +
#    ylab('Percent Relevant') + xlab("Date") +
#    scale_color_manual(values = c(cbPalette[1], cbPalette[7]), guide = F) +
#    scale_alpha_manual(values = c(0.01, 1), guide = F) +
#    plot_theme
#ggsave(p, filename = paste0(PRES_FIGURES, 'timeline_full_clf.png'), 
#       width = p_width, height = 0.5 * p_width)
#
## Same thing with average of all others (non clf)
#pdat <- group_by(dat, date, highlight) %>% 
#    summarize(proportion = mean(proportion))
#
#p <- ggplot(pdat, aes(x = date, y = proportion, color = highlight)) +
#    scale_y_continuous(labels = scales::percent, limits = c(0, 0.05)) +
#    geom_line() +
#    ylab('Percent Relevant') + xlab("Date") +
#    scale_color_manual(values = c(cbPalette[1], cbPalette[7]), guide = F) +
#    plot_theme
#ggsave(p, filename = paste0(PRES_FIGURES, 'timeline_full_means.png'), 
#       width = p_width, height = 0.5 * p_width)
#
## Difference
#pdat_diff <- data_frame(date = filter(pdat, highlight)$date,
#                        diff = filter(pdat, !highlight)$proportion - 
#                               filter(pdat, highlight)$proportion) %>%
#    mutate(true = filter(pdat, highlight)$proportion,
#           bias = diff / true)
#
#p <- ggplot(pdat_diff) +
#    geom_line(aes(x = date, y = bias)) + 
#    ylab("Bias (survey - true) / true") + xlab('Date') +
#    plot_theme
#ggsave(p, filename = paste0(PRES_FIGURES, 'timeline_full_mean_bias.png'), 
#       width = p_width, height = 0.5 * p_width)
#
#p <- ggplot(pdat_diff) +
#    geom_line(aes(x = date, y = diff * 100)) + 
#    ylab("% Difference (survey - true)") + xlab('Date') +
#    plot_theme
#ggsave(p, filename = paste0(PRES_FIGURES, 'timeline_full_mean_diff.png'), 
#       width = p_width, height = 0.5 * p_width)
#
#
## Crowdflower word table
#kw <- read_csv('../data/survey_data/crowdflower_keywords.csv')
#kw$translation <- NA
#
#tab <- xtable(kw, digits = 2, 
#              caption = "List of keywords suggested by survey participants.",
#              label = "tab:cf_keywords")
#print(tab, file = '../paper/tables/cf_keywords_long.tex', include.rownames = FALSE)
#
#tab <- xtable(kw[1:20,], digits = 2, 
#              caption = "List of keywords suggested by survey participants.",
#              label = "tab:cf_keywords")
#print(tab, file = '../paper/tables/cf_keywords_short.tex', include.rownames = FALSE)
#
#
#stop()
## ==============================================================================
## Miscelaneous Stuff
## ==============================================================================
#
## Some stats for the discussion of the results
#stats <- 
#    group_by(filter(df, measure == "recall", 
#                    (iteration == 0 | iteration == 99)), 
#             method, measure, iteration) %>%
#    summarize(min = min(value),
#              lo = quantile(value, 0.025),
#              avg = mean(value), 
#              median = median(value),
#              hi = quantile(value, 0.975),
#              max = max(value)) %>%
#    as.data.frame() %>%
#    print()
#
