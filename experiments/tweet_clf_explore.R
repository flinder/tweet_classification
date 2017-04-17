library(ggplot2)
library(dplyr)

df = tbl_df(read.csv('scores.csv')) %>%
    mutate(iter_clf = paste0(clf, '_', iteration))


# F1
ggplot(df, aes(x = train_size, y = f1, color = clf,
               fill = clf)) + 
    geom_line(aes(group = factor(iter_clf)), 
                se = FALSE, alpha = 0.05,
                size = 0.3, stat='smooth', method='loess') +
    geom_smooth() +
    theme_bw()


# Precision
ggplot(df, aes(x = train_size, y = precision, color = clf,
               fill = clf)) + 
    geom_line(aes(group = factor(iter_clf)), 
                se = FALSE, alpha = 0.05,
                size = 0.3, stat='smooth', method='loess') +
    geom_smooth() +
    theme_bw()


# Recall
ggplot(df, aes(x = train_size, y = recall, color = clf,
               fill = clf)) + 
    geom_line(aes(group = factor(iter_clf)), 
                se = FALSE, alpha = 0.05,
                size = 0.3, stat='smooth', method='loess') +
    geom_smooth() +
    theme_bw()