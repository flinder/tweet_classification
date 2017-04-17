library(dplyr)
library(ggplot2)

db <- src_postgres(dbname = "dissertation")
annotation_data <- tbl(db, from = "cr_results")

# Countries
trust_by_country <- group_by(annotation_data, country) %>%
    summarize(average_trust = mean(trust),
              n = n()) %>%
    arrange(average_trust, n) %>%
    tbl_df()

ggplot(trust_by_country) +
    geom_point(aes(y = factor(country, levels = unique(country)), 
                   x = average_trust, size = n)) + 
    theme_bw()

# Individual contributors
n_by_contibutor <- group_by(annotation_data, worker_id) %>%
    summarize(count = n(),
              trust = mean(trust)) %>%
    arrange(count, trust) %>%
    mutate(worker_id = as.character(worker_id)) %>%
    tbl_df()

ggplot(n_by_contibutor) +
    geom_point(aes(x = count, 
                   y = factor(worker_id, levels = unique(worker_id)),
                   size = trust), alpha = 0.6) +
    theme_bw()
   

agreement <- function(x) {
    
}

# Get aggregate judgments 
tweets <- mutate(annotation_data, 
                 )
    group_by(annotation_data, tweet_id) %>%
    summarize()