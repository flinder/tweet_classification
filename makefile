DTM_DIR=data/dtms/
RES_DIR=data/results/
FIG_DIR=paper/figures/

#Data preprocessing
$DTM_DIR/df.p $DTM_DIR/dtm_hashtags.p $DTM_DIR/dtm_non_normalized.p \
$DTM_DIR/dtm_users.p $DTM_DIR/ground_truth.p $DTM_DIR/kwords.p \
$DTM_DIR/non_norm_terms.p $DTM_DIR/norm_terms.p:
	cd experiment; \
	python data_preprocessing.py

# Bool vs clf simulation
$RES_DIR/boolean_vs_clf.csv:
	cd experiment; \
	python bool_vs_clf.py

# Run the main experiment with the three different expansion methods
$RES_DIR/experiment_results_lasso_automatic.csv: 
	cd experiment; \
	python main_experiment.py lasso
$RES_DIR/experiment_results_king_automatic.csv: 
	cd experiment; \
	python main_experiment.py king
$RES_DIR/experiment_results_monroe_automatic.csv: 
	cd experiment; \
	python main_experiment.py monroe

# Make the plots
$FIG_DIR/bool_vs_clf.png $FIG_DIR/evaluation_prec_rec.png \
$FIG_DIR/evaluation_similarity.png $FIG_DIR/evaluation_detail.png:
	Rscript analysis/result_visualization.R

#TODO: active_vs_random graphic for appendix
