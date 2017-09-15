DTM_DIR=data/dtms/
RES_DIR=data/results/

#Data preprocessing
$DTM_DIR/df.p $DTM_DIR/dtm_hashtags.p $DTM_DIR/dtm_non_normalized.p \
$DTM_DIR/dtm_users.p $DTM_DIR/ground_truth.p $DTM_DIR/kwords.p \
$DTM_DIR/non_norm_terms.p $DTM_DIR/norm_terms.p:
	cd experiment; \
	python data_preprocessing.py

# TODO Bool vs clf
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

# TODO: Active vs random from appendix


