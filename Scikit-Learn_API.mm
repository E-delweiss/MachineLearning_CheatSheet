<map version="1.0.0"><node Text="Carte heuristique"><node ID="FE0BA5E2-1A03-4603-81B4-1F6610658944" BACKGROUND_COLOR="#FFFFFF" TEXT="The methode &#8216;fit&#8217; is composed of 2 elements : 
(i) a learning algorthm
(ii) some model states" COLOR="#4B4B4B" POSITION="right" STYLE="bubble"><edge COLOR="#4B4B4B" WIDTH="4" /><font NAME="Helvetica-Oblique" SIZE="16" ITALIC="true" /></node>
<node ID="698B2B40-23F3-4C5F-BC22-CE7103172CA7" BACKGROUND_COLOR="#FFFFFF" TEXT="scikit-learn convention: If an attribute is learned from the data, its name ends with an underscore, as in mean_ and scale_ for the StandardScaler." COLOR="#4B4B4B" POSITION="right" STYLE="bubble"><edge COLOR="#4B4B4B" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="16" BOLD="true" /></node>
<node ID="3ECD9FBE-DCD0-4876-8734-48061D04C343" BACKGROUND_COLOR="#FFFFFF" TEXT="CrossValidation:
Note that by computing the std of the cross-validation scores, we can estimate the uncertainty of our model statistical performance. This is the main advantage of cross-validation and can be crucial in practice, for example when comparing different models to figure out whether one is better than the other or whether the statistical performance differences are within the uncertainty." COLOR="#4B4B4B" POSITION="right" STYLE="bubble"><edge COLOR="#4B4B4B" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="16" BOLD="true" /></node>
<node ID="C349CA4E-AC73-4347-A517-6C39D05AF453" BACKGROUND_COLOR="#FFFFFF" TEXT="Metrics:
Score : higher values mean better results
Error : lower values mean better results

In scikit-learn, any error can be transformed into a score to be used in cross_validate. To do so, we need to pass a string of the error metric with an additional &#8216;neg_&#8217; string at the front to the parameter scoring; for instance scoring=&#34;neg_mean_absolute_error&#34;." COLOR="#4B4B4B" POSITION="right" STYLE="bubble"><edge COLOR="#4B4B4B" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="16" BOLD="true" /></node>
<node ID="6DA45BE3-E1E9-4E2E-ADA9-B5E2E3493D95" BACKGROUND_COLOR="#FFFFFF" TEXT="Regularization :
Scaling the data before fitting a model is necessary when using a regularized model. " COLOR="#4B4B4B" POSITION="right" STYLE="bubble"><edge COLOR="#4B4B4B" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="16" BOLD="true" /></node>
<node ID="C6161683-8B31-40E4-9F21-EE367E55809A" BACKGROUND_COLOR="#FFFFFF" TEXT="Scoring:
To compute the score, the predictor first computes the predictions (using the predict method) and then uses a scoring function to compare the true target &#8216;y&#8217; and the predictions. Finally, the score is returned" COLOR="#4B4B4B" POSITION="right" STYLE="bubble"><edge COLOR="#4B4B4B" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="16" BOLD="true" /></node>
<node ID="2DB7304A-1948-4EC4-AE79-5658B8815C76" BACKGROUND_COLOR="#FF2600" TEXT="Scikit-Learn API" COLOR="#FFFFFF" POSITION="right" STYLE="bubble"><edge COLOR="#4B4B4B" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="40" BOLD="true" /><node ID="3288DA45-C2E9-4441-9CAB-912895EC15A7" BACKGROUND_COLOR="#FFFFFF" TEXT="Preprocessing" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#000000" WIDTH="6" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="FA213BFF-336D-4197-A8F6-3E4563016F85" BACKGROUND_COLOR="#FFFFFF" TEXT="StandardScaler" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="5" /><font NAME="Helvetica-Bold" SIZE="20" BOLD="true" /><node ID="F7504C84-126D-46BB-9A1E-3DEF38F05777" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data_train)
data_train_scaled = scaler.transform(data_train)
//
data_train_scaled = scaler.fit_transform(data_train)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Courier" SIZE="16" /><node ID="BD8433A1-175E-45F7-8CCC-B3738CF0789C" BACKGROUND_COLOR="#FFFFFF" TEXT="By default, the StandardScaler transformer transforms the data by centering each feature around 0.0 on average and by scaling the resulting values so that they have a std of 1.0 on the training set. In practice, this means that each feature will have 99.7% of the samples&#39; values (3 standard deviation) ranging from -3 to 3." COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Helvetica-Oblique" SIZE="18" ITALIC="true" /></node>
</node>
</node>
<node ID="DA187656-4A66-49EF-AF7B-6255BE2F272B" BACKGROUND_COLOR="#FFFFFF" TEXT="Pipeline" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#72C8FF" WIDTH="5" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="3EA7026C-8603-4858-A86D-E6FAC601D35D" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
                    StandardScaler(), 
                    LogisticRegression())" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#72C8FF" WIDTH="4" /><font NAME="Courier" SIZE="16" /></node>
</node>
<node ID="7C9C39F9-3CEB-4C2D-AC87-07AA8938BCFD" BACKGROUND_COLOR="#FFFFFF" TEXT="ColumnSelector / ColumnTransformer" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#FFCD3B" WIDTH="5" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="50476D8D-5937-408D-8435-1A4840592546" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer

cat_columns_select = selector(dtype_include=object)
categorical_columns = cat_columns_select(data)
preprocessor = ColumnTransformer([
    (&#39;cat-preprocessor&#39;, categorical_preprocessor, 
    categorical_columns)], remainder=&#39;passthrough&#39;,  
    sparse_threshold=0)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#FFCD3B" WIDTH="4" /><font NAME="Courier" SIZE="16" /></node>
</node>
<node ID="C9ADEB71-8E87-41D8-ACB4-12D36465395F" BACKGROUND_COLOR="#FFFFFF" TEXT="Encoder" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#FF5E69" WIDTH="5" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="CD806776-1177-4BA9-8632-DB942AC4A14E" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OrdinalEncoder()
encoder = OneHoteEncoder(sparse=False)
data_encoded = encoder.fit_transform(data_categorical)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#FF5E69" WIDTH="4" /><font NAME="Courier" SIZE="16" /><node ID="72BB4F93-9890-4C5B-85CF-713D4A5D1560" BACKGROUND_COLOR="#FFFFFF" TEXT="Sparse matrices are efficient data structures when most of your matrix elements are zero." COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#FF5E69" WIDTH="4" /><font NAME="Helvetica" SIZE="18" /><node ID="070B646D-D681-4A91-9F4B-8340F887ECBF" BACKGROUND_COLOR="#FFFFFF" TEXT="handle_unknown : it can be set to &#8216;use_encoded_value&#8217; and by setting &#8216;unknown_value&#8217; to handle rare categories." COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#FF5E69" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="18" BOLD="true" /></node>
</node>
</node>
</node>
<node ID="CC3EB2BC-854E-4611-98FE-E68FC44F220C" BACKGROUND_COLOR="#FFFFFF" TEXT="Imputer" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#64C8CD" WIDTH="5" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="36EE873E-F386-42C5-9D74-B16ED2A81CF4" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy=&#39;mean&#39;)
imp_mean.fit(data)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#64C8CD" WIDTH="4" /><font NAME="Courier" SIZE="16" /></node>
</node>
</node>
<node ID="6D1C5C67-E1D4-4EC2-88F9-278333A06E13" BACKGROUND_COLOR="#FFFFFF" TEXT="Training / Scoring" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#000000" WIDTH="6" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="061C7BF6-EC5D-4C0C-99A1-D4CAEDE37325" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(data_train, target_train)
accuracy = model.score(data_test, target_test)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Courier" SIZE="16" /><node ID="3B31ACF6-E8DC-4671-9B80-07B98D079607" BACKGROUND_COLOR="#FFFFFF" TEXT="CrossValidation" COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="18" BOLD="true" /><node ID="27676197-AC2E-4A97-BDCE-B93B9EB73254" BACKGROUND_COLOR="#FFFFFF" TEXT="K-Fold" COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="18" BOLD="true" /><node ID="80F095E5-9206-4D26-862D-701FAA08B4DC" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.model_selection import cross_validate

cv_result = cross_validate(model, features, target, cv=5)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Courier" SIZE="16" /><node ID="A9BE40B0-A8AA-4A23-9E29-481176F519D2" BACKGROUND_COLOR="#FFFFFF" TEXT="Cross_validate takes a parameter cv which defines the splitting strategy.
Setting cv=5 created 5 distinct splits to get 5 variations for the training and testing sets. Each training set is used to fit one model which is then scored on the matching test set." COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Helvetica" SIZE="18" /></node>
</node>
</node>
<node ID="53107242-0B3A-4922-9DEB-D2299E14057C" BACKGROUND_COLOR="#FFFFFF" TEXT="ShuffleSplit" COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="18" BOLD="true" /><node ID="4BFAB41F-1BC4-4545-8B84-403276626663" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.model_selection import cross_validate, ShuffleSplit

cvShuffle = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
cv_results = cross_validate(
                        model, data, target, cv=cvShuffle, 
                        scoring=&#34;neg_mean_absolute_error&#34;)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Courier" SIZE="16" /></node>
</node>
<node ID="718B048A-85A6-4418-93EE-D4B990DFF168" BACKGROUND_COLOR="#FFFFFF" TEXT="Other K-Fold strategies" COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="18" BOLD="true" /><node ID="B016CA12-1829-497F-8AF2-DB754BB8A67F" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import TimeSeriesSplit

cv = StratifiedKFold(n_splits=3)
cv = GroupKFold()
...

groups = quotes.index.to_period(&#34;Q&#34;)
cv = LeaveOneGroupOut()
cv = TimeSeriesSplit(n_splits=groups.nunique())
test_score = cross_val_score(regressor, data, target,
                           cv=cv, groups=groups, n_jobs=2)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Courier" SIZE="16" /></node>
</node>
<node ID="1031B9F9-9D4D-4EDD-A096-2B91DF258328" BACKGROUND_COLOR="#FFFFFF" TEXT="Nested cross-validation" COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="18" BOLD="true" /><node ID="BAF35850-A1DF-452C-9A1A-5E29B61BE014" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.model_selection import cross_val_score, KFold

# Declare the inner and outer cross-validation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)

# Inner cross-validation for parameter search
model = GridSearchCV(estimator=model_to_tune, 
                    param_grid=param_grid, cv=inner_cv, n_jobs=2)

# Outer cross-validation to compute the testing score
test_score = cross_val_score(model, data, target, cv=outer_cv, n_jobs=2)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#FF9559" WIDTH="4" /><font NAME="Courier" SIZE="16" /></node>
</node>
</node>
</node>
</node>
<node ID="F1D65E4F-E63A-42A0-87FD-205F1395C834" BACKGROUND_COLOR="#FFFFFF" TEXT="Curves" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#000000" WIDTH="6" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="CB4B8377-5DDB-44BB-83F4-F646EC0F62AC" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve&#8232;&#8232;max_depth = [1, 5, 10, 15, 20, 25]
train_scores, test_scores = validation_curve(
    model, data, target, param_name=&#34;max_depth&#34;, 
    param_range=max_depth, cv=cv, 
    scoring=&#34;neg_mean_absolute_error&#34;, n_jobs=2)

train_sizes = np.linspace(0.1, 1.0, num=5, endpoint=True)
results = learning_curve(
    regressor, data, target, train_sizes=train_sizes, cv=cv,
    scoring=&#34;neg_mean_absolute_error&#34;, n_jobs=2)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#008F51" WIDTH="5" /><font NAME="Courier" SIZE="16" /></node>
<node ID="A797217F-9474-468F-B9D6-78D44EC8FA34" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.metrics import plot_precision_recall_curve

disp = plot_precision_recall_curve(
    classifier, data_test, target_test, pos_label=&#39;donated&#39;,
    marker=&#34;+&#34;)
_ = disp.ax_.set_title(&#34;Precision-recall curve&#34;)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#521B92" WIDTH="5" /><font NAME="Courier" SIZE="16" /><node ID="8DBE2341-14A0-47E5-AE48-0944F9A5D678" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.metrics import plot_roc_curve

disp = plot_roc_curve(
    classifier, data_test, target_test, pos_label=&#39;donated&#39;,
    marker=&#34;+&#34;)
disp = plot_roc_curve(
    dummy_classifier, data_test, target_test, pos_label=&#39;donated&#39;,
    color=&#34;tab:orange&#34;, linestyle=&#34;--&#34;, ax=disp.ax_)
_ = disp.ax_.set_title(&#34;ROC AUC curve&#34;)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#521B92" WIDTH="5" /><font NAME="Courier" SIZE="16" /></node>
</node>
</node>
<node ID="97234A64-AF4E-4E95-A877-12D1982FB842" BACKGROUND_COLOR="#FFFFFF" TEXT="Metrics" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#000000" WIDTH="6" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="4AB0C575-A9C4-4DA5-BEC4-31C51E55A496" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#FF5E69" WIDTH="6" /><font NAME="Courier" SIZE="16" /></node>
<node ID="E0116CA9-F2E8-4546-A40F-40C16D216084" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.metrics import precision_score, recall_score

precision = precision_score(target_test, target_predicted,
                            pos_label=&#34;donated&#34;)
recall = recall_score(target_test, target_predicted, 
                            pos_label=&#34;donated&#34;)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#64C8CD" WIDTH="6" /><font NAME="Courier" SIZE="16" /><node ID="4ECF5DD7-5407-4326-9C87-6A2AB83863C5" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.metrics import balanced_accuracy_score

balanced_accuracy = balanced_accuracy_score(
                            target_test, target_predicted)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#64C8CD" WIDTH="6" /><font NAME="Courier" SIZE="16" /></node>
</node>
<node ID="312D9341-D884-4004-A4F9-2805DD237602" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#0432FF" WIDTH="6" /><font NAME="Courier" SIZE="16" /></node>
</node>
<node ID="C58113F0-0601-4A16-AC46-63B0F4341EEC" BACKGROUND_COLOR="#FFFFFF" TEXT="Hyperparameters" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#000000" WIDTH="6" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="C1385EE1-3456-412F-A21A-C1AAB818B431" BACKGROUND_COLOR="#FFFFFF" TEXT="model.get_params()
model.set_params(C=1e-3)
model.get_params()[&#39;C&#39;]" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#941100" WIDTH="5" /><font NAME="Courier" SIZE="19" /><node ID="BCFF412E-81FE-429A-B209-5F9C269D0E21" BACKGROUND_COLOR="#FFFFFF" TEXT="Fine Tuning" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#941100" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="9D42F3BC-9772-4DF6-9354-3AE9C48DFE83" BACKGROUND_COLOR="#FFFFFF" TEXT="GridSearchCV" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#941100" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="835B43C2-CF4E-4DB8-BC7D-CBC020613F02" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.model_selection import GridSearchCV

param_grid = {
    &#8216;hyperparam1: (0.05, 0.1, 0.5, 1, 5),
    &#8216;hyperparam2: (3, 10, 30, 100)}
model_grid_search = GridSearchCV(model, 
                                 param_grid=param_grid,
                                 n_jobs=2, cv=2)
model_grid_search.fit(data_train, target_train)
accuracy = model_grid_search.score(data_test, target_test)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#941100" WIDTH="5" /><font NAME="Courier" SIZE="16" /></node>
</node>
<node ID="1FFEFBC4-128E-4432-8CEA-EE2525E558A9" BACKGROUND_COLOR="#FFFFFF" TEXT="RandomizedSearchCV" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#941100" WIDTH="4" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="1C316515-CBB5-428D-978D-E0E96A44BA96" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.model_selection import RandomizedSearchCV

param_distributions = {&#8216;hyperparam1&#8217; : &#8230;.}
model_random_search = RandomizedSearchCV( model, 
                                      param_distributions=param_distributions, 
                                      n_iter=10, cv=5, verbose=1)
model_random_search.fit(data_train, target_train)
accuracy = model_random_search.score(data_test, target_test)
model_random_search.best_params_" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#941100" WIDTH="5" /><font NAME="Courier" SIZE="16" /><node ID="2EE14495-E802-483C-B5CD-096B6C7BBCA7" BACKGROUND_COLOR="#FFFFFF" TEXT="" COLOR="#AF4FC8" POSITION="right" STYLE="bubble"><edge COLOR="#941100" WIDTH="5" /><font NAME="Helvetica" SIZE="18" /><node ID="56D1E891-7830-4123-B0E6-B43BF7BC45DB" BACKGROUND_COLOR="#FFFFFF" TEXT="                                      Combining with Cross Validation

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

param_grid = {
    &#8216;hyperparam&#8217;1: (0.05, 0.1),
    &#8216;hyperparam2&#8217;: (30, 40)}
model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=2, cv=2)
cv_results = cross_validate(model_grid_search, data, target, 
                            cv=5, return_estimator=True)
scores = cv_results[&#34;test_score&#34;]" COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#941100" WIDTH="5" /><font NAME="Helvetica-Bold" SIZE="16" BOLD="true" /><node ID="787FFD56-9FEB-4B4F-A188-CDE4DEFC846C" BACKGROUND_COLOR="#FFFFFF" TEXT="The hyperparameters on each fold are potentially different since we nested the grid-search in the cross-validation. Thus, checking the variation of the hyperparameters across folds should also be analyzed.

for fold_idx, estimator in eumerate(cv_results[&#34;estimator&#34;]):
    print(f&#34;Best parameter found on fold #{fold_idx + 1}&#34;)
    print(f&#34;{estimator.best_params_}&#34;)" COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#941100" WIDTH="5" /><font NAME="Helvetica" SIZE="18" /></node>
</node>
</node>
</node>
</node>
</node>
</node>
</node>
<node ID="3D56B43E-6002-4BC5-BB86-F42126E8F824" BACKGROUND_COLOR="#FFFFFF" TEXT="Regularization" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#000000" WIDTH="6" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="0246815A-461A-4A9C-8FAC-6DF9171D0A35" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.linear_model import Ridge

ridge = Ridge(alpha=100)
cv_results = cross_validate(ridge,&#8230;)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#00FA92" WIDTH="6" /><font NAME="Courier" SIZE="19" /><node ID="80AD705D-1C69-4554-95F4-6BA1A04223D3" BACKGROUND_COLOR="#FFFFFF" TEXT="Fine Tuning" COLOR="#595959" POSITION="right" STYLE="bubble"><edge COLOR="#00FA92" WIDTH="6" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="08365149-C267-4433-AC0A-3BD80043CC26" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.linear_model import RidgeCV

alphas = np.logspace(-2, 0, num=20)
ridge = RidgeCV(alphas=alphas)
cv_results = cross_validate(ridge,&#8230;)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#00FA92" WIDTH="6" /><font NAME="Courier" SIZE="16" /></node>
</node>
</node>
</node>
<node ID="2332DDC2-104E-403B-8F18-DBFE480B46E5" BACKGROUND_COLOR="#FFFFFF" TEXT="Trees" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#000000" WIDTH="6" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="BA542E96-FB93-49A9-B978-08DC07DBD019" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeClassifier(max_depth=n)
tree.fit(data_train, target_train)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#FF89D8" WIDTH="6" /><font NAME="Courier" SIZE="19" /><node ID="994E0205-A229-4183-924D-4DBEE0F0D8D3" BACKGROUND_COLOR="#FFFFFF" TEXT="Plotting trees" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#FF89D8" WIDTH="6" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="9006D9AE-5D5C-4446-BFCC-590119E64B38" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.tree import plot_tree

_, ax = plt.subplots(figsize=(8, 6))
_ = plot_tree(tree, feature_names=[column_names],
              class_names=tree.classes_, 
              impurity=False, ax=ax)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#FF89D8" WIDTH="6" /><font NAME="Courier" SIZE="16" /></node>
</node>
</node>
</node>
<node ID="DFAF5563-B5A6-44F5-AEAD-67DAA63F9B1A" BACKGROUND_COLOR="#FFFFFF" TEXT="Chance Level" COLOR="#6D6D6D" POSITION="right" STYLE="bubble"><edge COLOR="#000000" WIDTH="6" /><font NAME="Helvetica-Bold" SIZE="22" BOLD="true" /><node ID="9F041D51-A290-4F3D-90F8-700376A6321F" BACKGROUND_COLOR="#FFFFFF" TEXT="from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import permutation_test_score

cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
regressor = DecisionTreeRegressor()

score, permutation_score, pvalue = permutation_test_score(
    regressor, data, target, cv=cv, scoring=&#34;neg_mean_absolute_error&#34;,
    n_jobs=2, n_permutations=30)
errors_permutation = pd.Series(-permutation_score, name=&#34;Permuted error&#34;)" COLOR="#AF50C8" POSITION="right" STYLE="bubble"><edge COLOR="#000000" WIDTH="6" /><font NAME="Courier" SIZE="19" /></node>
</node>
</node>
</node>
</map>
