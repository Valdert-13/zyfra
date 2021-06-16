import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold
import copy




class Score_model_dev():
    """
    Class builds container with predictive models based

    Parameters
    ----------

    df: pandas DataFrame
        Datafrae with model train data. Must include column with name equal to TARGET_NAME parameter

    TARGET_NAME: string
        Name of column in train dataframe with training target (0 or 1 for classification)

    models: dictionary
        Python dictionary with models like {'model_title': model_object}. Where 'model_title': string model name;
        model_object: model object with preset parameters.
        Model objects must have folowing methods: fit, predict, predict_proba

    cat_columns: list of strings
        List of categorial columns name

    KFold: integer
        Number of Stratified K folds splits and consiquently number of models
        (each SKF split is used for validation while left data is used for model training)
        Default value: 5

    random_state: integer
        random_state fix random seed for model building
        Default value: 42

    display: bool
        display report


    """

    def __init__(self,
                 df,
                 TARGET_NAME,
                 cat_columns: list = [],
                 KFold: int = 5,
                 random_state: int = 42,
                 display: bool = True,
                 models: dict = {}):

        # Vertion of model
        self.version = '1'

        self.df = df
        self.trget_name = TARGET_NAME
        self.cat_columns = cat_columns
        self.KFold = KFold
        self.random_state = random_state
        self.display = display
        self.models = []
        self.train_result = df[[TARGET_NAME]]

        for model in models.keys():
            self.models.append({'model_name': model,
                                'model_sample': models[model],
                                'folds': []})

    # Models training
    def fit(self):

        """
        Обучение моделей

        """

        for model in self.models:
            self.train_result[model['model_name']] = float("NaN")
            self.train_result[model['model_name'] + '_p'] = float("NaN")

        cv = KFold(n_splits=self.KFold, random_state=self.random_state, shuffle=True)

        X = self.df.drop([self.trget_name], axis=1)
        y = self.df[self.trget_name]

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

            x_train, x_valid = X.loc[X.index.intersection(train_idx)], X.loc[X.index.intersection(valid_idx)]
            y_train, y_valid = y.loc[y.index.intersection(train_idx)], y.loc[y.index.intersection(valid_idx)]

            if self.display:
                print(f'\nFOLD {fold + 1} REPORT')
                print('\033[4m{:<50}{:>32}{:>10}{:>12}{:>10}{:>10}\033[0m'.format('Model', 'f1 score',
                                                                                  'Recall', 'Precission', 'ROC AUC',
                                                                                  'Gini'))

            for model in self.models:
                model_name = model['model_name']

                model_to_train = copy.deepcopy(model['model_sample'])

                model_to_train.fit(x_train, y_train)

                y_pred_proba = model_to_train.predict_proba(x_valid)[:, 1]

                y_pred = np.round(y_pred_proba).astype('int')

                train_result = self.train_result.copy()

                valid = x_valid.copy()
                valid[model_name + '_p'] = y_pred_proba
                valid[model_name] = y_pred

                self.train_result.update(valid)

                del valid

                #                 mask = x_valid.index
                #                 self.train_result.loc[mask, model_name + '_p'] = y_pred_proba
                #                 self.train_result.loc[mask, model_name] = y_pred

                score = self._show_score_report(y_valid, y_pred, model)

                model['folds'].append({'fold': fold + 1, 'score': score, 'model': model_to_train})

        if self.display:
            print(f'\nFINAL REPORT')
            print('\033[4m{:<50}{:>32}{:>10}{:>12}{:>10}{:>10}\033[0m'.format('Model', 'f1 score',
                                                                              'Recall', 'Precission', 'ROC AUC',
                                                                              'Gini'))

        #         slice_index = self.df.index
        for model in self.models:
            slice_index = self.train_result[
                self.train_result[model['model_name'] + '_p'].isnull() == False].index.tolist()

            score = self._show_score_report(y_true=self.train_result[self.trget_name].loc[slice_index],
                                            y_pred=self.train_result[model['model_name'] + '_p'].loc[
                                                slice_index].round(),
                                            model=model)

    def _show_score_report(self, y_true, y_pred, model):
        f1 = round(f1_score(y_true, y_pred), 4)
        pr = round(precision_score(y_true, y_pred), 4)
        re = round(recall_score(y_true, y_pred), 4)
        roc = round(roc_auc_score(y_true, y_pred), 4) if (y_true.value_counts().shape[0] > 1) else '-'
        gini = round(roc * 2 - 1, 4) if (roc != '-') else roc

        if self.display:
            print('{:<50}{:>32}{:>10}{:>12}{:>10}{:>10}'.format(model['model_name'], f1, re, pr, roc, gini))

        score = {'f1': f1,
                 'recall': re,
                 'prec': pr,
                 'roc_auc': roc,
                 'gini': gini}

        return score

    def predict(self, test):
        """
        Predict result

        Parameters
        ----------

        test: pandas DataFrame
            Датасет для предсказания
        """

        result = test.copy()

        for model in self.models:

            result[model['model_name']] = np.zeros(result.shape[0])

            for fold in model['folds']:
                result[model['model_name']] += fold['model'].predict_proba(test)[:, 1]

            result[model['model_name']] = result[model['model_name']] / len(model['folds'])

        # Можно построить ансамбль
        return result

