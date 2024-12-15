# import pandas as pd
# import matplotlib as plt

# # Load the CSV file into a DataFrame
# input_filename = 'train.csv'  # Replace with your input file name
# df = pd.read_csv(input_filename)

# # Calculate the threshold for missing values
# threshold = 0.1 * len(df)  # 10% of the total number of rows

# # Remove columns with more than 10% missing values
# cleaned_df = df.dropna(axis=1, thresh=len(df) - threshold)

# # Save the cleaned DataFrame to a new CSV file
# output_filename = 'cleaned_file.csv'  # Replace with your desired output file name
# cleaned_df.to_csv(output_filename, index=False)

# print(f"Cleaned DataFrame saved to {output_filename}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка CSV файла в DataFrame
input_filename = 'valid.csv'  # Замените на имя вашего входного файла
df = pd.read_csv(input_filename)

# Удаление столбцов с более чем 10% пропущенных значений
threshold = 0.1 * len(df)  # 10% от общего количества строк
cleaned_df = df.dropna(axis=1, thresh=len(df) - threshold)

# Удаление столбцов, в которых заполнено менее 90% значений
min_filled_threshold = 0.9 * len(cleaned_df)  # 90% от общего количества строк
cleaned_df = cleaned_df.dropna(axis=1, thresh=min_filled_threshold)

# Удаление незаполненных строк
cleaned_df = cleaned_df.dropna()  # Удаляем строки, в которых есть хотя бы одно пропущенное значение

# Проверка и удаление дубликатов
cleaned_df = cleaned_df.drop_duplicates()  # Удаляем дубликаты

# Функция для удаления выбросов
def remove_outliers(df):
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Удаление выбросов
cleaned_df = remove_outliers(cleaned_df)

# Визуализация тепловой таблицы пропусков
plt.figure(figsize=(10, 8))
sns.heatmap(cleaned_df.isnull(), cbar=False, cmap='viridis')
plt.title('Тепловая карта пропусков')
plt.show()

# Сохранение очищенного DataFrame в новый CSV файл
output_filename = 'tr_file.csv'  # Замените на желаемое имя выходного файла
cleaned_df.to_csv(output_filename, index=False)

# Вывод информации о количестве оставшихся строк и столбцов
print(f"Количество оставшихся строк: {cleaned_df.shape[0]}")
print(f"Количество оставшихся столбцов: {cleaned_df.shape[1]}")

# Сообщение о сохранении
print(f"Очищенный DataFrame сохранен в {output_filename}")

# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from catboost import CatBoostClassifier, Pool
# from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report
# from scipy import stats

# from tqdm import tqdm, trange

# # Under/Over Sampling
# from imblearn.under_sampling import TomekLinks
# from imblearn.over_sampling import RandomOverSampler, ADASYN, BorderlineSMOTE

# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

# from imblearn.ensemble import lancedBaggingClassifieBar, EasyEnsembleClassifier
# from imblearn.ensemble import RUSBoostClassifier, BalancedRandomForestClassifier



# from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report, precision_score, recall_score, f1_score


# # Настройки
# # Убираем ограничение отображемых колонок
# pd.options.display.max_columns = None
# pd.options.display.max_rows = 100
# # Устанавливаем тему по умолчанию
# sb_dark = sns.dark_palette('skyblue', 8, reverse=True) # teal
# sns.set(palette=sb_dark)


# train_filename = "train.csv"
# test_filename =  "valid.csv"


# df_train = pd.read_csv(train_filename, parse_dates=["report_date"])
# df_train.shape

# # Смотрим сколько признаков имеют NaN для большинства объектов
# # Отдельно по классам и если для обоих классов эти признаки не значимы, тогда удаляем признаки
# threshold_drop = 0.9
# nan_df_cls1 = df_train[df_train["target"] == 1].isna().sum()
# nan_df_cls0 = df_train[df_train["target"] == 0].isna().sum()
# count_cls1 = len(df_train[df_train["target"] == 1] )
# count_cls0 = len(df_train[df_train["target"] == 0] )
# drop_col_cls1 =  nan_df_cls1[nan_df_cls1 > count_cls1*threshold_drop]
# drop_col_cls0 =  nan_df_cls0[nan_df_cls0 > count_cls0*threshold_drop]
# drop_columns = set(drop_col_cls1.index)&set(drop_col_cls0.index)

# # Исключаем признаки с большим кол-ом пропусков
# df_train = df_train.drop(columns=drop_columns, errors='ignore')
# df_train.shape

# # Анализ col1454 - признак схож с client_id
# print(len(df_train), len(df_train['client_id'].unique()), df_train['col1454'].unique().shape)
# df_train[df_train['col1454'] == '01febac0-b083-494e-8589-f98400074b94']

# target_column = 'target'
# report_date_column = 'report_date'
# id_client_columns = ["col1454", "client_id"]
# feature_columns = list(set(df_train.columns) - set([report_date_column]) - set([target_column]) - set(id_client_columns))

# # Поиск константных признаков
# constant_features = []
# # Если везде одно значение  - это константа
# for column in feature_columns:
#     if len(df_train[column].value_counts()) == 1:
#         constant_features.append(column)
# df_train = df_train.drop(columns=constant_features, errors='ignore')
# feature_columns = list(set(feature_columns) - set(constant_features))
# df_train.shape

# # Не числовые признаки
# not_num_feature_columns = df_train.select_dtypes('object').columns
# not_num_feature_columns = list(set(not_num_feature_columns)&set(feature_columns))
# num_feature_columns = list(set(feature_columns) - set(not_num_feature_columns))

# # Проверяем уровень значимости p-value и уровень корреляции с целевой функцией
# # Если p-value высокое, а корреляция с таргетом низкая, тогда признак считается не значимым для предсказания
# high_p_value = []

# for column in num_feature_columns:
#     feat = df_train[column]
#     target = df_train[target_column]
#     res = stats.pearsonr(feat, target)
#     corr, p_value = res
#     if p_value > 0.05 and abs(corr) < 0.002:
#         high_p_value.append(column)
# #         print(f"column: {column}, p_value ({p_value}) > 0.05, corr ({abs(corr)}) < 0.01")
# len(high_p_value)
# high_p_value = []
# print(len(feature_columns))
# feature_columns = list(set(feature_columns) - set(high_p_value))
# num_feature_columns = list(set(num_feature_columns) - set(high_p_value))
# print(len(feature_columns))

# # Удаялем дубликаты строк когда все признаки повторяются и различна только дата репорта
# df_train = df_train.drop_duplicates(subset=list(set(df_train.columns) - set(['report_date', 'target'])))
# df_train.shape

# # Проверяем что не осталось NaN
# nan_df = df_train[feature_columns].isna().sum()
# nan_df[nan_df > 0]

# def split_by_client(df, test_size=0.5):
#     """ Метод разделения трейн теста, таким образом чтобы одинаковые клиенты не попадали в разные наборы,
#     и при этом сохранилась стратификация по данным
#     """
#     clients_target_1 = df[df["target"] == 1]["client_id"].unique()
#     clients_t1_train, clients_t1_test = train_test_split(clients_target_1, test_size=test_size, shuffle=True, random_state=53)

#     clients_target_0 = df[df["target"] == 0]["client_id"].unique()
#     clients_t0_train, clients_t0_test = train_test_split(clients_target_0, test_size=test_size, shuffle=True, random_state=53)

#     clients_t0_train = list(set(clients_t0_train) - set(clients_t1_test))
#     clients_t0_test = list(set(clients_t0_test) - set(clients_t1_train))

#     train = pd.concat([df[(df['client_id'].isin(clients_t0_train))], df[(df['client_id'].isin(clients_t1_train))] ] )
#     test = pd.concat([df[(df['client_id'].isin(clients_t0_test))], df[(df['client_id'].isin(clients_t1_test))]])

#     train = train.drop_duplicates(subset=["report_date", "client_id"])
#     test = test.drop_duplicates(subset=["report_date", "client_id"])

#     return train, test

# train_data, val_data = split_by_client(df_train, test_size=0.2)
# val_data, test_data = split_by_client(val_data, test_size=0.5)

# # Проверяем, что нет лика данных по клиентам между трайн/вал/тест
# assert len(set(train_data["client_id"])&set(val_data["client_id"])) == 0, "Лик train val"
# assert len(set(train_data["client_id"])&set(test_data["client_id"])) == 0, "Лик train test"
# assert len(set(test_data["client_id"])&set(val_data["client_id"])) == 0, "Лик test val"

# train_data.shape, test_data.shape, val_data.shape

# # Проверяем дисбаланс разбиения train/val/test
# train_data["target"].value_counts(), val_data["target"].value_counts(), test_data["target"].value_counts()

# X_train = train_data[feature_columns]
# y_train = train_data[target_column]

# X_val = val_data[feature_columns]
# y_val = val_data[target_column]

# X_test = test_data[feature_columns]
# y_test = test_data[target_column]

# X_train.shape, y_train.shape, X_val.shape, y_val.shape, y_test.shape, X_test.shape

# corr_matrix = df_train[feature_columns + [target_column]].corr()

# plt.figure(figsize=(16, 14))
# heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap='RdBu')
# heatmap.set_title('Матрица корреляции')

# # Вывод графика ROC-AUC
# def plot_roc_auc(y_true, y_pred):
#     fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred)
#     roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)

#     plt.figure(figsize=(10, 3))
#     plt.plot(fpr, tpr, color='darkorange',
#              lw=2, label='ROC curve (area = %0.4f)' % roc_auc, alpha=0.5)

#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)

#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.grid(True)
#     plt.xlabel('False Positive Rate', fontsize=12)
#     plt.ylabel('True Positive Rate', fontsize=12)
#     plt.title('Receiver operating characteristic', fontsize=16)
#     plt.legend(loc="lower right", fontsize=12)
#     plt.show()
#     return roc_auc

# # Вывод графика feature importance
# def plot_feature_importance(importance, names, model_name="", top_n=-1, skip_columns=[]):
#     """Функция вывода feature importance
#         :importance - массив важности фичей, полученный от модели
#         :names - массив названий фичей
#         :model_name - название модели
#         :top_n - кол-во выводимых фичей
#         :skip_columns: какие фичи пропустить, такое может понадобиться чтобы временно убрать
#                         из отображаемых горячие фичи, и изучить менее сильные
#         :return - fi_df - feature importance датафрейм
#     """
#     feature_importance = np.array(importance)
#     feature_names = np.array(names)

#     data={'feature_names':feature_names,'feature_importance':feature_importance}
#     fi_df = pd.DataFrame(data)
#     fi_df = fi_df[~fi_df['feature_names'].isin(skip_columns)]
#     fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

#     plt.figure(figsize=(10,8))
#     sns.barplot(x=fi_df['feature_importance'][:top_n], y=fi_df['feature_names'][:top_n])
#     if top_n != -1:
#         plt.title(f"{model_name} FEATURE IMPORTANCE (Top: {top_n})")
#     else:
#         plt.title(f"{model_name} FEATURE IMPORTANCE")
#     plt.xlabel('FEATURE IMPORTANCE')
#     plt.ylabel('FEATURE NAMES')
#     return fi_df

# def metrics_classifie(y_test, y_pred,model, name='model'):
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     print(f'{name} - Precision: {precision:.2f} | Recall: {recall:.2} | F1-score: {f1:.2} | ROCAUC: {roc_auc_score(y_true=y_test, y_score=model.predict_proba(X_test)[:,1])}')

#     # Расчет дисбалнса классов
# classes = np.unique(y_train)
# weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weights = dict(zip(classes, weights))
# class_weights

# model = CatBoostClassifier(eval_metric = "AUC", early_stopping_rounds=200, class_weights=class_weights, cat_features=cat_columns, random_state=53)
# model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=True, verbose=False)

# # Для рассчета ROC-AUC на baseline моделе используем тестовые данные
# y_pred_proba = model.predict_proba(X_test)[:,1]
# y_pred = model.predict(X_test)
# # Строим график ROC-AUC
# roc_auc = plot_roc_auc(y_true=y_test, y_pred=y_pred_proba)
# print(roc_auc)
# print(classification_report(y_test, y_pred))

# dfi = plot_feature_importance(model.get_feature_importance(), X_test.columns, top_n=30)

# # Разбиваем на фолды
# cv_count = 4
# prc_size_fold = 100/(cv_count+1)

# current_size = 100
# folds_list = []
# # Объединяем трайн и вал. Тест остается тестом
# all_folds = pd.concat([train_data, val_data])
# for i in range(cv_count-1):
#     current_size -= prc_size_fold
#     all_folds, current_fold = split_by_client(all_folds, test_size=prc_size_fold/current_size)
#     folds_list.append(current_fold)
#     # Проверяем, что нет лика данных по клиентам между трайн/вал/тест
#     assert len(set(all_folds["client_id"])&set(current_fold["client_id"])) == 0, "Лик train val"
#     assert len(set(all_folds["client_id"])&set(test_data["client_id"])) == 0, "Лик train test"
#     assert len(set(test_data["client_id"])&set(current_fold["client_id"])) == 0, "Лик test val"

# folds_list.append(all_folds)


# # Обучаем k моделей
# proba_predictions = []
# for i in trange(cv_count):
#     pool_train_data = folds_list.copy()

#     val_data = pool_train_data.pop(i)
#     train_data = pd.concat(pool_train_data)

#     X_train = train_data[feature_columns]
#     y_train = train_data[target_column]

#     X_val = val_data[feature_columns]
#     y_val = val_data[target_column]

#     # Проверяем, что нет лика данных по клиентам между трайн/вал/тест
#     assert len(set(train_data["client_id"])&set(val_data["client_id"])) == 0, "Лик train val"
#     assert len(set(train_data["client_id"])&set(test_data["client_id"])) == 0, "Лик train test"
#     assert len(set(test_data["client_id"])&set(val_data["client_id"])) == 0, "Лик test val"

#     model = CatBoostClassifier(eval_metric = "AUC", early_stopping_rounds=200, class_weights=class_weights, cat_features=cat_columns, random_state=53)
#     model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=True, verbose=False)

#     y_pred = model.predict(X_test)
#     metrics_classifie(y_test, y_pred, model)
#     proba_predictions.append(model.predict_proba(X_test))



#     print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, y_test.shape, X_test.shape)

#     # Усреднение вероятностей
# average_proba_predict = np.mean(np.array(proba_predictions), axis=0)
# print(f'ROCAUC: {roc_auc_score(y_true=y_test, y_score=average_proba_predict[:,1])}')

# # Строим график ROC-AUC
# roc_auc = plot_roc_auc(y_true=y_test, y_pred=average_proba_predict[:,1])
# print(classification_report(y_true=y_test, y_pred=[1 if pred >= 0.5 else 0 for pred in average_proba_predict[:,1]]))
# roc_auc

# scale_pos_weight = len(df_train[df_train['target']==0])/len(df_train[df_train['target']==1])
# prc_weight = min(y_train.value_counts())/max(y_train.value_counts())
# classifiers = [
#                ['BalancedBaggingClassifier :', BalancedBaggingClassifier(n_estimators=200, max_samples=prc_weight, random_state=53)],
#                ['EasyEnsembleClassifier :', EasyEnsembleClassifier(random_state=53)],
#                ['ExtraTreesClassifier :', ExtraTreesClassifier(class_weight=class_weights, random_state=53)],
#                ['RandomForest :', RandomForestClassifier(n_estimators=200, class_weight=class_weights, random_state=53)],
#                ['AdaBoostClassifier :', AdaBoostClassifier(random_state=53)],
#                ['GradientBoostingClassifier: ', GradientBoostingClassifier(random_state=53)],
#                # ['XGB :', XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=53)],
#                ['LGBM :', LGBMClassifier(scale_pos_weight=scale_pos_weight, verbose=-1, random_state=53)],
#                ['CatBoost :', CatBoostClassifier(eval_metric = "AUC", early_stopping_rounds=200, class_weights=class_weights, cat_features=cat_columns, verbose=False, random_state=53)]]

# # Обучаем k-1 модель
# proba_predictions = []
# for i in trange(cv_count):
#     pool_train_data = folds_list.copy()

#     val_data = pool_train_data.pop(i)
#     train_data = pd.concat(pool_train_data)

#     X_train = train_data[feature_columns]
#     y_train = train_data[target_column]

#     X_val = val_data[feature_columns]
#     y_val = val_data[target_column]

#     # Проверяем, что нет лика данных по клиентам между трайн/вал/тест
#     assert len(set(train_data["client_id"])&set(val_data["client_id"])) == 0, "Лик train val"
#     assert len(set(train_data["client_id"])&set(test_data["client_id"])) == 0, "Лик train test"
#     assert len(set(test_data["client_id"])&set(val_data["client_id"])) == 0, "Лик test val"

#     for name, model in tqdm(classifiers):
#         print(name)
#         model.fit(X_train, y_train)  # Предполагается, что X_train и y_train подготовлены
#         y_pred = model.predict(X_test)
#         metrics_classifie(y_test, y_pred, model, name)
#         proba_predictions.append(model.predict_proba(X_test))

#     print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, y_test.shape, X_test.shape)

# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from catboost import CatBoostClassifier, Pool
# from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report
# from scipy import stats

# from tqdm import tqdm, trange

# # Under/Over Sampling
# from imblearn.under_sampling import TomekLinks
# from imblearn.over_sampling import RandomOverSampler, ADASYN, BorderlineSMOTE

# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

# from imblearn.ensemble import lancedBaggingClassifieBar, EasyEnsembleClassifier
# from imblearn.ensemble import RUSBoostClassifier, BalancedRandomForestClassifier



# from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report, precision_score, recall_score, f1_score


# # Настройки
# # Убираем ограничение отображемых колонок
# pd.options.display.max_columns = None
# pd.options.display.max_rows = 100
# # Устанавливаем тему по умолчанию
# sb_dark = sns.dark_palette('skyblue', 8, reverse=True) # teal
# sns.set(palette=sb_dark)


# train_filename = "train.csv"
# test_filename =  "valid.csv"


# df_train = pd.read_csv(train_filename, parse_dates=["report_date"])
# df_train.shape

# # Смотрим сколько признаков имеют NaN для большинства объектов
# # Отдельно по классам и если для обоих классов эти признаки не значимы, тогда удаляем признаки
# threshold_drop = 0.9
# nan_df_cls1 = df_train[df_train["target"] == 1].isna().sum()
# nan_df_cls0 = df_train[df_train["target"] == 0].isna().sum()
# count_cls1 = len(df_train[df_train["target"] == 1] )
# count_cls0 = len(df_train[df_train["target"] == 0] )
# drop_col_cls1 =  nan_df_cls1[nan_df_cls1 > count_cls1*threshold_drop]
# drop_col_cls0 =  nan_df_cls0[nan_df_cls0 > count_cls0*threshold_drop]
# drop_columns = set(drop_col_cls1.index)&set(drop_col_cls0.index)

# # Исключаем признаки с большим кол-ом пропусков
# df_train = df_train.drop(columns=drop_columns, errors='ignore')
# df_train.shape

# # Анализ col1454 - признак схож с client_id
# print(len(df_train), len(df_train['client_id'].unique()), df_train['col1454'].unique().shape)
# df_train[df_train['col1454'] == '01febac0-b083-494e-8589-f98400074b94']

# target_column = 'target'
# report_date_column = 'report_date'
# id_client_columns = ["col1454", "client_id"]
# feature_columns = list(set(df_train.columns) - set([report_date_column]) - set([target_column]) - set(id_client_columns))

# # Поиск константных признаков
# constant_features = []
# # Если везде одно значение  - это константа
# for column in feature_columns:
#     if len(df_train[column].value_counts()) == 1:
#         constant_features.append(column)
# df_train = df_train.drop(columns=constant_features, errors='ignore')
# feature_columns = list(set(feature_columns) - set(constant_features))
# df_train.shape

# # Не числовые признаки
# not_num_feature_columns = df_train.select_dtypes('object').columns
# not_num_feature_columns = list(set(not_num_feature_columns)&set(feature_columns))
# num_feature_columns = list(set(feature_columns) - set(not_num_feature_columns))

# # Проверяем уровень значимости p-value и уровень корреляции с целевой функцией
# # Если p-value высокое, а корреляция с таргетом низкая, тогда признак считается не значимым для предсказания
# high_p_value = []

# for column in num_feature_columns:
#     feat = df_train[column]
#     target = df_train[target_column]
#     res = stats.pearsonr(feat, target)
#     corr, p_value = res
#     if p_value > 0.05 and abs(corr) < 0.002:
#         high_p_value.append(column)
# #         print(f"column: {column}, p_value ({p_value}) > 0.05, corr ({abs(corr)}) < 0.01")
# len(high_p_value)
# high_p_value = []
# print(len(feature_columns))
# feature_columns = list(set(feature_columns) - set(high_p_value))
# num_feature_columns = list(set(num_feature_columns) - set(high_p_value))
# print(len(feature_columns))

# # Удаялем дубликаты строк когда все признаки повторяются и различна только дата репорта
# df_train = df_train.drop_duplicates(subset=list(set(df_train.columns) - set(['report_date', 'target'])))
# df_train.shape

# # Проверяем что не осталось NaN
# nan_df = df_train[feature_columns].isna().sum()
# nan_df[nan_df > 0]

# def split_by_client(df, test_size=0.5):
#     """ Метод разделения трейн теста, таким образом чтобы одинаковые клиенты не попадали в разные наборы,
#     и при этом сохранилась стратификация по данным
#     """
#     clients_target_1 = df[df["target"] == 1]["client_id"].unique()
#     clients_t1_train, clients_t1_test = train_test_split(clients_target_1, test_size=test_size, shuffle=True, random_state=53)

#     clients_target_0 = df[df["target"] == 0]["client_id"].unique()
#     clients_t0_train, clients_t0_test = train_test_split(clients_target_0, test_size=test_size, shuffle=True, random_state=53)

#     clients_t0_train = list(set(clients_t0_train) - set(clients_t1_test))
#     clients_t0_test = list(set(clients_t0_test) - set(clients_t1_train))

#     train = pd.concat([df[(df['client_id'].isin(clients_t0_train))], df[(df['client_id'].isin(clients_t1_train))] ] )
#     test = pd.concat([df[(df['client_id'].isin(clients_t0_test))], df[(df['client_id'].isin(clients_t1_test))]])

#     train = train.drop_duplicates(subset=["report_date", "client_id"])
#     test = test.drop_duplicates(subset=["report_date", "client_id"])

#     return train, test

# train_data, val_data = split_by_client(df_train, test_size=0.2)
# val_data, test_data = split_by_client(val_data, test_size=0.5)

# # Проверяем, что нет лика данных по клиентам между трайн/вал/тест
# assert len(set(train_data["client_id"])&set(val_data["client_id"])) == 0, "Лик train val"
# assert len(set(train_data["client_id"])&set(test_data["client_id"])) == 0, "Лик train test"
# assert len(set(test_data["client_id"])&set(val_data["client_id"])) == 0, "Лик test val"

# train_data.shape, test_data.shape, val_data.shape

# # Проверяем дисбаланс разбиения train/val/test
# train_data["target"].value_counts(), val_data["target"].value_counts(), test_data["target"].value_counts()

# X_train = train_data[feature_columns]
# y_train = train_data[target_column]

# X_val = val_data[feature_columns]
# y_val = val_data[target_column]

# X_test = test_data[feature_columns]
# y_test = test_data[target_column]

# X_train.shape, y_train.shape, X_val.shape, y_val.shape, y_test.shape, X_test.shape

# corr_matrix = df_train[feature_columns + [target_column]].corr()

# plt.figure(figsize=(16, 14))
# heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap='RdBu')
# heatmap.set_title('Матрица корреляции')

# # Вывод графика ROC-AUC
# def plot_roc_auc(y_true, y_pred):
#     fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred)
#     roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)

#     plt.figure(figsize=(10, 3))
#     plt.plot(fpr, tpr, color='darkorange',
#              lw=2, label='ROC curve (area = %0.4f)' % roc_auc, alpha=0.5)

#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)

#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.grid(True)
#     plt.xlabel('False Positive Rate', fontsize=12)
#     plt.ylabel('True Positive Rate', fontsize=12)
#     plt.title('Receiver operating characteristic', fontsize=16)
#     plt.legend(loc="lower right", fontsize=12)
#     plt.show()
#     return roc_auc

# # Вывод графика feature importance
# def plot_feature_importance(importance, names, model_name="", top_n=-1, skip_columns=[]):
#     """Функция вывода feature importance
#         :importance - массив важности фичей, полученный от модели
#         :names - массив названий фичей
#         :model_name - название модели
#         :top_n - кол-во выводимых фичей
#         :skip_columns: какие фичи пропустить, такое может понадобиться чтобы временно убрать
#                         из отображаемых горячие фичи, и изучить менее сильные
#         :return - fi_df - feature importance датафрейм
#     """
#     feature_importance = np.array(importance)
#     feature_names = np.array(names)

#     data={'feature_names':feature_names,'feature_importance':feature_importance}
#     fi_df = pd.DataFrame(data)
#     fi_df = fi_df[~fi_df['feature_names'].isin(skip_columns)]
#     fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

#     plt.figure(figsize=(10,8))
#     sns.barplot(x=fi_df['feature_importance'][:top_n], y=fi_df['feature_names'][:top_n])
#     if top_n != -1:
#         plt.title(f"{model_name} FEATURE IMPORTANCE (Top: {top_n})")
#     else:
#         plt.title(f"{model_name} FEATURE IMPORTANCE")
#     plt.xlabel('FEATURE IMPORTANCE')
#     plt.ylabel('FEATURE NAMES')
#     return fi_df

# def metrics_classifie(y_test, y_pred,model, name='model'):
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     print(f'{name} - Precision: {precision:.2f} | Recall: {recall:.2} | F1-score: {f1:.2} | ROCAUC: {roc_auc_score(y_true=y_test, y_score=model.predict_proba(X_test)[:,1])}')

#     # Расчет дисбалнса классов
# classes = np.unique(y_train)
# weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weights = dict(zip(classes, weights))
# class_weights

# model = CatBoostClassifier(eval_metric = "AUC", early_stopping_rounds=200, class_weights=class_weights, cat_features=cat_columns, random_state=53)
# model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=True, verbose=False)

# # Для рассчета ROC-AUC на baseline моделе используем тестовые данные
# y_pred_proba = model.predict_proba(X_test)[:,1]
# y_pred = model.predict(X_test)
# # Строим график ROC-AUC
# roc_auc = plot_roc_auc(y_true=y_test, y_pred=y_pred_proba)
# print(roc_auc)
# print(classification_report(y_test, y_pred))

# dfi = plot_feature_importance(model.get_feature_importance(), X_test.columns, top_n=30)

# # Разбиваем на фолды
# cv_count = 4
# prc_size_fold = 100/(cv_count+1)

# current_size = 100
# folds_list = []
# # Объединяем трайн и вал. Тест остается тестом
# all_folds = pd.concat([train_data, val_data])
# for i in range(cv_count-1):
#     current_size -= prc_size_fold
#     all_folds, current_fold = split_by_client(all_folds, test_size=prc_size_fold/current_size)
#     folds_list.append(current_fold)
#     # Проверяем, что нет лика данных по клиентам между трайн/вал/тест
#     assert len(set(all_folds["client_id"])&set(current_fold["client_id"])) == 0, "Лик train val"
#     assert len(set(all_folds["client_id"])&set(test_data["client_id"])) == 0, "Лик train test"
#     assert len(set(test_data["client_id"])&set(current_fold["client_id"])) == 0, "Лик test val"

# folds_list.append(all_folds)


# # Обучаем k моделей
# proba_predictions = []
# for i in trange(cv_count):
#     pool_train_data = folds_list.copy()

#     val_data = pool_train_data.pop(i)
#     train_data = pd.concat(pool_train_data)

#     X_train = train_data[feature_columns]
#     y_train = train_data[target_column]

#     X_val = val_data[feature_columns]
#     y_val = val_data[target_column]

#     # Проверяем, что нет лика данных по клиентам между трайн/вал/тест
#     assert len(set(train_data["client_id"])&set(val_data["client_id"])) == 0, "Лик train val"
#     assert len(set(train_data["client_id"])&set(test_data["client_id"])) == 0, "Лик train test"
#     assert len(set(test_data["client_id"])&set(val_data["client_id"])) == 0, "Лик test val"

#     model = CatBoostClassifier(eval_metric = "AUC", early_stopping_rounds=200, class_weights=class_weights, cat_features=cat_columns, random_state=53)
#     model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=True, verbose=False)

#     y_pred = model.predict(X_test)
#     metrics_classifie(y_test, y_pred, model)
#     proba_predictions.append(model.predict_proba(X_test))



#     print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, y_test.shape, X_test.shape)

#     # Усреднение вероятностей
# average_proba_predict = np.mean(np.array(proba_predictions), axis=0)
# print(f'ROCAUC: {roc_auc_score(y_true=y_test, y_score=average_proba_predict[:,1])}')

# # Строим график ROC-AUC
# roc_auc = plot_roc_auc(y_true=y_test, y_pred=average_proba_predict[:,1])
# print(classification_report(y_true=y_test, y_pred=[1 if pred >= 0.5 else 0 for pred in average_proba_predict[:,1]]))
# roc_auc

# scale_pos_weight = len(df_train[df_train['target']==0])/len(df_train[df_train['target']==1])
# prc_weight = min(y_train.value_counts())/max(y_train.value_counts())
# classifiers = [
#                ['BalancedBaggingClassifier :', BalancedBaggingClassifier(n_estimators=200, max_samples=prc_weight, random_state=53)],
#                ['EasyEnsembleClassifier :', EasyEnsembleClassifier(random_state=53)],
#                ['ExtraTreesClassifier :', ExtraTreesClassifier(class_weight=class_weights, random_state=53)],
#                ['RandomForest :', RandomForestClassifier(n_estimators=200, class_weight=class_weights, random_state=53)],
#                ['AdaBoostClassifier :', AdaBoostClassifier(random_state=53)],
#                ['GradientBoostingClassifier: ', GradientBoostingClassifier(random_state=53)],
#                # ['XGB :', XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=53)],
#                ['LGBM :', LGBMClassifier(scale_pos_weight=scale_pos_weight, verbose=-1, random_state=53)],
#                ['CatBoost :', CatBoostClassifier(eval_metric = "AUC", early_stopping_rounds=200, class_weights=class_weights, cat_features=cat_columns, verbose=False, random_state=53)]]

# # Обучаем k-1 модель
# proba_predictions = []
# for i in trange(cv_count):
#     pool_train_data = folds_list.copy()

#     val_data = pool_train_data.pop(i)
#     train_data = pd.concat(pool_train_data)

#     X_train = train_data[feature_columns]
#     y_train = train_data[target_column]

#     X_val = val_data[feature_columns]
#     y_val = val_data[target_column]

#     # Проверяем, что нет лика данных по клиентам между трайн/вал/тест
#     assert len(set(train_data["client_id"])&set(val_data["client_id"])) == 0, "Лик train val"
#     assert len(set(train_data["client_id"])&set(test_data["client_id"])) == 0, "Лик train test"
#     assert len(set(test_data["client_id"])&set(val_data["client_id"])) == 0, "Лик test val"

#     for name, model in tqdm(classifiers):
#         print(name)
#         model.fit(X_train, y_train)  # Предполагается, что X_train и y_train подготовлены
#         y_pred = model.predict(X_test)
#         metrics_classifie(y_test, y_pred, model, name)
#         proba_predictions.append(model.predict_proba(X_test))

#     print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, y_test.shape, X_test.shape) код не работает, исправь ошибки и если нужно удали лишнее