"""
使用特征训练模型得到 WSI 分类器 以及对分类器进行 ROC 评估
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/../../'))

FEATURE_START_INDEX = 6

parser = argparse.ArgumentParser(description='Train and test the wsi classification')
parser.add_argument('probs_map_features_train', default=None, metavar='PROBS_MAP_FEATURES_TRAIN', type=str,
                    help='Path to the probs map features training data file.')
parser.add_argument('probs_map_features_test', default=None, metavar='PROBS_MAP_FEATURES_TEST', type=str,
                    help='Path to the probs map features testinging data file.')
parser.add_argument('test_csv_gt', default=None, metavar='TEST_CSV_GT', type=str,
                    help='Path to the lable file of test data.')

def plot_roc(gt_y, prob_predicted_y):
    """
    ROC 评估
    """
    predictions = prob_predicted_y[:, 1]
    fpr, tpr, _ = roc_curve(gt_y, predictions)

    roc_auc = auc(fpr, tpr)

    plt.figure(0).clf()
    # r'$\alpha_i > \beta_i$'
    plt.plot(fpr, tpr, 'b', label=r'$AUC_{Proposed} = %0.4f$' % roc_auc)

    plt.title('ROC curves')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    lg = plt.legend(loc='lower right', borderaxespad=1.)
    lg.get_frame().set_edgecolor('k')
    plt.grid(True, linestyle='-')
    plt.show()


def validate(x, gt_y, clf):
    """
    预测
    """
    predicted_y = clf.predict(x)
    prob_predicted_y = clf.predict_proba(x)
    logging.info('confusion matrix:')
    logging.info(pd.crosstab(gt_y, predicted_y, rownames=['Actual'], colnames=['Predicted']))

    return predicted_y, prob_predicted_y


def train(x, y):
    """
    训练分类器
    """
    # 随机森林
    # clf = RandomForestClassifier(n_estimators=50, n_jobs=2)
    # clf.fit(x, y)
    
    # K近邻
    # clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    # clf.fit(x, y)

    # c-支持向量机
    clf = svm.SVC(kernel='linear', C=1.5, probability=True)
    clf.fit(x, y)

    # 高斯朴素贝叶斯
    # clf = GaussianNB()
    # clf.fit(x, y)

    return clf


def load_train_test_data(f_train, f_test, f_test_gt):
    """
    载入训练数据
    没搞懂为什么需要单独读入测试集的标签
    根据 extract_feature_probsmap.py 特征的最后一行就是标签
    """
    df_train = pd.read_csv(f_train) # 训练数据集
    df_test = pd.read_csv(f_test) # 测试数据集
    df_test_gt = pd.read_csv(f_test_gt, header=None) # test 的标签
    # print(df_test_gt)
    df_test_gt.at[df_test_gt[1] == 'Tumor'] = 1
    df_test_gt.at[df_test_gt[1] == 'Normal'] = 0
    # print(df_test_gt)
    test_gt = df_test_gt.ix[:, 1]

    # 特征的最后一个是标签
    n_columns = len(df_train.columns)
    feature_column_names = df_train.columns[FEATURE_START_INDEX:n_columns - 1]
    label_column_name = df_train.columns[n_columns - 1]

    return df_train[feature_column_names], df_train[label_column_name], df_test[feature_column_names], test_gt


def run(args):
    train_x, train_y, test_x, test_y = load_train_test_data(args.probs_map_features_train,
                                                            args.probs_map_features_test,
                                                            args.test_csv_gt)
    model = train(train_x, train_y)
    predict_y, prob_predict_y = validate(test_x, test_y, model)
    plot_roc(test_y, prob_predict_y)

def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

