import data_cleaning as dc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score

if __name__ == '__main__':
    players_df = pd.read_csv('dota-2-matches/players.csv')
    heros_chart = pd.read_csv('dota-2-matches/hero_names.csv')
    games_df = pd.read_csv('dota-2-matches/match.csv')
    players_time_df = pd.read_csv('dota-2-matches/player_time.csv')

    hero_selection_df = dc.construct_hero_selection_df(players_df, heros_chart)

    # The entire following seciton encompasses testing the feasibility of base models
    # with hero selection by both teams as the input, and match prediction as the output

    X = hero_selection_df.values
    y = games_df.radiant_win.astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    '''
    Logistic Regression
    '''
    print '----------------Logistic Regression----------------'
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    score_lr = cross_val_score(lr, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1).mean()
    test_score_lr = lr.score(X_test, y_test)
    print 'Cross Validation Score for the Logistic Regression Base Model is: {}'.format(score_lr)
    print 'Test Score for the Logistic Regression Base Model is: {}'.format(test_score_lr)
    print '\n'

    '''
    Random Forest Classifier
    '''
    print '--------------Random Forest Classifier--------------'
    rf = RandomForestClassifier(n_estimators=150, criterion='entropy', n_jobs=-1)
    rf.fit(X_train, y_train)
    score_rf = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1).mean()
    test_score_rf = rf.score(X_test, y_test)
    print 'Cross Validation Score for the Random Forest Base Model is: {}'.format(score_rf)
    print 'Test Score for the Random Forest Base Model is: {}'.format(test_score_rf)
    print '\n'

    '''
    K-Nearest Neighbors
    '''
    print '----------------K-Nearest Neighbors----------------'
    knn = KNeighborsClassifier(n_jobs=-1)
    knn.fit(X_train, y_train)
    score_knn = cross_val_score(knn, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1).mean()
    test_score_knn = knn.score(X_test, y_test)
    print 'Cross Validation Score for the KNN Base Model is: {}'.format(socre_knn)
    print 'Test score for the KNN Base Model is: '.format(test_score_knn)

    ###########################################################################################

    '''
    Advanced Models, Some Feature Engineering Requried
    '''
    # Filling all NaNs with 0, all with good reasons
    players_df.fillna(0, inplace=True)

    #-#-#-#-#-#-#-#-#-#
    # First Engineered Feature: Ten Minute Bench Mark

    ten_min_bench_mark = dc.construct_x_seconds_benchmark(players_time_df, x=600)
    first_adv_feature_mat = hero_selection_df.join(ten_min_bench_mark)
    X = first_adv_feature_mat.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, X_test, y_test)

    '''
    Logistic Regression
    '''
    print '----------------Logistic Regression----------------'
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    score_lr = cross_val_score(lr, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1).mean()
    test_score_lr = lr.score(X_test, y_test)
    print 'Cross Validation Score for the Logistic Regression Adv Model 1 is: {}'.format(score_lr)
    print 'Test Score for the Logistic Regression Adv Model 1 is: {}'.format(test_score_lr)
    print '\n'

    '''
    Random Forest Classifier
    '''
    print '--------------Random Forest Classifier--------------'
    rf = RandomForestClassifier(n_estimators=150, criterion='entropy', n_jobs=-1)
    rf.fit(X_train, y_train)
    score_rf = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1).mean()
    test_score_rf = rf.score(X_test, y_test)
    print 'Cross Validation Score for the Random Forest Adv Model 1 is: {}'.format(score_rf)
    print 'Test Score for the Random Forest Adv Model 1 is: {}'.format(test_score_rf)
    print '\n'

    '''
    K-Nearest Neighbors
    '''
    print '----------------K-Nearest Neighbors----------------'
    knn = KNeighborsClassifier(n_jobs=-1)
    knn.fit(X_train, y_train)
    score_knn = cross_val_score(knn, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1).mean()
    test_score_knn = knn.score(X_test, y_test)
    print 'Cross Validation Score for the KNN Adv Model 1 is: {}'.format(socre_knn)
    print 'Test score for the KNN Adv Model 1 is: '.format(test_score_knn)

    #-#-#-#-#-#-#-#-#-#
    # Second Engineered Feature(s): Group mean & variance of gold growth in the first 10 Min

    gold_growth_benchmark = dc.construct_x_seconds_gold_growth_benchmark(x_seconds_df)
    second_adv_feature_mat = first_adv_feature_mat.join(gold_growth_benchmark)
    X = second_adv_feature_mat.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    '''
    Logistic Regression
    '''
    print '----------------Logistic Regression----------------'
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    score_lr = cross_val_score(lr, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1).mean()
    test_score_lr = lr.score(X_test, y_test)
    print 'Cross Validation Score for the Logistic Regression Adv Model 2 is: {}'.format(score_lr)
    print 'Test Score for the Logistic Regression Adv Model 2 is: {}'.format(test_score_lr)
    print '\n'

    '''
    Random Forest Classifier
    '''
    print '--------------Random Forest Classifier--------------'
    rf = RandomForestClassifier(n_estimators=150, criterion='entropy', n_jobs=-1)
    rf.fit(X_train, y_train)
    score_rf = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1).mean()
    test_score_rf = rf.score(X_test, y_test)
    print 'Cross Validation Score for the Random Forest Adv Model 2 is: {}'.format(score_rf)
    print 'Test Score for the Random Forest Adv Model 2 is: {}'.format(test_score_rf)
    print '\n'
