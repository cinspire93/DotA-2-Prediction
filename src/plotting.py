import pandas as import pd
import numpy as np
import data_cleaning as dc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def random_forest_plotting(nums_trees, X, y):
    accuracy = []
    for num_trees in nums_trees:
        rf = RandomForestClassifier(n_estimators=num_trees, max_features=0.9, criterion='entropy', n_jobs=-1)
        score = cross_val_score(rf, X, y, scoring='accuracy', cv=8, n_jobs=-1).mean()
        accuracy.append(score)
    plt.plot(nums_trees, accuracy)
    plt.savefig('accuracy_vs_numbers_of_trees')
    plt.show()

def plot_accuracy_vs_time(players_time_df, hero_selection_df, y, time_range=600, step=60):
    lr = LogisticRegression(penalty='l1', C=10)
    time_marks = np.linspace(60, time_range, time_range/step, dtype=int)
    scores = []
    for time_mark in time_marks:
        x_seconds_df = dc.construct_x_seconds_df(players_time_df, threshold=time_mark)
        x_seconds_max_wealth = dc.construct_x_seconds_max_wealth(x_seconds_df)
        feature_matrix = hero_selection_df.join(x_seconds_max_wealth)
        X = feature_matrix.values
        scores.append(cross_val_score(lr, X, y, scoring='accuracy', cv=10, n_jobs=-1).mean())
    fig = plt.figure(figsize=(8, 8))
    plt.plot(time_marks, scores, 'r-o', lw=3, label='Accuracy vs. Observed Time')
    plt.title('Accuracy Behavior of Model Over Time', fontsize=25)
    plt.xticks(time_marks, time_marks, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Observed Time Marks', fontsize=20)
    plt.ylabel('Accuracy of Prediction', fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig('accuracy_vs_time')
    plt.show()
