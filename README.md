# Capstone Project
## Welcome to the repo for my Capstone Project as Galvanize San Francisco.

The project focuses on predicting DotA 2 match outcomes by looking at the first 10 minutes of each match. Here are the goals:

1. Work on a set of 50,000 ranked matches in DotA 2 before the infamous 7.00 patch, and later expand into an even larger dataset offered by yasp.co on Academic Torrents.
2. Data for each game contains massive amounts of information. However, no standalone features is provided. I have to create my feature matrix from scratch by aggregating information over multiple datasets.
3. Start by focusing on the first 10 minutes of each game, find the influence of hero drafting and player information within that timeframe over the outcome.
4. There are many ways I could aggregate player information within the first 10 minutes of the game, I will try multiple feature engineering strategies to find the best set of features.
5. Investigate the interaction between player(hero) roles and match specific information to see if I could improve prediction accuracy.
6. Plot accuracy of my model over varying observed timeframe of a match to see if there are generic deciding moments in matches.

For those of you interested in the dataset I used, here's the link:

https://www.kaggle.com/devinanzelmo/dota-2-matches

## Objective:
One of the most nagging issue in the gaming industry is game balance. Imagine a 40-minute game, whose outcome can be predicted reliably at the 10-minute mark every single time. Would you play this game? My guess is no. Any predictable game is, for the lack of a better word, lame. In gaming terms, such is game is called unbalanced. Game imbalance is almost always caused by some overpowered mechanics that could single-handedly decide the outcome of a game. My goal is to investigate whether DotA 2 is a balanced game.

DotA 2, or Defense of the Ancients 2, is a multiplayer online battle arena(MOBA) game, in which 10 players get split into two teams of 5, called the radiant and the dire, who then try to destroy each other's ancient. It is currently the second most popular MOBA game in the industry, netting over 200 millions per year for its developer--Valve. It should not come as a surprise that DotA 2's balance is of utmost interest to its developer and players.

Since game balance is directly tied to predictability, I am checked the balance of DotA 2 by predicting the outcome after observing only the first 10 minutes of any given match. The higher my prediction accuracy get, the more imbalanced the game would be. This main objective also allowed me to accomplish two side goals simultaneously. First, an accurate prediction model can offer solid strategic advice to players looking to maximize their win rates. The features in my model will help players better study the game mechanics. Second, in-game wagering systems that allow players to bet the outcome of their current match in the first minute can benefit a lot from a good prediction model as well.

## Case Study:
### Preprocessing
Unlike any data I have dealt with before, this dataset was very widely distributed in terms of content. There were 19 tables in total, each of them representing a different concentration of data. Luckily, the data was relatively well-behaved, considering that all the data came directly from the DotA 2 api. All the missing values I had to deal with could be filled in uniformly with 0 while conforming to each feature's definition. There was a slight hiccup, however, with the coding of hero_id. The original hero_names.csv skipped 24 as a hero_id. This problem was also present in players.csv. I fixed them with a correction mapping.

### Methodology
The target variable (outcome of the match) is very balanced. The heavy lifting was in feature engineering. Although the dataset was super rich, it did not offer any useful in-place features that I could use immediately. This motivated me to create every one of my features from scratch using domain knowledge.

#### Feature Engineering
1. Hero Selection: Every player has to select a different hero for each game. DotA 2 heroes do not just look different, they also serve different purposes. It is pretty intuitive that certain hero compositions will help players win a game. There were 112 heroes in total. I doubled the roster to create a list for both the radiant and the dire team, and gave 1 to selected heroes and 0 otherwise.
2. Net Worth at the 10-minute mark: Net worth represents the total amount of gold (in terms of both item assets and gold itself). The higher the net worth, the better items a hero could get. Good items both amplify a hero's power and cover its deficiency. Thus, heroes with higher net worth will be more likely to win the game. I got this feature by referencing the player_time.csv in the dataset.
3. Others: Other features that I engineered include net death counts from team fights before 10 minutes, team composition of hero roles and the interaction between each hero's role and its net worth. Higher team fight death counts implies that a team is more likely to lose. Having a bad composition of roles (too many or few of one role) may put a team at a disadvantage. A non-core hero doing extremely well in terms of net worth may indicate that a team is dominating the game. Sadly, these features did not help much.
