# Classifying Wine Varietals

![Banner image of Okanagan Valley](images/banner.jpg)

## Motivation and Goals

In recent years, since moving a few blocks from a great wine store, I've become more and more interested in wine. I've always enjoyed trying new things and since I've been going to tastings I've been blown away by just how much different types of wine can vary. That said tasting notes have always seemed incredibly subjective to me, and I've always wondered how accurate you can really be in predicting a wine in a blind taste test. To that end I set out on this project to see how accurately a machine learning model could predict a varietal by tasting notes alone.

## The Data

The data was found on [kaggle](https://www.kaggle.com/zynicide/wine-reviews) and was originally sourced from [Wine Enthusiast](https://www.winemag.com/) via webscraping. The data is presented in 2 .csv files that on their face seem to contain 280 thousand reviews. However, due to the nature of the scraping algorithm used there are numerous duplicates within and between the two which, after elimination leave ~170 thousand unique reviews. For the time being most of the features in the data were dropped due to irrelevance to the immediate question.

![Snapshot of Raw Dataframe](images/full_df.png)

The features of immediate interest kept were only 'description' and 'variety' containing the review with tasting notes and the variety of wine. All datapoints had a full description but a single datapoint did have a null value for variety and was dropped.

Initial EDA and research focused on the spread of varieties. Exploration of the descriptions will be discussed shortly in the section on featurization. One of the largest concerns in the varieties was the distribution of reviews. 756 varieties are reviewed in the data set but the majority of them have very little representation. Below you can see the top 10 most reviewed varieties.

![Top 10 reviewed wines](images/top_varieties.png)

After some debating I settled on a sub-sample of the top 15 most reviewed wines to move forward with for text featurization and model creation. The choice was based on a few factors but chief among them the top 15 included all varieties with more than 3,000 reviews and accounted for 65% or ~110 thousand datapoints from our original dataset. Furthermore the wines represented in the top 15 walk a line between diversity and similarity, issues addressed within just this sub-sample should hopefully be extendable to further varieties. Future investigation will require more data on these less represented wines. Below are the 15 wines used and their respective review counts.

|||||||
|:---:|:---:|:---:|:---:|:---:|:---:|
| Pinot Noir | 16651 | Chardonnay | 15625 |Cabernet Sauvignon | 13262 |
| Red Blend | 11214 | Bordeaux-style Red Blend | 8997 | Sauvignon Blanc |  6803 |
| Riesling | 6602 | Syrah | 5856 | Merlot | 4757 |
| Rosé | 3916 | Zinfandel | 3804 | Sangiovese | 3634 |  
| Malbec | 3439 | Nebbiolo | 3123 | White Blend | 3110 |

## Featurization



## Model Creation

## Next Steps