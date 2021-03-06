# Classifying Wine Varietals

![Banner image of Okanagan Valley](images/banner.jpg)

## Motivation and Goals

In recent years, since moving a few blocks from a great wine store, I've become more and more interested in wine. I've always enjoyed trying new things and since I've been going to tastings I've been blown away by just how much different types of wine can vary. That said, tasting notes have always seemed incredibly subjective to me, and I've always wondered how accurate you can really be in predicting a wine in a blind taste test. To that end I set out on this project to see how accurately a machine learning model could predict a varietal by tasting notes alone, with the eventual end goal of using said model for a recommender.

## The Data

The data was found on [kaggle](https://www.kaggle.com/zynicide/wine-reviews) and was originally sourced from [Wine Enthusiast](https://www.winemag.com/) via webscraping. The data is presented in 2 .csv files that on their face seem to contain 280 thousand reviews. However, due to the nature of the scraping algorithm used there are numerous duplicates within and between the two which, after elimination, leave ~170 thousand unique reviews. For the time being most of the features in the data were dropped due to irrelevance to the immediate question.

| country | description | designation | points | price | province | region_1 | region_2 | taster_name |variety | winery |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Italy | Aromas include tropical fruit, broom, brimstone and dried... | Vulkà Bianco | 87 | nan | Sicily & Sardinia | Etna | nan | Kerin O’Keefe | White Blend | Nicosia |
| Portugal | This is ripe and fruity, a wine that is smooth while still... | Avidagos | 87 | 15 | Douro | nan | nan | Roger Voss | Portuguese Red | Quinta dos Avidagos |
| US | Tart and snappy, the flavors of lime flesh and rind... | nan | 87 | 14 | Oregon | Willamette Valley | Willamette Valley | Paul Gregutt | Pinot Gris | Rainstorm |

The features of immediate interest kept were only 'description' and 'variety' containing the review with tasting notes and the variety of wine. All datapoints had a full description but a single datapoint did have a null value for variety and was dropped.

Initial EDA and research focused on the spread of varieties. Exploration of the descriptions will be discussed shortly in the section on featurization. One of the largest concerns in the varieties was the distribution of reviews. 756 varieties are reviewed in the data set but the majority of them have very little representation. Below you can see the top 10 most reviewed varieties.

![Top 10 reviewed wines](images/top_varieties.png)

After some debating I settled on a sub-sample of the top 15 most reviewed wine varieties to move forward with for text featurization and model creation. The choice was based on a few factors but chief among them the top 15 included all varieties with more than 3,000 reviews and accounted for 65% or ~110 thousand datapoints from our original dataset. Furthermore the wines represented in the top 15 walk a line between diversity and similarity, issues addressed within just this sub-sample should hopefully be extendable to further varieties. Future investigation will require more data on these less represented wines. Below are the 15 wines used and their respective review counts.

| Wine | Reviews | Wine | Reviews | Wine | Reviews |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Pinot Noir | 16651 | Chardonnay | 15625 |Cabernet Sauvignon | 13262 |
| Red Blend | 11214 | Bordeaux-style Red Blend | 8997 | Sauvignon Blanc |  6803 |
| Riesling | 6602 | Syrah | 5856 | Merlot | 4757 |
| Rosé | 3916 | Zinfandel | 3804 | Sangiovese | 3634 |  
| Malbec | 3439 | Nebbiolo | 3123 | White Blend | 3110 |

## Featurization

As with most NLP focused projects a large amount of effort has gone into featurizing the text, at this point in the project it's safe to say the majority of my time has gone to it. Immediate concerns I looked at were the relative length and complexity of reviews incase certain wines were discussed more per review than others. Thankfully for my purposes Wine Enthusiast likes to keep their reviews uniformly short and to the point, they all range from 3 to 4 short sentences.

Moving forward I spent a good deal of time dealing with stop words to remove from the reviews. I'm operating under the assumption that the main differentiating features in the reviews should be flavors and scents. To that end standard english stop words were sourced from NLTK. While looking at the reviews though I noticed a problem fairly quickly, many of them included the name of the varietal. To avoid data leakage all varietals present in the full dataset were added to the stop words. Running through with these stopwords I took a look at the most frequent words in order to get a sense of any others that should be eliminated.

![Initial wordcloud](images/first_cloud.png)

Based on the prevelance of such non-descriptive words as 'wine', 'drink', and 'finish' those and similar words were added to our stop words. As a final note on featurization we should address stemming and lemmatization. As things currently stand default procedures are being run by our vectorizer, in this case tf-idf. Once again the distinguishing terms in our reviews are the descriptions of flavors and scents as such the subtleties of different stemming and lemmatization are somewhat lost on these nouns and adjectives. That is not to say futher investigation will not be attempted in coming development.

## ELMo Vectorization

While investigating feturization ELMo embedding was attempted. The pertinent details of the process, which shared stop word selection with tf-idf as discussed above, can be found within in_development. At this time model performance based on ELMo vectors has yet to surpass models based on tf-idf vectors. As such there is not much to discuss at this point but further trials are planned.

## Model Creation

Throughout the process of featurization there was a good deal of iteration. To that end I relied of Naive Bayes for it's relatively quick fitting and testing times. Of the models I chose a Complement model to address the unbalanced nature of my top 15 varieties, while they may be the largest they still vary greatly in size. Out of the box it performed relatively well, in fact in comparative tests it did as well if not better than Random Forests.

Primary efforts thus far have been focused on understanding the challenges any model will face in trying to categorize these varieties. Initial testing for proof of code and concept was done with just the two most reviewed varieties, Pinot Noir and Chardonnay. Results were incredible out the box with ~98% accuracy on both training and test data. However, with addition of further wines this accuracy dropped dramatically. Before delving into the reasons behind this I just want to mention that accuracy is being used to compare models since there is no difference in consequences between false positives and false negatives.

| Number of Wines | Accuracy of Model |
|:---:|:---:|
| 2 | 97% |
| 3 | 88% |
| 5 | 75% |
| 10 | 63% |
| 15 | 58% |

While a dropoff of accuracy when expanding possible categories in classification isn't necessarily something to be alarmed about I was interested to find out why exactly. Pulling up the most heavily weighted words for our 15 wines made the picture pretty clear.

| Variety | Most Relevant Words |
|:---:|:---:|
| Pinot Noir | cherry, cola, raspberry, acidity, tannins, cherries, light, oak, ripe, dry |
| Merlot | cherry, tannins, plum, soft, aromas, oak, dry, berry, blackberry, chocolate |
| Chardonnay | acidity, oak, pear, pineapple, ripe, vanilla, toast, lemon, citrus, peach |
| Sauvignon Blanc | citrus, green, grapefruit, acidity, crisp, lime, aromas, lemon, fresh, clean |


Looking at these the issue seems like it could lie more in the overlap of varieties than in the complication of the model. Since many red wines have similar flavor profiles the descriptions tend to be similar, and the same holds for white wines. To further test this theory I ran a few more model tests pitting similar and dissimilar wines against eachother.

| Wine 1 | Accuracy | Wine 2|
|:---:|:---:|:---:|
| Pinot Noir | 97% | Chardonnay |
| Pinot Noir | 84% | Merlot |
| Sauvignon Blanc | 85% | Chardonnay |

As you can see similar styles consistently perform worse regardless of sample size. While this is far from an earth shattering conclusion I did find this clear division satisfying.

Once these difficulties were diagnosed I began model tuning to try and create the best possible version for use in the recommender. Despite a great deal of experimentation I was actually unable to get better performance than that of a Complement Naive Bayes. Several models had comparable performance but couldn't compete with the speed of a Naive Bayes and as such they were noted but not advanced with. That said experimentation with ensemble methods did result in some improvement from the metrics above. By taking the probability of each class from the Naive Bayes model and passing those into a Random Forest Classifier as features I was able to increase performance by 2%. Nothing drastic but nonetheless useful.

As with any data project further tuning is always possible and I fully intend to explore that. As always in theory a neural network could increase performance and I am interested in seeing the possibilitise there. Also as further vectorization techniques are explored model performance could change dramatically.

## Recommender & Flask Implementation

There isn't too much to say for the recommender, given a string input it vectorizes and runs the result through our ensembled Naive Bayes and Random Forest. The only real change from simply classifying the string as belonging to a single class the probability of all classes are taken and the top 5 are returned as our recommendations. Really the main concern in constructing this recommender was making sure that those top 5 actually made sense to recommend based on the given tasting notes. Thankfully throughout a lot of testing they all seem to branch off of certain notes in the input.

Once I arrived at a solid working recommender I started work on Flask implementation so that it could be used by others, especially people not comfortable working with code or terminal input. Honestly there isn't too much to talk about in terms of this process, Flask is straight forward and HTML templates were simply taken from Bootstrap and modified for my specific needs. Our vectorizer and models were pickled for ease of preservation and use in the app but beyond that the results can be seen here.

## Next Steps 

Some next steps have already been mentioned throughout this README but lets pull them all together and give a clear direction to the project. While we now have featurized data, a solid recommender, and a better understanding of the unique challenges of this dataset there's a lot more I'd like to explore. Of particular interest to me are different vectorization methods and the use of neural networks. Outside of these two main goals there are obviously hundreds of other small tweaks and improvements to made as well as the potential web scraping of further reviews.

## Acknowledgements

I would just like to give thanks to [zackthoutt](https://www.kaggle.com/zynicide) who created the [dataset](https://www.kaggle.com/zynicide/wine-reviews) I used and by extension Wine Enthusiast for the reviews themselves.