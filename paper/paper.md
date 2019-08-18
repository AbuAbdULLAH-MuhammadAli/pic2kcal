---
title: "pic2kcal: End-to-End Calorie Estimation From Food Pictures"
author: |
    Robin Ruede, Lukas Frank, Verena Heußer \
    Karlsruhe Institute of Technology
date: 2019-08-02
abstract: |
    Latest approaches to predicting calories of food usually use models that consist of several pipeline steps, such as segmenting the image, estimating the weight and classifying the ingredient. 
     In this article we present a novel end-to-end approach to estimate the kcal directly from a picture. 
      Since there is no large-scale publicly available dataset to train models on this task, we also collected data from recipes, including images, and matched the ingredients of the recipes with ground truth nutritional information of a food database.
figPrefix: [Figure, Figures]
tblPrefix: [Table, Tables]
secPrefix: [Section, Sections]
citekeys:
    chokr: https://dl.acm.org/citation.cfm?id=3297871 # https://aaai.org/ocs/index.php/IAAI/IAAI17/paper/view/14204/13719
    takumi: https://dl.acm.org/citation.cfm?doid=3126686.3126742 # http://img.cs.uec.ac.jp/pub/conf17/171024ege_0.pdf
    salvador: https://arxiv.org/abs/1812.06164
    usda: https://ndb.nal.usda.gov/ndb/
    fasttext: https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00051
    googleuniv: https://ai.google/research/pubs/pub46808/
    miyazaki: https://ieeexplore.ieee.org/abstract/document/6123373 # http://sci-hub.tw/https://ieeexplore.ieee.org/abstract/document/6123373
    survey: https://ieeexplore.ieee.org/abstract/document/8666636 # http://sci-hub.tw/https://ieeexplore.ieee.org/abstract/document/8666636
    myers: https://ieeexplore.ieee.org/document/7410503 # https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44321.pdf
    caloriemama: https://caloriemama.ai/
    DenseNet: https://ieeexplore.ieee.org/document/8099726
    ResNet: https://ieeexplore.ieee.org/document/7780459
    

citation-style: template/ieee.csl
link-citations: true
---

# Introduction

In recent years the awareness of healthier and more balanced diets has risen a lot. Tracking the exact amount and quality of food eaten is important for successfully following a diet, but doing this manually takes a lot of time and effort, leading to non-optimal results.

Currently, calorie tracking is mostly done by looking up specific ingredients and amounts. There are also a number of tools to help track calorie intake, such as the app Im2Calories by Google [@myers] from 2015, or CalorieMama [@caloriemama], with varying automation.

In this work we propose a method to predict the caloric content of any prepared food based only on a single picture of one portion in an end-to-end fashion.

We collected a dataset of recipes, pictures, and nutritional properties. Then we experimented with various features and models to best predict calories and other nutritional information directly from an image. We measure our results objectively and show that multi-task learning improves the performance.

<!-- 
# Related Work

There's some other papers like [@chokr; @takumi; ]. Ours is more end to end and also BETTER

-   [@chokr]:

    -   details von verena, wsl zu detailliert:
    -   supervised, pipeline: predict food-type -> predict size in g -> predict kcal based on size and food-type (nicht end-to-end)
    -   Dataset: Pittsburg fast-food image dataset (61Essens-Katergorien, 101Sub-Kategorien, jew. 3 Instanzen des Gerichts -> Annotiert mit Größe und Art des Gerichts; hier nur Subset verwendet (6 Klassen angerichten, ~1100 Bilder))
    -   Architektur / Ablauf

        -   PCA: auf 23Features runter
        -   Food-type-Classifier: verschiedene Varianten ausprobiert; beste: SVM
        -   Size Predictor: beste: Random forests
        -   Calories Predictor: NN als predictor
            -   input: 23 visual features + food type + food size
            -   Performance: Vergleich nur gegen eine eigene Studie, keine Angabe von Accuracy (Vergleich des pipeline-Ansatzes (visualfeatures+type+size) gegen Modell nur trainiert auf vis. features
        -   Handgemachte Annotationen für die Kcal-Angaben

-   [@myers]:

    -   wsl zu detailliert
    -   Restaurant specific im2calories
    -   Task1: Is food in Image ? -> Food vs. non-food (binäre Klassifikation)
        -   mit googLeNet CNN (pretrained on ImageNet)
    -   Task2: Content Analysis -> Calorie Prediction
        -   Training:
            1. vorhersage der Zutaten über multi-label classifier
            2. Lookup der Zutaten für mapping Zutat -> kcal
            3. dann schätzen der Gesamt-kcal-anzahl (Summe über die Zutaten-kcal)
        -   Test: auf Datensatz ‘MenuMatch’
    -   Mapping Zutat -> Kcal über FNDDS

-   calorie mama: recognizes ingredients and meals from pictures. pretty impressive tbh.
-   [@miyazaki]: "in which they searched the calorie-annotatedfood photo database for the top 5 similar images based onconventional hand-crafted features such as SURF-based BoFand color histograms and estimated food calories by averag-ing the food calories of the top 5 food photos"
-   [@salvador]: recipe generation (ingredients list, instructions, no amounts or kcal)
-   [@takumi]: multi-task VGG: kcal estimation, food categorization, ingredients estimation, cooking instructions. probably closest to ours? ingredients are not predicted individually, but as a single averaged word2vec embedding to make kcal prediction better
-   [@survey]: comparison of many different things: used datasets, segmentation methods, classification approaches, volume estimation methods of 10+ other papers

-->

# Dataset Extraction and Preprocessing

## Collection

We collected a dataset from a popular German recipe website that contains ingredient lists, cooking instructions, and pictures of the resulting meals. The recipes are from many different cuisines and also include things like cakes, cocktails, and others. Most recipes have at least one picture. Smoe of the pictures are uploaded by the original author of the recipe or by third parties. Some of the pictures are of a single plate of food, others are for example of a whole casserole. We do not have any information about whether a picture contains a single portion. Around 10% of recipes contain a user-given value for how many calories per portion the recipe supposedly has.

## Matching / Preprocessing

Since the dataset only has user-given calorie information for a small part of the data and doesn't include any details regarding the macronutrient composition, and since the user given information is often inaccurate (see [@fig:crappy]), we decided to match the list of ingredients against a database of nutritional values to sum up the proportions of macronutrients as well as the total calories.

![A recipe with an obviously incorrect user-given calorie count.](img/crappy-kcal.png){#fig:crappy}

To facilitate this, we collected a secondary dataset from a German website of nutritional values. The website contains values for the amount of fat, protein, and carbohydrates in grams per 100g of product. Additionally, it contains user-sourced food amounts like "1 medium-sized apple = 130g". The data is partially sourced from the USDA Food Composition Database [@usda], and partially crowd sourced from manufacturer-given data of specific products.

Matching the recipe ingredients to the nutritional database has two main problems.

Firstly, the given ingredient name often includes information that is not relevant to the product itself, but rather to its preparation or visual qualities. These additional text snippets are hard to separate from information that is relevant. For example: `3 onions, diced` and `3 onions, in slices` refer to the same product, while `500g pasta, cooked` and `500g pasta, raw` vary significantly in their caloric density. We experimented with three approaches [reformulate] to solve this problem. Firstly, we tried simply matching the ingredient to the nearest ingredient based on character edit distance (Levenshtein distance). This resulted in very bad matchings because of missing handling of synonyms and the above issue. To solve this we tokenize the ingredient name to words, embed each word to a vector with Word2Vec or FastText [@fasttext], and then use average the word vectors to get an ingredient vector. This is the same method as used in the fasttext library for extracting sentence vectors. This still lead to unsatisfactory results, since each word in the ingredient name has the same weight, even though some specify less important details. For example in "red onion" vs "red apple", the word "red" is much less important than "onion" and "apple". We got the best result by using the Google Universal Sentence Encoder [@googleuniv], which creates 512-dimensional embeddings of any amount of text. We find the best matches for an ingredient by comparing the embedding of user-given free text from the recipe to the embeddings for all food items for which we have nutritional data using the cosine distance, and then try to find a conversion for the given amount to a normalized gram or milliliter amount.

The second problem is matching the amounts. For ingredients given in grams this is trivial, but for many items the recipe authors use other units of measure like e.g. can, piece, tablespoon, "some", "2 large X", "salt 'by taste'". Since spices usually have little impact on the nutritional values, we exclude ingredients that are "by taste" and similar. For the other amounts, we match the unit name (like tablespoon or "medium large") exactly and multiply it with the given amount. We also add some special cases like matching "can" to "can (drained weight)" and similar.

The amount matching is applied to all possible ingredient matches that are similar by more than 84% (measured by cosine distance) [why lol] in decending order, or to the single closest ingredient if there is no match more accurate than 84%.

If the amount matching fails, the ingredient is marked as unmatched. If a recipe has at least one unmatched ingredient, it is discarded.

As a final step, we filter out all data points where the summed up calories of the recipe is outside of two standard deviations from the mean repeatedly until it converges. This is necessary because some recipes contain obviously wrong information (for example in a carrot cake recipe the author specified to use a million carrots).

## Dataset Statistics

In total, the recipe website contains 330 thousand recipes. Of these, 210 thousand have at least one picture. Around 20 thousand recipes with pictures have user-given calorie information, though we didn't use these in the end. The recipes contain a total of 374 thousand unique ingredients. This high number is caused by slight differences in spelling or irrelevant details. In total, we collected 900 thousand pictures. On average, each recipe has 3 pictures.

The database of nutritional values contains a total of 390 thousand ingredients. Many of these are incomplete or duplicates, so we filter them by popularity to 123 thousand ingredients.

After matching the ingredients to the recipes, we have 50 to 85 thousand recipes with full nutritional information, depending on whether we aggregate calories per recipe, per portion or per 100g of raw mass (see [@sec:experiments]). We lose 60% of recipes during matching because our matching discards recipes quickly when the ingredients don't fully match. This is so we can ensure we only retain data points that are accurate, and it could be improved with further tweaking. When aggregating per portion, we lose even more data points since we have to exclude all recipes where the user did not supply information about how many portions a recipe consists of.

In total, we have around 179 to 308 thousand data points (because each recipe has multiple images). We split these into train, validation and test set such that multiple pictures of the same recipe are in the same data split.

<!-- todo: convert table to latex probably so it is consistent -->

|                            | per portion | per 100 g | per recipe |
| -------------------------- | ----------- | --------- | ---------- |
| recipes count before       | 211k        | 211k      | 211k       |
| removed no ings match      | 127k        | 127k      | 127k       |
| removed no portions        | 31k         | 0k        | 0k         |
| kcal mean                  | 425 kcal    | 179 kcal  | 1791 kcal  |
| kcal stddev                | 207 kcal    | 73 kcal   | 1007 kcal  |
| kcal outliers              | 11k         | 14k       | 21k        |
| final recipe count         | 42k         | 70k       | 63k        |
| **final data point count** | 179k        | 308k      | 267k       |

The 20 most common ingredients are shown in [@tbl:ings]. Note how common baking ingredients are. This indicates a cake bias, i.e. the dataset may be biased towards sweet meals and desserts.

\begin{table}
\begin{center}
\begin{tabular}{|r|l|}
\hline
Count & Ingredient \\
\hline\hline
119244 & Salz \\
59066 & Zucker \\
58185 & Ei, vom Huhn \\
46069 & Mehl \\
45891 & Butter \\
41206 & Zwiebel, frisch \\
24531 & Milch (3,8 \%) \\
24011 & Vanillezucker \\
23476 & Zucker \\
22822 & Öl \\
22781 & Paprika, orange \\
21348 & Knoblauch \\
20359 & Wasser \\
19935 & Knoblauch, frisch \\
19336 & Pfefferbreze \\
18928 & Olivenöl \\
15966 & Backpulver \\
15039 & Sahne \\
14751 & Zitrone, frisch \\
13077 & Paprikapulver \\
12487 & Gemüsebrühe, pflanzlich \\
12136 & Backpulver \\
11960 & Käse \\
11673 & Kartoffeln \\
10926 & Eigelb, vom Huhn \\
10780 & Butter, Durchschnittswert \\
9591 & Puderzucker \\
9439 & Petersilie, frisch \\
8708 & Zucchini, grün, frisch \\
8293 & Mehl, Weizenmehl Typ 405 \\
\hline
\end{tabular}
\end{center}
\caption{Most common ingredients after matching.\label{tbl:ings}}
\end{table}

# Models

We followed an end-to-end approach to solve the calorie prediction problem of food images. To do so we used pretrained ResNet [@ResNet] and DenseNet [@DenseNet] architectures as base models. We kept the feature extractor layers and replaced the last fully-connected classification layer. We tried solve the problem interpreting it on the one hand as a classification task and on the other hand as a regression problem. Furthermore we introduced additional learning feedback following a multi-task learning approac.

In the following, only the last layer of the neural network is described. 

In the regression case we trained a model to predict the kcal information with one output neuron and additionally to predict protein, fat and carbohydrate information using three additional neurons. The two models were trained using a L1 and smooth L1 loss.

We adapted the base architecuters to the classification problem by quantizing the regression outputs. So we introduced 50 class buckets for each regression output. The models were trained using a cross entropy loss.

The multi-task model is based on the regression model including the nutritional information with additional binary outputs to predict the top n ingredients. The resulting layer has four regression outputs with 50 binary outputs. The used loss combines a smooth L1 loss for the regression outputs and a binary cross entropy loss for the top 50 ingredients. To get the same scaling of the two learning signals we scaled the binary cross entropy loss with a factor of 400. $$ loss = L1 + 400 * BCE $$

As there are no reference papers working with similar approaches or similar data, the results could not be compared to other studies.
Hence, a simple baseline was implemented to get evidence that our models actually learn and that they are better than random guessing. 

The baseline for the kcal prediction basically is the mean of all samples in the train dataset. 
A basleine model would predict during inference only the mean of the already seen kcal values seen. 
The same baseline was used for predicting the nutritional data. 

# Experiments {#sec:experiments}

We divided the generated dataset into train/test/validation (.7/.15/.15) splits. Our training set contains xxx samples with around xxx images for each recipe. The network was trained 40 epochs using a batch size of 50 samples each batch. The samples of the batches were shuffled every epoch and we evaluated the performance of the model every fiftieth batch. We implemented all networks using Pytorch.

To evaluate the performance of the model we trained several networks and run several experiments to evaluated them using firstly the evaluation data set to get quick feedback. To measure the performance of the model we only compared the given kcal information with the prediction of the network.

Firstly we used our raw data set to train the kcal-model. We wanted the network to predict the kcal information of the recipe visualized on the given input image. To perform well in this task the model needs to learn the concept of the recipe size and predict the calories according it. We assumed due to the amount of samples and the capacity of the model the problem is well learnable. Unfortunately the trained regression model performed not well on the task probably because of outlier recipes in our dataset with not valid kcal information provided by the users. Even after outlier removal, prediction of normalized kcal information of portion and trying a classification approach the model was only slightly better than a baseline model.

Second we evaluated if the additional nutritional information supports the networks capability to generalize on the recipe and portion size. Both the classification and regression objectives performed not well with the further information.

Lastly we reformulated the training objective to a slightly easier problem. We trained the network to predict the calory density of the visualized image. Because of the normalization the network only needs to grasp how many calories are in for instance 100g of the meal. This modification led to significant better results.

We could furthermore improve the results of the model using the multi-task approach. The top 50 ingredients of the recipes were injected as further information to support the model predicting the kcal information. We report the results of this model in the result section.

# Results

Our results can be seen in [@tbl:res]. Example outputs can be seen in [@fig:results].

\begin{table}
\begin{center}
\begin{tabular}{|l|c|}
\hline
Method & kcal relative error \\
\hline\hline
baseline & 0.464 \\
ours (kcal only) & 0.361 \\
ours (w/ macros) & 0.352 \\
ours (w/ macros+ings) & 0.328 \\
\hline
\end{tabular}
\end{center}
\caption{Results per 100g. Note that multitask learning improves performance.\label{tbl:res}}
\end{table}

![Some example results, showing predicted calories, fat, protein, carbohydrates and ingredients.](img/results.png){#fig:results}

# Problems/Fails
Following we describe the most time consuming difficutlties we faced while working on the practical course task. 

- **Scraping:** It was straight forward to crawl the recipes internet page because the graceful HTML structure. Whereas it was challenging to extract the nutritional data of the other website. The main problem was to find a proper method to extract the needed information out of the a HTML table which was modeld using several div tags.  

- **Ingredient matching:** The matching of the recipe ingredients and the food database ingredients could not be solved with a simple method like Levenshtein distance. The obvious pairs could be matched but once for instance further information like *peeled potato*  was provided often  the optimal match was not found. 

- **Kcal outlier:** It was not possible to use the raw user given kcal details because some of them were not accurate. The loss of the trained regression models exploded with bad results as consequence. Based on the matching we calculated kcal information and filtered the outlier recipes. 

- **Tensorboard image visualization:** We used TensorBoard to be able to debug the models. Therefore we logged meta data including the images of the recipes. Because of an error related to the image normalization the images got destroyed. 
 


# Future Work

We currently only use a portion of the information in our extracted dataset. There are other interesting attributes such as the type of the meal (cake, side dish), the ingredient amounts, the cooking instructions, the rating, and further properties. These could be used to further improve our calorie prediction model as did adding prediction of the ingredients. The current dataset contains all available photos of each recipe. It may make sense to implement sanity checks to filter images out if they do not match the recipe in a proper way.

The cooking instructions could also be used to improve the ingredient matching by fine-tuning the text embedding model. We only used the pretrained Universal Sentence Encoder which is trained on online sources like Wikipedia, news, and discussion forums. Since these sources are very generic all ingredients are probably very close in the embedding space even though they may be different. The user-given free text formulation of the cooking instructions usually contain the names of the ingredients in text. Fine-tuning the Sentence Encoder with those instructions should help better encode similar ingredients (e.g. you usually add either baking powder or yeast at a specific cooking stage, so they are fairly similar ingredients).

Further problems related to food could also be approached using the dataset. For some people it may be interesting to know if a meal contains a specific ingredient because of allergies, if it is vegan or vegetarian, or if it fits a specific diet. The dataset provides needed information to train a variety of different models to solve problems related to food.

Currently our kcal prediction model is not highly optimized for the task since it is built on top of pretrained models. As we have shown it is beneficial to inject other data into the model therefore it may be interesting to do further investigation on different model architectures. For instance the representation of the top-n ingredient neurons could be changed from a binary value to a regression predicting the amount of the ingredient. It may also make sense to build entire new architectures using kernels with a size which match the requirements of predicting/classifying food images.

# References
