# Data Analysis and Visualization of Articles

## Overview
This project involves analyzing a dataset of articles to extract insights, visualize key aspects, and perform topic modeling. The analysis includes creating word clouds, performing sentiment analysis, extracting named entities, and modeling topics.

## Libraries Used
- `pandas`: For data manipulation and analysis.
- `plotly.express`: For interactive visualizations.
- `wordcloud`: To generate word clouds.
- `matplotlib`: For plotting figures.
- `textblob`: For sentiment analysis.
- `spacy`: For Named Entity Recognition (NER).
- `collections`: For creating and managing counters and defaultdicts.
- `sklearn`: For text vectorization and topic modeling.

## Requirements
Make sure to install the necessary libraries using:

```bash
pip install pandas plotly wordcloud matplotlib textblob spacy sklearn
```

Also, download the SpaCy language model:

```bash
python -m spacy download en_core_web_sm
```

## Steps

### 1. Importing Libraries
```python
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import spacy
from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
```

### 2. Loading Data
```python
nlp = spacy.load('en_core_web_sm')
data = pd.read_csv("articles.csv", encoding='latin-1')
data.head()
```

We load the dataset `articles.csv` which contains the articles to analyze.

### 3. Visualizing the Word Cloud
```python
# Combine all titles into a single string
titles_text = ' '.join(data['Title'])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(titles_text)

# Plot the Word Cloud
fig = px.imshow(wordcloud, title='Word Cloud of Titles')
fig.update_layout(showlegend=False)
fig.show()
```

This step generates a word cloud from the article titles, visualizing the most common words.

### 4. Analyzing the Distribution of Sentiments
```python
# Sentiment Analysis
data['Sentiment'] = data['Article'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Sentiment Distribution
fig = px.histogram(data, x='Sentiment', title='Sentiment Distribution')
fig.show()
```

Sentiment analysis is performed on the articles, and the distribution of sentiments is visualized using a histogram.

### 5. Performing Named Entity Recognition (NER)
```python
# NER
def extract_named_entities(text):
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return dict(entities)

data['Named_Entities'] = data['Article'].apply(extract_named_entities)

# Visualize NER
entity_counts = Counter(entity for entities in data['Named_Entities'] for entity in entities)
entity_df = pd.DataFrame.from_dict(entity_counts, orient='index').reset_index()
entity_df.columns = ['Entity', 'Count']

fig = px.bar(entity_df.head(10), x='Entity', y='Count', title='Top 10 Named Entities')
fig.show()
```

Named Entity Recognition (NER) is applied to extract entities from the articles. The most frequent named entities are visualized using a bar chart.

### 6. Topic Modeling
```python
# Topic Modeling
vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
tf = vectorizer.fit_transform(data['Article'])
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topic_matrix = lda_model.fit_transform(tf)

# Visualize topics
topic_names = ["Topic " + str(i) for i in range(lda_model.n_components)]
data['Dominant_Topic'] = [topic_names[i] for i in lda_topic_matrix.argmax(axis=1)]
topic_counts = data['Dominant_Topic'].value_counts().reset_index()
topic_counts.columns = ['index', 'count']  # Rename the columns to match what Plotly expects

fig = px.bar(topic_counts, x='index', y='count', title='Topic Distribution')
fig.show()
```

Latent Dirichlet Allocation (LDA) is used for topic modeling. The distribution of topics across articles is visualized using a bar chart.

### Conclusion
This notebook provides a comprehensive analysis of the articles, including text visualization, sentiment analysis, named entity extraction, and topic modeling. The interactive visualizations created using Plotly enhance the understanding of the data and the insights derived from it.

Feel free to explore the code and adapt it to your specific needs or datasets!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

