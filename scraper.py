import requests
from bs4 import BeautifulSoup
import spacy


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def fetch_author_info(author_name):
    search_url = f"https://en.wikipedia.org/wiki/{author_name.replace(' ', '_')}"
    response = requests.get(search_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        author_info = ' '.join([para.text for para in paragraphs])
        return author_info
    return None


nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    """
    geo, norp, lang = [], [], []
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            geo.append(ent.text)
        elif ent.label_ == 'NORP':
            norp.append(ent.text)
        elif ent.label_ == 'LANGUAGE':
            lang.append(ent.text)
        return set(geo), set(norp), set(lang)
    """
    return ' '.join(tokens)

"""
geo, norp, lang = preprocess_text(auth)
print('Geo', geo)
print('Norp ', norp)
print('Lang', lang)
"""


def train_model(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])

    model_pipeline.fit(X_train, y_train)
    print(f"Model Accuracy: {model_pipeline.score(X_test, y_test)}")

    return model_pipeline

def predict_nationality(model, author_info):
    processed_text = preprocess_text(author_info)
    nationality = model.predict([processed_text])
    return nationality[0]

def main(authors, data, labels):
    # Load or train the model
    # model = load_model('models/nationality_model.pkl')
    model = train_model(data, labels)  # Assume data and labels are available

    for author in authors:
        author_info = fetch_author_info(author)
        if author_info:
            nationality = predict_nationality(model, author_info)
            print(f"Author: {author}, Nationality: {nationality}")
        else:
            print(f"Author: {author}, Info not found.")


if __name__ == "__main__":

    training_data = []
    data = [preprocess_text(fetch_author_info(i)) for i in training_data]
    labels = []

    authors = ["George Orwell", "Albert Camus", "Jack Kerouac", "Cornelia Funke"]
    main(authors, data, labels)