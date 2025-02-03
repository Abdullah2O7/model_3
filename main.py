import re
import pickle
from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import emoji
import contractions
from pymongo import MongoClient
from config import Config


nltk.data.path.append('./nltk_data')  
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', download_dir='./nltk_data')

try:
    nltk.data.find('tokenizers/punkt_tab/english.pickle')
except LookupError:
    nltk.download('punkt_tab', download_dir='./nltk_data')
    
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt', download_dir='./nltk_data')

try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise FileNotFoundError("model.pkl not found. Please check the file path.")

try:
    with open('tokenizer.pkl', 'rb') as tokenizer_file:
        vectorizer = pickle.load(tokenizer_file)
except FileNotFoundError:
    raise FileNotFoundError("tokenizer.pkl not found. Please check the file path.")

app = Flask(__name__)

app.config.from_object(Config)
client = MongoClient(app.config['MONGO_URI'])

db = client['users']
users_collection = db['users']

journal_db = client['Journal']
journal_collection = journal_db['journal']

def enhanced_preprocess_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = emoji.demojize(text)  # Convert emojis to text
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters

    tokens = word_tokenize(text)
    
    spell = SpellChecker()
    misspelled = spell.unknown(tokens)
    tokens = [spell.correction(word) if word in misspelled else word for word in tokens]
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)


def predict_journal(journal_entry):
    label_mapping = {0: 'Depressed', 1: 'Normal'}

    preprocessed_text = enhanced_preprocess_text(journal_entry)

    transformed_text = vectorizer.transform([preprocessed_text])

    prediction = model.predict(transformed_text)[0]

    return label_mapping.get(prediction, 'Unknown')


# journal = "I feel so tired and sad all the time. Nothing excites me anymore."
# predict_journal(journal)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_id = data['user_id']
        journal_id = data['journal_id']
        journal_entry = data['journal']

        if not journal_entry:
            return jsonify({'error': 'No journal entry provided'}), 400

        result = predict_journal(journal_entry)

        # Find the journal entry by user ID and journal ID
        update_result = journal_collection.update_one(
            {
                '_id': user_id,
                'journal.entries._id': journal_id
            },
            {
                '$set': {
                    'journal.$[].entries.$[entry].prediction': result
                }
            },
            array_filters=[{'entry._id': journal_id}]
        )

        if update_result.matched_count == 0:
            return jsonify({'error': 'Journal entry not found'}), 404

        return jsonify({
            'message': 'Prediction saved successfully!',
            'prediction': result
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
# def predict():
#     try:
#         data = request.get_json()
#         user_id = data['user_id']
#         journal_id = data['journal_id']
#         journal_entry = data['journal']

#         if not journal_entry:
#             return jsonify({'error': 'No journal entry provided'}), 400

#         result = predict_journal(journal_entry)

#         print(result)

#         return jsonify({'prediction': result})
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500






# import re
# import pickle
# from flask import Flask, request, jsonify
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from spellchecker import SpellChecker
# import emoji
# from pymongo import MongoClient
# from config import Config
# from catboost import CatBoostClassifier
# from sklearn.decomposition import TruncatedSVD
# import numpy as np
# import string


# # Ensure NLTK data is downloaded
# nltk.data.path.append('./nltk_data')
# try:
#     stopwords.words('english')
# except LookupError:
#     nltk.download('stopwords', download_dir='./nltk_data')

# try:
#     word_tokenize("test")
# except LookupError:
#     nltk.download('punkt', download_dir='./nltk_data')

# # Load models and preprocessors
# try:
#     model = CatBoostClassifier()
#     model.load_model('catboost_model.cbm')
# except FileNotFoundError:
#     raise FileNotFoundError("catboost_model.cbm not found. Please check the file path.")

# try:
#     with open('vectorizer.pkl', 'rb') as vectorizer_file:
#         vectorizer = pickle.load(vectorizer_file)
# except FileNotFoundError:
#     raise FileNotFoundError("vectorizer.pkl not found. Please check the file path.")

# try:
#     with open('svd.pkl', 'rb') as svd_file:
#         svd = pickle.load(svd_file)
# except FileNotFoundError:
#     raise FileNotFoundError("svd.pkl not found. Please check the file path.")

# # Flask App Configuration
# app = Flask(__name__)
# app.config.from_object(Config)

# # MongoDB Configuration
# client = MongoClient(app.config['MONGO_URI'])
# db = client['users']
# users_collection = db['users']

# journal_db = client['Journal']
# journal_collection = journal_db['journal']

# def enhanced_preprocess_text(text):
#     text = text.lower()
#     # text = np.fix(text)
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text)
#     text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'@\w+|#\w+', '', text)
#     text = emoji.demojize(text)
#     text = re.sub(r'[^a-z\s]', '', text)
#     tokens = word_tokenize(text)
    
#     spell = SpellChecker()
#     tokens = [spell.correction(word) for word in tokens]
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
    
#     tokens = [word for word in tokens if len(word) > 2]
#     cleaned_text = ' '.join(tokens)
    
#     return cleaned_text

# def Journal_prediction(new_sentences, vectorizer, model):
    
    

#     preprocessed_sentences = [enhanced_preprocess_text(sentence) for sentence in new_sentences]

#     transformed_sentences = vectorizer.transform(preprocessed_sentences)
#     reduced_sentences = svd.transform(transformed_sentences)
    
#     new_predictions = model.predict(reduced_sentences)
    
#     if new_predictions[0] == 0:
#         return "Anexity"
#     elif new_predictions[0] == 1:
#         return "Depression"
#     elif new_predictions[0] == 2:
#         return "Normal"
#     elif new_predictions[0] == 3:
#         return "Sucidel"


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         user_id = data['user_id']
#         journal_id = data['journal_id']
#         journal_entry = data['journal']

#         if not journal_entry:
#             return jsonify({'error': 'No journal entry provided'}), 400

#         result = Journal_prediction([journal_entry], vectorizer, model)

#         # Find the journal entry by user ID and journal ID
#         update_result = journal_collection.update_one(
#             {
#                 '_id': user_id,
#                 'journal.entries._id': journal_id
#             },
#             {
#                 '$set': {
#                     'journal.$[].entries.$[entry].prediction': result
#                 }
#             },
#             array_filters=[{'entry._id': journal_id}]
#         )

#         if update_result.matched_count == 0:
#             return jsonify({'error': 'Journal entry not found'}), 404

#         return jsonify({
#             'message': 'Prediction saved successfully!',
#             'prediction': result
#         }), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
