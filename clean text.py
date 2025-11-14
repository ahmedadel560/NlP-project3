import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def remove_noise(text):
    # remove metions
    text = re.sub(r'@\w+', '', text)
    # remove Hahtags
    text = re.sub(r'#\w+', '', text)
    # remove linls
    text = re.sub(r'http\S+|www\S+', '', text)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove path
    text = re.sub(r'[^\x00-\x7F\u0600-\u06FF]', '', text)
    # return text after removing
    return text

def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words and len(t) > 1]

def apply_stemming(tokens):
    return [stemmer.stem(t) for t in tokens]

def apply_lemmatization(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = remove_noise(text)
    text = remove_emojis(text)
    text = to_lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = apply_stemming(tokens)
    tokens = apply_lemmatization(tokens)
    return " ".join(tokens)


# Load Dataset
df = pd.read_csv("dataset.csv")

# Show info
# السطر ده بيعرفك
#   الاتى :
#   1-عدد الصفوف وبيظهر اسماء الاعمدة كلها -وعدد القيم الفارغة فى كل عمود ونوع كل عمود
df.info()

# Check duplicates in the correct column and print number of them
print("\nDuplicated review_text:", df['reviews.text'].duplicated().sum())

# Apply cleaning on the correct column
df['cleaned_review'] = df['reviews.text'].apply(preprocess_text)

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False, encoding='utf-8-sig')
print("\nSaved cleaned dataset as 'cleaned_dataset.csv'\n")

# Print samples
print("\nSample before and after cleaning:\n")
for i in range(10):
    print("Original:", df['reviews.text'][i])
    print("Cleaned :", df['cleaned_review'][i])
    print("-" * 80)
