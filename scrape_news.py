import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Send a GET request to the website
url = 'https://www.nytimes.com/'
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Find sections on the webpage
sections = soup.find_all('a', class_='css-kz1pwz')  # Assuming sections have this class

# Dictionary to hold section names and their URLs
section_urls = {}

# Extract section names and URLs
for section in sections:
    section_name = section.text.strip()
    section_url = section['href']
    section_urls[section_name] = section_url

# Loop through sections to get article headlines and their respective categories
articles_data = []
for section, section_url in section_urls.items():
    section_response = requests.get(section_url)
    section_soup = BeautifulSoup(section_response.content, 'html.parser')
    
    # Extract article headlines and categories
    articles = section_soup.find_all('h2', class_='css-1d38tss')
    for article in articles:
        article_headline = article.text.strip()
        articles_data.append({'Section': section, 'Headline': article_headline})

# Store the scraped data in a DataFrame
df = pd.DataFrame(articles_data)

# Text classification setup
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Headline'])
y = df['Section']

# Splitting the dataset for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Classification report
report = classification_report(y_test, y_pred)

# Save the classification report to a CSV file
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report.csv', index=True)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
