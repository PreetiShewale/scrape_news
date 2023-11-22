# Web Scraping and Text Classification

This Python script scrapes news headlines from The New York Times website and performs text classification on the article headlines based on their sections.

## Description

The script consists of two main parts:
1. **Web Scraping:** It uses requests and BeautifulSoup to extract news headlines and their respective sections from The New York Times website.
2. **Text Classification:** It employs a Naive Bayes classifier to categorize news headlines into their corresponding sections.

## Getting Started

### Prerequisites
- Python 3.x
- Libraries: requests, BeautifulSoup, pandas, scikit-learn

### Installation
1. Clone this repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

Run the Python script `scrape_and_classify.py`:

```bash
python scrape_and_classify.py
