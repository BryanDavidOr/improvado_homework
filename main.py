import requests
import torch
from bs4 import BeautifulSoup
import pandas as pd
import re
from transformers import BertForQuestionAnswering, BertTokenizer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    url = 'https://improvado.io/'
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the page title
    page_title = soup.title.string

    # Extract all the links on the page
    links = []
    for link in soup.find_all('a'):
        links.append(link.get('href'))

    # Extract all the headings on the page
    headings = []
    for heading in soup.find_all(['h1', 'h2', 'h3']):
        headings.append(heading.text.strip())

    # Extract all the paragraphs on the page
    paragraphs = []
    for paragraph in soup.find_all('p'):
        paragraphs.append(paragraph.text.strip())

    data = {'Page Title': [page_title],
            'Links': [', '.join(links)],
            'Headings': [', '.join(headings)],
            'Paragraphs': [', '.join(paragraphs)]}
    #save the information to review with a more specialized tool in csv
    df = pd.DataFrame(data)
    df.to_csv('improvado.csv', index=False)


    #I read the information once I understand the scheme of the information collected and save
    # Load data from CSV file using pandas
    data = pd.read_csv('improvado.csv')
    page_title = data['Page Title'][0]
    links = data['Links'][0]
    products = re.findall(r"/products/(\w+\-?\w+)", links)
    products = list(set(products))
    #Preparation of questions and answers
    question = [
                'What title does Improvado have on your website?',
                'How many products does Improvado have on your website?',
                'What products does Improvado have on its website?',
                'How many links does Improvado have on your website?'
                ]
    answers = [page_title,
               str(len(products)),
               str(products),
               str(len(links.split(',')))]
    data_to_df = {"questions": question, "answers": answers}
    data_df = pd.DataFrame(data_to_df)

    # Load pre-trained tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Iterate through each row of the dataframe
    for index, row in data_df.iterrows():
        # Extract context and question from the current row
        context = row['answers']
        question = row['questions']

        # Encode the question and context using the tokenizer
        encoding = tokenizer.encode_plus(question, context, return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Use the pre-trained model to predict start and end positions for the answer
        start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False)

        # Find the token positions of the answer using the highest score
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1

        # Convert the token positions to actual text and print the result
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))

        print("Question:", question)
        print("Answer:", answer)
        print("-------------")

