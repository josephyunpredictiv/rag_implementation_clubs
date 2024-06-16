# -*- coding: utf-8 -*-
"""
Original file is located at (not avalibale publicly)
    https://colab.research.google.com/drive/1Lz8glQLlPbYUPOtPgVjx8HFiiFjxOetf
"""

!pip install faiss-cpu openai
import nltk
from nltk.tokenize import sent_tokenize
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import numpy as np
import torch
import faiss
from openai import OpenAI

nltk.download('punkt')

# Set up OpenAI API key
OpenAI.api_key = 'xxxxxx'

# Example data
data = """Pasion Latino	Meets Mon, Tue, Wed Weekly during Lunch	In: G46
Sponsor: Arenberg	tags: Dance, Culture

WJ Ski Club	Meets Fri Biweekly during Lunch	In: 135
Sponsor: Rosenthal	tags: Interest, Competitive, Games/Sports	@wjskiclub

Political Speakers Club	Meets Wed Monthly during Lunch	In: Room 117
Sponsor: Merrill	tags: Interest

Machine Learning and Computer Science Club	Meets Fri Weekly during Lunch	In: 248
Sponsor: Menchukov	tags: Interest, Competitive, Academic, STEM	Instagram: @wjaiclub Discord: https://discord.gg/fePyaYzk3N

Treats that Treat	Meets Wed Monthly during Other	In: G20
Sponsor: Lipsitz	tags: Interest, Charity/Activism	treatsthattreat.org (website)

Paws For a Cause	Meets Thu Biweekly during Lunch	In: 214
Sponsor: Krents	tags: Interest, Charity/Activism	WJPaws4ACause

Ukrainian Club	Meets Mon Weekly during Lunch	In: 113
Sponsor: Matro	tags: Charity/Activism, Culture	wj_ukrainianclub

Japanese Game and Culture Club	Meets Thu Biweekly during Lunch	In: 117
Sponsor: Merrill	tags: Interest, Competitive, Games/Sports, Culture

Government AP Preparation (GAPP) Club	Meets Mon Weekly during Lunch	In: 114
Sponsor: Meier	tags: Academic	@wjgappclub on Instagram

Culinary Club	Meets Tue Biweekly during Lunch	In: G02
Sponsor: Martinez	tags: Interest, Charity/Activism, Culture	wj.culinary

Business and Investment Club	Meets Thu Weekly during Lunch	In: 110
Sponsor: Rodman	tags: Academic	@wjbusiness

WJ Model United Nations	Meets Wed Weekly during Lunch	In: Room 110
Sponsor: Rodman	tags: Competitive, Academic	Instagram: @WJModelUN

Kaffeeklatsch	Meets Wed Biweekly during Lunch	In: 113
Sponsor: Matro	tags: Interest, Culture

Leveling the Playing Field	Meets Tue Biweekly during Lunch	In: 193
Sponsor: Payne	tags: Charity/Activism	Insta: @wjlpfclub

Sources of Strength	Meets Thu Biweekly during Lunch	In: 117
Sponsor: Kennedy	tags: Charity/Activism

So What Else? club	Meets Thu Biweekly during Lunch	In: Room 247
Sponsor: Besch	tags: Charity/Activism

Maison Shalom	Meets Tue Monthly during Lunch	In: 117
Sponsor: Merrill	tags: Charity/Activism	Instagram- maisonshalomrefugeehelp

Wiffle Ball at WJ	Meets Wed Weekly during Lunch	In: meeting room is 196
Sponsor: Krakower	tags: Games/Sports	wiffleball_wj

Sports Debate	Meets Mon Monthly during Lunch	In: Room #105
Sponsor: Butler	tags: Games/Sports

C.A.R.E club	Meets Thu Biweekly during Lunch	In: G 34
Sponsor: Kinani	tags: Charity/Activism	wj C.A.R.E

WJ Turkish Class	Meets Fri Biweekly during Lunch	In: P15
Sponsor: Worden	tags: Culture

Pre-Med Club	Meets Wed Biweekly during Lunch	In: 113
Sponsor: Matro	tags: Interest, Academic, STEM	Instagram @wjpremedclub

AMVG Club	Meets Wed, Fri Weekly during Lunch	In: Portable 09
Sponsor: Green	tags: Interest	https://www.instagram.com/wj_anvg_official/

Car club	Meets Mon Biweekly during Lunch	In: P08
Sponsor: Laukaitis	tags: Interest, Games/Sports	wjcar_connoissuers

Black Student Union	Meets Mon Weekly during Lunch	In: G14
Sponsor: Alvarez-Garcia	tags: Charity/Activism, Culture	Instagram: @wj.bsu

Filipino Club	Meets Wed Biweekly during Lunch	In: Room 229
Sponsor: Fraser	tags: Culture	@wj.filipinoclub

Minority Scholars Program	Meets Thu Weekly during Lunch	In: Student Commons
Sponsor: Ladson, Hoefling	tags: Culture, Academic	@wjhsmsp

Debate Team	Meets Tue Weekly during After School	In: Room 193
Sponsor: Waldman	tags: Competitive, Academic	@wjdebateteam

Nature Enthusiasts Club (NEC)	Meets Wed Weekly during Lunch	In: #249
Sponsor: Erickson	tags: Arts, Charity/Activism	@wjhs.nec on instagram

Young Hearts Club	Meets Tue Biweekly during Lunch	In: #234
Sponsor: Ducklow	tags: Charity/Activism	will email when it opens

Quiz Bowl	Meets Tue, Thu Weekly during Lunch	In: Room 147
Sponsor: Meyer	tags: Competitive, Games/Sports, Academic	wj_quizbowl

Jewish Student Union	Meets Fri Biweekly during Lunch	In: Media Center Conference Room
Sponsor: Ravick and Zussman	tags: Interest, Culture	@walterjohnsonjsu

WJ's American Cancer Society	Meets Mon Biweekly during Lunch	In: room #117
Sponsor: Merrill	tags: Charity/Activism	Insta: @wjamericancancersociety

Formula 1 Club	Meets Thu Biweekly during Lunch	In: P8
Sponsor: Laukaitis	tags: Interest	@wjf1club on Instagram

Robotics	Meets Mon/Wed Weekly during after school	In: 211
Sponsor: Daney, Menchukov	tags: Competitive, Academic

Rocket League Esports Team	Meets Tue, Wed, Thu Weekly during After School	In: Room #108
Sponsor: Scher	tags: Competitive, Games/Sports	@wjesports

Swifite Sensations	Meets Tue, Wed Weekly during Lunch	In: 229
Sponsor: Fraser	tags: Interest, Culture, Music	@wj.swifties (not confirmed yet)

Learn4Lanka	Meets Wed Biweekly during Lunch	In: P10
Sponsor: Fullenkamp	tags: Charity/Activism

Japanese Language and Culture Club	Meets Tue Biweekly during Lunch	In: G34
Sponsor: Kinani	tags: Interest, Charity/Activism, Culture

WJ Wildlife Care Club	Meets Mon Biweekly during Lunch	In: 211
Sponsor: Daney	tags: Interest, Charity/Activism	@wj_wildlifecareclub

A Wider Circle Club	Meets Thu Biweekly during Other	In: room 117
Sponsor: Merrill	tags: Charity/Activism	awidercircle.wj

Equality Through Education (ETE)	Meets Tue Monthly during Lunch	In: 232
Sponsor: Kerr	tags: Charity/Activism, Academic	@wj_ete (instagram)

Tea Club	Meets Mon Weekly during Lunch	In: 190
Sponsor: Helgerman	tags: Interest, Culture

Chess Club	Meets Mon Weekly during Lunch	In: Room 204
Sponsor: Rooney	tags: Interest, Games/Sports	@wjchess"""

# Split data into sentences
def preprocess_data(data):
    entries = data.split('\n\n')
    sentences = []
    for entry in entries:
        sentences.extend(sent_tokenize(entry))
        #parts = entry.split('\t')
        #for part in parts:
        #    sentences.extend(sent_tokenize(part))
    return sentences

all_sentences = preprocess_data(data)

# Load context encoder and tokenizer
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Encode and normalize the sentences
context_embeddings = []
for sentence in all_sentences:
    inputs = context_tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        embedding = context_encoder(**inputs).pooler_output
    normalized_embedding = embedding / torch.norm(embedding, p=2)
    context_embeddings.append(normalized_embedding.numpy())

context_embeddings = np.vstack(context_embeddings)

# Create and populate FAISS index
dimension = context_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(context_embeddings)

# Load question encoder and tokenizer
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# Function to retrieve relevant sentences
def retrieve_sentences(query, top_k=5):
    inputs = question_tokenizer(query, return_tensors='pt')
    with torch.no_grad():
        question_embedding = question_encoder(**inputs).pooler_output
    normalized_question_embedding = question_embedding / torch.norm(question_embedding, p=2)
    distances, indices = index.search(normalized_question_embedding.numpy(), top_k)
    return [all_sentences[i] for i in indices[0]]

# Function to generate response using OpenAI
def generate_response_rag(query):
    retrieved_sentences = retrieve_sentences(query)
    input_text = query + " " + " ".join(retrieved_sentences)

    client = OpenAI(api_key="xxxxxx")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant aiding students and parents who are looking for clubs at Walter Johnson High School."},
            {"role": "user", "content": input_text}
        ]
    )
    return response.choices[0].message.content, retrieved_sentences

def generate_response(query):
    client = OpenAI(api_key="xxxxxxx")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant aiding students and parents who are looking for clubs at Walter Johnson High School."},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

# Example interaction
user_query = "What is a good club for people who enjoy skiing?"
bot_response = generate_response(user_query)
print("bot response without rag")
print(bot_response)
print()
print()

user_query = "What is a good club for people who enjoy skiing?"
bot_response = generate_response_rag(user_query)
print("bot response with rag")
print(bot_response[0])
print()
print("retrived sentences")
print(bot_response[1])

# Example interaction
user_query = "What is a good club for poeple that want to learn how to code?"
bot_response = generate_response(user_query)
print("bot response without rag")
print(bot_response)
print()
print()
bot_response = generate_response_rag(user_query)
print("bot response with rag")
print(bot_response[0])
print()
print("retrived sentences")
print(bot_response[1])

# Example interaction
user_query = "Is there a chess club?"
bot_response = generate_response(user_query)
print("bot response without rag")
print(bot_response)
print()
print()
bot_response = generate_response_rag(user_query)
print("bot response with rag")
print(bot_response[0])
print()
print("retrived sentences")
print(bot_response[1])

# Example interaction
user_query = "Is there a club for people interested in politics?"
bot_response = generate_response(user_query)
print("bot response without rag")
print(bot_response)
print()
print()
bot_response = generate_response_rag(user_query)
print("bot response with rag")
print(bot_response[0])
print()
print("retrived sentences")
print(bot_response[1])

# Example interaction
user_query = "What clubd are good about people interested in true crime?"
bot_response = generate_response(user_query)
print("bot response without rag")
print(bot_response)
print()
print()

print("bot response with rag")
print(bot_response[0])
print()
print("retrived sentences")
print(bot_response[1])

