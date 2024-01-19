#!/usr/bin/env python
# coding: utf-8
import Levenshtein
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
from collections import Counter
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import re
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os,sys
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import argparse


class BioNEN:
    def __init__(self, model_name, dictionary, dfs_data, epsilon, function_name):
        self.model_name = model_name
        self.dictionary = dictionary
        self.dfs_data = dfs_data
        self.epsilon = epsilon
        self.stemmer = SnowballStemmer("english")
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = stopwords.words('english')
        self.function_name = function_name

    def pubtator_to_dict(self, path):
        dataframes_with_ta = {}
        df = pd.DataFrame(columns=['mentions','id'])
        dataframes = {}

        with open(path) as f:
            lines = f.readlines()   
            counter = 0
            cntr2 = 0
            for line in lines:
                line = line.strip()
                parts = line.split('|')

                if (len(parts) >= 3):
                    if (parts[1] == 't'):
                        dataframes_with_ta['title' + str(counter)] = parts[2]
                    if (parts[1] == 'a'):
                        dataframes_with_ta['abstract' + str(counter)] = parts[2]

                if (len(parts) == 1):                
                    x = parts[0].split('\t')
                    if parts[0] == '':
                        dataframes_with_ta['df'+str(counter)] = df
                        dataframes['df'+str(counter)] = df
                        counter += 1
                        df = pd.DataFrame(columns=['mentions','id'])
                    else:
                        if parts[0].split('\t')[1] == 'CID':
                            continue
                        if x[-1] == '-1':
                            continue
                        df.loc[cntr2, 'mentions'] = x[3]
                        df.loc[cntr2, 'id'] = x[-1].replace('MESH:','')
                        cntr2 += 1
        return dataframes
    
    
    def remove_stopwords(self, text, language='english'):
        words = nltk.word_tokenize(text)
        stopwords_set = set(nltk.corpus.stopwords.words(language))
        filtered_words = [word for word in words if word.lower() not in stopwords_set]
        clean_text = ' '.join(filtered_words)
        return clean_text


    def stem_text(self, text):
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return " ".join(stemmed_words)


    def lemmatizer(self, text):  
        doc = self.nlp(text)
        return ' '.join([word.lemma_ for word in doc])


    def get_bert_embeddings(self, model_name, texts):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = BertModel.from_pretrained(model_name)
        encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_outputs = bert_model(**encoded_inputs)
            embeddings = model_outputs.last_hidden_state[:, 0, :]
        return embeddings


    def get_clusters(self, df, ep, col):
        dbscan = DBSCAN(eps=ep, min_samples=1, metric='cosine')
        mention_embeddings = self.get_bert_embeddings(args.model_name, df['mentions'].to_list())
        df['cluster'] = dbscan.fit_predict(mention_embeddings)
        cluster_mapping = df.groupby('cluster')[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
        df = df.merge(cluster_mapping, on='cluster', suffixes=('', '_common'))
        df['dbscan_id'] = df.apply(lambda row: row[str(col) + '_common'] if pd.isna(row[col]) else row[col], axis=1)
        df.drop(columns=str(col) + '_common', inplace=True)
        df['mentions'] = df['mentions'].apply(lambda x: re.sub('[^0-9a-zA-Z]+', ' ', x).lower())
        # If the abbreviation is found, if the clusters are different assign them into same clusters
        abbreviations = {}
        for index , mention in enumerate(df['mentions']):
            words = mention.split()
            if (len(words) > 1) & (len(mention)>3):
                for word in words:
                    word = word.lower()
                    abbreviation = "".join(word[0] for word in words).lower()
                    abbreviations[mention] = abbreviation
        for idx, mention in enumerate(df['mentions']):
            for key, val in abbreviations.items():
                if (mention == val) | (mention.replace(' ','') == val):
                    abbreviation_cluster = df[df['mentions'] == key]['cluster'].values[0]
                    df.loc[idx,'cluster'] = abbreviation_cluster
                    abbreviation_id = df[df['mentions'] == key]['dbscan_id'].values[0]
                    df.loc[idx, 'dbscan_id'] = abbreviation_id    
        return df


    def get_taxonomy_id(self, scientific_name, relaxed, dct):
        scientific_name = scientific_name.lower()
        punctuation_pattern = r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
        scientific_name = re.sub(punctuation_pattern, '', scientific_name)


        max_score_value = None
        max_score = 0

        # Try searching the dictionary first
        if scientific_name in dct:
            return dct[scientific_name]

        words = scientific_name.split()
        n = len(words)

        # If not found, proceed with searching len-1, len-2, ..., 1 words in the dictionary
        for i in range(n - 1, 0, -1):
            partial_name = ' '.join(words[:i])
            if len(partial_name.split()) > 1:
                if partial_name in dct:
                    return dct[partial_name]

        # If still not found, check 'es' and 's' from the end of the last word and check again
        if len(words[-1]) > 1:
            if words[-1][-2:] == 'es':
                words[-1] = words[-1][:-2]
                partial_name = " ".join(words)
                if partial_name in dct:
                    return dct[partial_name]
            if words[-1][-1] == 's':
                words[-1] = words[-1][:-1]
                partial_name = " ".join(words)
                if partial_name in dct:
                    return dct[partial_name]

        # If relaxed, retrieve similarities
        if relaxed:
            # Use the selected function or default to jaro_similarity
            similarity_function = args.function_name or 'Jaro'
            for key, value in dct.items():
                similarity_score = self.run_selected_function(similarity_function, self.stem_text(self.remove_stopwords(scientific_name)), self.stem_text(self.remove_stopwords(key)))
                if similarity_score > max_score:
                    max_score_value = value
                    max_score = similarity_score

            if max_score > 0.7:
                return max_score_value
            else:
                return None
        return None


    def jaccard_similarity(self, str1, str2):
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        intersection = len(set1.intersection(set2))
        union = len(set1) + len(set2) - intersection
        return intersection / union if union != 0 else 0


    def apply_similarity(self, df):
        df['jaccard_id'] = df['dbscan_id']

        for index, row in df.iterrows():
            if pd.isna(row['dbscan_id']):
                cluster_id = row['cluster']
                cluster_mentions = df[df['cluster'] == cluster_id]['mentions'].values.tolist()

                most_similar_id = None
                max_similarity = 0

                # Use the selected function or default to jaro_similarity
                similarity_function = args.function_name or 'Jaro'
                for other_index, other_row in df.iterrows():
                    if not pd.isna(other_row['dbscan_id']) and index != other_index and other_row['cluster'] != cluster_id:
                        similarity_score = self.run_selected_function(similarity_function, row['mentions'], other_row['mentions'])
                        if similarity_score > max_similarity:
                            max_similarity = similarity_score
                            most_similar_id = other_row['dbscan_id']

                if most_similar_id is not None:
                    if max_similarity > 0.7:
                        df.loc[df['cluster'] == cluster_id, 'jaccard_id'] = most_similar_id

        return df


    def calculate_purity(df, ep, mention_embeddings):
        dbscan = DBSCAN(eps=ep, min_samples=1, metric='cosine')
        cluster_labels = dbscan.fit_predict(mention_embeddings)
        df['cluster'] = cluster_labels
        cluster_majority = df.groupby('cluster')['mentions'].apply(lambda x: Counter(x).most_common(1)[0][0])
        total_correct = sum(df[df['cluster'].isin(cluster_majority.index)]['mentions'] == df['cluster'].map(cluster_majority))
        return total_correct / len(df)


    def calculatePurity(df):
        cluster_majority = df.groupby('cluster')['mentions'].apply(lambda x: Counter(x).most_common(1)[0][0])
        total_correct = sum(df[df['cluster'].isin(cluster_majority.index)]['mentions'] == df['cluster'].map(cluster_majority))
        return total_correct / len(df)


    def calculate_silhouette(df, ep, mention_embeddings):
        dbscan = DBSCAN(eps=ep, min_samples=1, metric='cosine')
        cluster_labels = dbscan.fit_predict(mention_embeddings)
        if len(set(cluster_labels)) == 1:
            return -1.0  
        score = silhouette_score(mention_embeddings, cluster_labels, metric='cosine')
        return score


    def calculate_accuracy(self, dfs, col):
        counter = 0
        avg = 0
        avg_list = []
        for key, df in dfs.items():
            if (df.empty == False):
                avg = df[df['id'] == df[col]].shape[0] / df.shape[0]
                avg_list.append(avg)

        return sum(avg_list) / len(avg_list)


    def prepare_lin(path):
        df = pd.read_csv(path, sep='\t')
        df = df[['span','code']].rename(columns = {'span':'mentions','code':'id'})
        df = df.drop_duplicates(subset='mentions').reset_index(drop=True)
        return df


    def dictionary_results(self, df_dictionary, dct):
        df_dict = df_dictionary.copy()
        for key, df in tqdm(df_dict.items()):
            df_copy = df.copy()
            df_copy = df_copy.reset_index(drop=True)
            df_copy = df_copy.fillna(pd.NA)
            for i in range(len(df_copy['mentions'])):
                df_copy.loc[i, 'dict_id'] = self.get_taxonomy_id(df_copy.loc[i, 'mentions'].lower().strip(), False, dct)
                # print(df_copy.loc[i,'id'], df_copy.loc[i, 'dict_id'], df_copy.loc[i, 'mentions'])
                # df_copy.loc[i, 'relaxed_dict_id'] = self.get_taxonomy_id(df_copy.loc[i, 'mentions'].lower().strip(), True, dct)
            df_dict[key] = df_copy
        return df_dict


    def cluster_results(self, df_dictionary, ep, col):
        df_dict = df_dictionary.copy()
        for key, df in tqdm(df_dict.items()):
            if not df.empty:
                df_copy = df.copy()
                df_copy = df_copy.reset_index(drop=True)
                df_copy = df_copy.fillna(pd.NA)
                df_copy = self.get_clusters(df_copy, ep, col)
                df_dict[key] = df_copy
        return df_dict

    def similarity_results(self, df_dictionary):
        df_dict = df_dictionary.copy()
        for key, df in df_dict.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy = self.apply_similarity(df_copy)
                df_dict[key] = df_copy
        return df_dict


    def search_dict(self, dictionary, string):
        return [str(key) + ' _ ' + str(value) for key, value in dictionary.items() if string in key.lower()]


    def prepare_ncbi(self, path):
        dataframes_with_ta = {}
        df = pd.DataFrame(columns=['mentions','id'])
        dataframes = {}

        with open(path) as f:
            lines = f.readlines()   
            counter = 0
            cntr2 = 0
            for line in lines:
                line = line.strip()
                parts = line.split('|')

                if (len(parts) >= 3):
                    if (parts[1] == 't'):
                        dataframes_with_ta['title' + str(counter)] = parts[2]
                    if (parts[1] == 'a'):
                        dataframes_with_ta['abstract' + str(counter)] = parts[2]

                if (len(parts) == 1):
                    x = parts[0].split('\t')
                    if parts[0] == '':
                        dataframes_with_ta['df'+str(counter)] = df
                        dataframes['df'+str(counter)] = df
                        counter += 1
                        df = pd.DataFrame(columns=['mentions','id'])
                    else:
                        df.loc[cntr2, 'mentions'] = x[3]
                        df.loc[cntr2, 'id'] = x[-1]
                        cntr2 += 1
        return dataframes


    def plot_graph(x_values, y_values, title):
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='o', linestyle='-')
        plt.xlabel('Epsilon Values')
        plt.ylabel('Score')
        plt.title(title)
        plt.grid(True)
        return plt.show()


    def levenshtein_similarity(self, str1, str2):
        distance = Levenshtein.distance(str1, str2)
        max_distance = max(len(str1), len(str2))
        similarity = 1 - (distance / max_distance)
        return similarity


    def jaro_similarity(self, str1, str2):
        # If either string is empty, the similarity is 0
        if not str1 or not str2:
            return 0.0

        # Define the matching distance threshold
        match_distance = max(len(str1), len(str2)) // 2 - 1

        # Initialize variables for matches and transpositions
        matches = 0
        transpositions = 0

        # Lists to store whether a character has been matched in each string
        str1_matches = [False] * len(str1)
        str2_matches = [False] * len(str2)

        # Count matches and transpositions
        for i in range(len(str1)):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len(str2))
            for j in range(start, end):
                if not str2_matches[j] and str1[i] == str2[j]:
                    str1_matches[i] = True
                    str2_matches[j] = True
                    matches += 1
                    break

        if matches == 0:
            return 0.0

        # Count transpositions
        k = 0
        for i in range(len(str1)):
            if str1_matches[i]:
                while not str2_matches[k]:
                    k += 1
                if str1[i] != str2[k]:
                    transpositions += 1
                k += 1

        transpositions //= 2

        # Calculate Jaro Similarity
        jaro_similarity = (
            (matches / len(str1)) +
            (matches / len(str2)) +
            ((matches - transpositions) / matches)
        ) / 3.0
        # return jaro_similarity
        # Calculate Jaro-Winkler Similarity
        prefix_length = 0
        for i in range(min(4, min(len(str1), len(str2)))):
            if str1[i] == str2[i]:
                prefix_length += 1
            else:
                break

        jaro_winkler_similarity = jaro_similarity + (prefix_length * 0.1 * (1 - jaro_similarity))
        return jaro_winkler_similarity  

    def dictionary_similarity(self, df_dictionary, dct):
        df_dict = df_dictionary.copy()
        for key, df in tqdm(df_dict.items()):
            if not df.empty:
                df_copy = df.copy()
                df_copy = df_copy.reset_index(drop=True)
                df_copy = df_copy.fillna(pd.NA)
                df_copy['dict_id_2'] = df_copy['jaccard_id']
                df_copy['relaxed_dict_id_2'] = df_copy['jaccard_id']

                preprocessed_mentions = [(i, self.remove_stopwords(mention.lower().strip()), self.stem_text(self.remove_stopwords(mention.lower().strip()))) for i, mention in enumerate(df_copy['mentions']) if pd.isnull(df_copy.loc[i, 'jaccard_id'])]

                for i, mention, preprocessed_mention in preprocessed_mentions:
                    dict_id = dct.get(preprocessed_mention, None)
                    if dict_id:
                        df_copy.loc[i, 'dict_id_2'] = dict_id
                        df_copy.loc[i, 'relaxed_dict_id_2'] = dict_id
                    else:
                        # Only call get_taxonomy_id if the preprocessed mention is not in the dictionary
                        df_copy.loc[i, 'dict_id_2'] = self.get_taxonomy_id(mention, False, dct)
                        df_copy.loc[i, 'relaxed_dict_id_2'] = self.get_taxonomy_id(mention, True, dct)

                df_dict[key] = df_copy
        return df_dict


    def run_selected_function(self, function_name, str1, str2):
        # Map function names to their implementations
        function_mapping = {
            'Jaro Winkler': self.jaro_similarity,
            'Jaro': self.jaro_similarity,
            'Levenshtein': self.levenshtein_similarity,
            'Jaccard': self.jaccard_similarity,
        }

        # Get the selected function, or use the default function (jaro_similarity)
        selected_function = function_mapping.get(function_name, self.jaro_similarity)
        # print(selected_function)
        # Call the selected function
        result = selected_function(str1, str2)

        return result
    
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Biomedical Text Named Entity Normalization Pipeline')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for BERT embeddings')
    parser.add_argument('--dict_file', type=str, required=True, help='Please specify the dictionary path to the pickle file that you want to use')
    parser.add_argument('--dfs_data',type=str, required=True, help='Please specify the path to the dataframe dictionary')
    parser.add_argument('--epsilon', type=float, required=True, help='Please specify the epsilon value for DBSCAN')
    parser.add_argument('--function_name', choices=['Jaro Winkler', 'Jaro', 'Levenshtein', 'Jaccard'], help='Choose a function')
    args = parser.parse_args()
    
    experiment_pipeline = BioNEN(args.model_name, args.dict_file, args.dfs_data, args.epsilon, args.function_name)

    with open(args.dict_file, 'rb') as file:
        dictionary = pickle.load(file)
    
    dfs_test = experiment_pipeline.pubtator_to_dict(args.dfs_data)
    
    for key, value in dfs_test.items():
        dfs_test[key] = value.reset_index(drop=True)

    dfs_test2 = experiment_pipeline.dictionary_results(dfs_test, dictionary)
    print('Dictionary Acc:', experiment_pipeline.calculate_accuracy(dfs_test2, 'dict_id'))
    dfs_test3 = experiment_pipeline.cluster_results(dfs_test2, args.epsilon, 'dict_id')
    dfs_test4 = experiment_pipeline.similarity_results(dfs_test3)
    dfs_test5 = experiment_pipeline.dictionary_similarity(dfs_test4, dictionary)
    
    accuracy1 = experiment_pipeline.calculate_accuracy(dfs_test5, 'dict_id')
    accuracy2 = experiment_pipeline.calculate_accuracy(dfs_test5, 'dbscan_id')
    accuracy3 = experiment_pipeline.calculate_accuracy(dfs_test5, 'jaccard_id')
    accuracy4 = experiment_pipeline.calculate_accuracy(dfs_test5, 'relaxed_dict_id_2')
    print(f"Accuracy of dictionary: {accuracy1}, Accuracy of dbscan: {accuracy2}, Accuracy of file based similarity: {accuracy3}, Accuracy of relaxed dictionary similarity: {accuracy4}")
