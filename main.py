import os
import json
import pandas as pd
import re
import scispacy
import spacy
import sys

from openai import OpenAI
client = OpenAI(api_key='',)

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
#nltk.download('punkt')

# Function to load and process JSON data
def process_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    nested_data = data['PubTator3'][0]
    abstract_text = ""
    annotations_list = []
    for passage in nested_data['passages']:
        if passage['infons']['type'] == 'abstract':
            abstract_text = passage['text']
            abstract_offset = passage['offset']
            for annotation in passage['annotations']:
                try:
                    name = annotation['infons']['name']
                    text = annotation['text']
                    identifier = annotation['infons']['identifier']
                except KeyError:
                    continue
                annotations_list.append({
                    'text': text,
                    'name': name,
                    'identifier': identifier,
                    'type': annotation['infons']['type'],
                    'source': annotation['infons']['database'],
                    'pmid': nested_data['id'],
                    "location": f"({annotation['locations'][0]['offset']},{annotation['locations'][0]['length']})"
                })
    annotations_df = pd.DataFrame(annotations_list)
    return annotations_df, abstract_text, abstract_offset

# Helper function to extract the first element of a location string
def extract_first_element(text):
    pattern = re.compile(r"\(([^,]+),")
    match = pattern.search(text)
    if match:
        return match.group(1)
    else:
        return "No match found"

def get_bioNode(annotations_df):
    bioNode = annotations_df.groupby('name').agg({
        'text': 'first',
        'identifier': 'first',
        'type': 'first',
        'source': 'first',
        'pmid': 'first',
        'location': list,
        'offset': list
    }).reset_index(drop=True)
    bioNode['frequency'] = bioNode['location'].apply(len)
    return bioNode


# Load spaCy's English model
nlp = spacy.load("en_core_sci_lg")

def extract_entities(abstract_text):
    text = abstract_text

    # Process the text with spaCy
    doc = nlp(text)
    unique_entities = set()

    for ent in doc.ents:
      unique_entities.add(ent.text)

    return unique_entities

def remove_pubtator(entities, annotation_list):
    annotation_list_lower = [x.lower() for x in annotation_list]
    df_text_lower = set(annotation_list_lower)
    # New set to store results after filtering
    new_entities = set()
    # Loop through each entity and check against all texts in the DataFrame
    for entity in entities:
        # Check if any part of the entity text is present in the DataFrame's 'text' column (case-insensitive)
        if not any(df_text.lower() in entity.lower() for df_text in df_text_lower):
            new_entities.add(entity)
    return new_entities

def combine_indentical(new_entities):
    seen = set()
    result = []
    for word in new_entities:
        # Normalize word to lowercase for comparison
        lower_word = word.lower()
        if lower_word not in seen:
            seen.add(lower_word)
            result.append(word)  # Append the original word
    # sort the result
    result = sorted(result)
    return result

categories = "1. healthcare access and quality; 2. education access and quality; 3. social and community context; 4. economic stability; 5. neighborhood and built environment; 6. does not belong to categories 1-5"

def remove_verb(text, model="gpt-3.5-turbo"):
  response = client.chat.completions.create(
  model=model,  # or another model like GPT-4 if available
  messages=[
          {"role": "system", "content": "You are an AI trained to identify and remove phrases from a list if they are verbs, adverbs, or prepositions phrases. Delete the entire phrase."},
          {"role": "user", "content": f"""Given this list of phrases: {text}, remove each phrase entirely if it is not a pure noun phrase. 
           Please only return the list of phrases, do not return anything else."""},
      ],
      temperature=0
    )
  return response.choices[0].message.content

def classify_text(text, model="gpt-3.5-turbo"):
  response = client.chat.completions.create(
  model=model,  # or another model like GPT-4 if available
  messages=[
          {"role": "system", "content": "You are an AI assistant to answer question about biomedicine and Social Determinants of Health."},
          {"role": "user", "content": f"""Social Determinants of Health are non-medical conditions that affect one's health and impact health-related outcomes.
           Classify the given phrases: {text} into the most relevant categories: {categories}. 
           Exclude phrases that are solely verbs, adverbs, or prepositions. 
           Please only return the results in this form: 'phrase: the number of its category', each result on a separate line, do not return anything else."""},
      ],
      temperature=0
    )
  return response.choices[0].message.content

def process_results(input_string):
    #pattern = re.compile(r'^\s*[\w\s]+:\s*\d+\s*$', re.MULTILINE)
    pattern = re.compile(r"^\s*[\w\s'/()+&-]+:\s*\d+\s*$", re.MULTILINE)
    pairs = input_string.split('\n')
    # Parse these "key: value" strings into a dictionary
    data_dict = {}
    for pair in pairs:
        pair = pair.strip().strip("'\"")
        # check if the pair matches this format: 'key: value'
        if re.match(pattern, pair.strip()):
            key, value = pair.split(': ')
            data_dict[key.strip()] = int(value.strip())
        else:
            print(f"Warning: Input string does not match process_results.{pair}")

    # Load the dictionary into a DataFrame
    df = pd.DataFrame(list(data_dict.items()), columns=['text', 'type'])

    # Filter out rows where the value is 6
    result_df = df[df['type'] != 6].reset_index(drop=True)

    return result_df

def break_df(df):
    if len(df) > 0:
        type1 = df[df['type'] == 1].reset_index(drop=True)
        type2 = df[df['type'] == 2].reset_index(drop=True)
        type3 = df[df['type'] == 3].reset_index(drop=True)
        type4 = df[df['type'] == 4].reset_index(drop=True)
        type5 = df[df['type'] == 5].reset_index(drop=True)
    else:
        type1 = type2 = type3 = type4 = type5 = pd.DataFrame()
    
    return type1, type2, type3, type4, type5

sub_categories = ["1.Access to Health Services; 2.Access to Primary Care; 3.Health Literacy; 4.does not belong to categories 1-3",
                  "1.Early Childhood Development and Education; 2.Enrollment in Higher Education; 3.High School Graduation; 4.Language and Literacy; 5.does not belong to categories 1-4",
                  "1.Civic Participation; 2.Discrimination; 3.Incarceration; 4.Social Cohesion; 5.does not belong to categories 1-4",
                  "1.Employment; 2.Food Insecurity; 3.Housing Instability; 4.Poverty; 5.does not belong to categories 1-4",
                  "1.Access to Foods That Support Healthy Dietary Patterns; 2.Crime and Violence; 3.Environmental Conditions; 4.Quality of Housing; 5.does not belong to categories 1-4"]


def classify_sub(df, type, model="gpt-3.5-turbo"):
  response = client.chat.completions.create(
  model=model,  # or another model like GPT-4 if available
  messages=[
          {"role": "system", "content": "You are an AI assistant to answer question about biomedicine and Social Determinants of Health."},
          {"role": "user", "content": f"""Classify the given phrases: {df['text'].tolist()} into the most relevant categories: {sub_categories[type]}. 
           Exclude phrases that are solely verbs, adverbs, or prepositions. 
           Please only return the results in this form: 'phrase: the number of its category', each result on a separate line, do not return anything else."""},
      ],
      temperature=0
    )
  return response.choices[0].message.content

def process_results_sub(input_string, type):
    pattern = re.compile(r"^\s*[\w\s&'/()+&-]+:\s*\d+\s*$", re.MULTILINE)
    pairs = input_string.split('\n')
    # Parse these "key: value" strings into a dictionary
    data_dict = {}
    for pair in pairs:
        pair = pair.strip().strip("'\"")
        if re.match(pattern, pair.strip()):
            key, value = pair.split(': ')
            data_dict[key.strip()] = int(value.strip())
        else:
            print(f"Warning: Input string does not match process_results.{pair}")

    # Load the dictionary into a DataFrame
    df = pd.DataFrame(list(data_dict.items()), columns=['text', 'type'])

    if type == 0:
        result_df = df[df['type'] != 4].reset_index(drop=True)
    else:
        result_df = df[df['type'] != 5].reset_index(drop=True)

    return result_df

def process_sub(dfs, model):
    modell = model
    dfs1 = break_df(dfs)
    df = [0,0,0,0,0]
    for i in range(5):
        if len(dfs1[i]) > 0:
            temp = classify_sub(dfs1[i], i, modell)
            df[i] = process_results_sub(temp, i)
        else:
            df[i] = dfs1[i]
        # change 'type' to 'df_type'
        df[i].rename(columns={'type': 'sub_type'}, inplace=True)
        #add a new column'type' with the value of i+1
        df[i]['type'] = i+1
        
    #concate the 5 dataframes
    result_df = pd.concat([df[0], df[1], df[2], df[3], df[4]], axis=0).reset_index(drop=True)
    return result_df

def find_occurrences(text, paragraph,abstract_offset):
    stemmer = PorterStemmer()
    stemmed_text = stemmer.stem(text.lower())
    stemmed_paragraph = paragraph.lower()
    start = 0
    locations = []

    while True:
        start = stemmed_paragraph.find(stemmed_text, start)
        if start == -1:
            return locations
        
        # Calculate the start offset in the original paragraph
        original_start = start + abstract_offset
        locations.append(original_start)

        # Move past this match
        start += len(stemmed_text)
    
    return locations

def get_location(df, abstract_text, abstract_offset):
    df['offset'] = df['text'].apply(lambda x: find_occurrences(x, abstract_text,abstract_offset))
    df['frequency'] = df['offset'].apply(len)
    return df

def get_probability(biodf, sdohdf, abstract_text):
    df1 = biodf.drop(columns=['location','identifier','pmid'])
    df2 = sdohdf.drop(columns=['sub_type', 'type'])

    # Example paragraph
    paragraph = abstract_text

    # Tokenize paragraph into sentences
    sentences = sent_tokenize(paragraph)

    # Function to find sentences containing a given text
    def find_sentences_with_text(text, sentences):
        return {s for s in sentences if text.lower() in s.lower()}

    # Calculate conditional probabilities
    results = []
    for _, row1 in df1.iterrows():
        sentences_with_text1 = find_sentences_with_text(row1['text'], sentences)
        #text1_count = len(sentences_with_text1)

        for _, row2 in df2.iterrows():
            sentences_with_text2 = find_sentences_with_text(row2['text'], sentences)
            #text2_count = len(sentences_with_text2)
            sentences_with_both = sentences_with_text1.intersection(sentences_with_text2)

            if row1['frequency'] > 0:  # Avoid division by zero
                probability = len(sentences_with_both) / max(row1['frequency'], row2['frequency'])
                #probability = len(sentences_with_both) / row1['frequency']
            else:
                probability = 0

            results.append({
                'Bio': row1['text'],
                'SDoH factor': row2['text'],
                'probability': probability
            })

    # Create results DataFrame and sort by probability
    results_df = pd.DataFrame(results)
    # if results_df is empty, end the program
    if results_df.empty:
        return results_df
    results_df = results_df.sort_values(by='probability', ascending=False).reset_index(drop=True)

    return results_df

def threshold_probability(probability_df, bioNode, SDoH_df, threshold=0.5):
    filtered_results = probability_df[probability_df['probability'] >= threshold]
    # add the identifier and pmid from bioNode to the filtered results according to the Bio and text columns
    # full_result = pd.merge(filtered_results, bioNode, left_on='Bio', right_on='text', how='left')
    # full_result = pd.merge(full_result, SDoH_df, left_on='SDoH factor', right_on='text', how='left')
    #full_result = full_result[['Bio', 'SDoH factor', 'probability', 'identifier', 'pmid', 'sub_type', 'type']]
    
    # add the identifier and pmid from bioNode to the filtered results according to the Bio and text columns
    full_result = pd.merge(filtered_results, bioNode, left_on='Bio', right_on='text', how='left')
    full_result = pd.merge(full_result, SDoH_df, left_on='SDoH factor', right_on='text', how='left')
    full_result = full_result[['Bio', 'SDoH factor', 'probability', 'identifier', 'type_x', 'source', 'pmid', 'sub_type', 'type_y']]
    full_result = full_result.rename(columns={'type_x': 'Bio_type', 'type_y': 'SDoH_type', 'sub_type': 'SDoH_subtype', 'identifier': 'Bio_identifier'})
    return full_result

def find_relation(abstract_text, annotation_list, sdoh_list, model="gpt-3.5-turbo"):
  response = client.chat.completions.create(
  model=model,  # or another model like GPT-4 if available
  messages=[
          {"role": "system", "content": "You are an AI assistant to answer question about biomedicine and Social Determinants of Health."},
          {"role": "user", "content": f"""Read the following abstract, extract the relationships between each biomedicine entity and Social Determinants of Health entity.
           You can choose the relation from: (covaries, interacts, regulates, resembles, downregulates, upregulates, associates, binds, treats, palliates), or generate a new predicate to describe the relationship between the two entities.
           If there is no relationship between the two entities, please skip it.
           If no relationships are found, please return ''.
           Output all the extract triples in the format of "head | relation | tail". For example: "Alzheimer's disease | associates | memory deficits"
           Abstract: {abstract_text}
           biomedicine entity: {annotation_list}
           Social Determinants of Health entity: {sdoh_list} """},
      ],
      temperature=0
    )
  return response.choices[0].message.content

def process_relation(input_string):
    #print(input_string)
    pattern1 = re.compile(r'^\d+\. [^|]+\|[^|]+\|[^|]+$')
    pattern2 = re.compile(r'^[^|]+\|[^|]+\|[^|]+$')
    
    entries = input_string.split('\n')
    # Create a list of dictionaries from the entries
    data = []
    for entry in entries:
        if re.match(pattern1, entry.strip()):
            number, rest = entry.split('. ', 1)
            parts = rest.split(' | ')
        elif re.match(pattern2, entry.strip()):
            parts = entry.split(' | ')
        else:
            print("Warning: Entry does not match the expected format:", entry)
            continue
        if len(parts) != 3:
            print("Warning: Entry does not have exactly three parts:", entry)
            continue
        data.append({
            'Bio': parts[0].strip(),
            'Relationship': parts[1].strip(),
            'SDoH factor': parts[2].strip()
        })
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    return df

def gptr_df(df, annotations_df, sdoh):
    # # match df 'Bio' with annotation_df 'text', and add the corresponding 'identifier' and 'pmid' to the df
    # merged_df = pd.merge(df, annotations_df, left_on='Bio', right_on='text', how='left')
    # merged_df = pd.merge(merged_df,sdoh, left_on='SDoH factor', right_on='text', how='left')
    # # Select necessary columns
    # final_df = merged_df[['Bio', 'Relationship', 'SDoH factor', 'identifier', 'pmid', 'sub_type', 'type']]
    # # delete the duplicated rows
    # final_df = final_df.drop_duplicates()
    # final_df = final_df.drop_duplicates(subset=['SDoH factor', 'identifier'])
    # # reset the index
    # gpt_final_df = final_df.reset_index(drop=True)

    merged_df = pd.merge(df, annotations_df, left_on='Bio', right_on='text', how='left')
    merged_df = pd.merge(merged_df,sdoh, left_on='SDoH factor', right_on='text', how='left')
    #final_df = merged_df.drop(columns=['text_x', 'name', 'location', 'offset_x', 'offset_y', 'text_y', 'frequency'])
    final_df = merged_df.drop(columns=['text_x', 'name', 'location', 'offset','text_y'])
    final_df.drop_duplicates(inplace=True)
    final_df = final_df.drop_duplicates(subset=['SDoH factor', 'identifier'])
    final_df = final_df.rename(columns={'identifier': 'Bio_identifier', 'type_x': 'Bio_type', 'sub_type': 'SDoH_subtype', 'type_y': 'SDoH_type'})
    gpt_final_df = final_df.reset_index(drop=True)
    return gpt_final_df

# Main function to run the script
def main(folder_path, gptmodel):
    print(f"Processing data from {folder_path} using model {gptmodel}")
    #folder_path = input("Enter the folder path to JSON files: ")
    #gptmodel = input("Enter the GPT model name: ")
    os.makedirs(folder_path, exist_ok=True)

    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            annotations_df, abstract_text, abstract_offset = process_json(file_path)
            if annotations_df.empty:
                print(f"No annotations found in the file: {file}")
                continue
            annotations_df['offset'] = [int(extract_first_element(x)) for x in annotations_df['location']]
            annotation_list = []
            for text in annotations_df['text'].unique():
                annotation_list.append(text)
            bioNode = get_bioNode(annotations_df)
            unique_entities = extract_entities(abstract_text)
            new_entities = remove_pubtator(unique_entities, annotation_list)
            new_entities = combine_indentical(new_entities)
            if len(new_entities) == 0:
                print(f"No entities found in the file: {file}")
                continue
            demo = '; '.join(new_entities)
            re_v = remove_verb(demo, model=gptmodel)
            results = classify_text(re_v, model=gptmodel)
            processed_df = process_results(results)
            if processed_df is None:
                print(f"classify error in the file: {file}")
                continue
            elif processed_df.empty:
                print(f"No sdoh matched to category: {file}")
                continue
            sdoh = process_sub(processed_df, model=gptmodel)
            if sdoh.empty:
                print(f"No sdoh is matched to sub-category: {file}")
                continue
            sdoh_list = sdoh['text'].to_list()
            r = find_relation(abstract_text, annotation_list, sdoh_list, model=gptmodel)
            rdf = process_relation(r)
            if rdf.empty:
                print("No relationships found")
                continue
            # print("rdf:")
            # print(rdf)
            # print("annotations_df:")
            # print(annotations_df)
            # print("sdoh:")
            # print(sdoh)
            gpt_df = gptr_df(rdf, annotations_df, sdoh)

            SDoH_df = get_location(sdoh, abstract_text, abstract_offset)
            probability_df = get_probability(bioNode, SDoH_df, abstract_text)
            if probability_df.empty:
                print("No results found")
                continue
            coo_df = threshold_probability(probability_df, bioNode, SDoH_df, threshold=0.4)
            result_df = pd.merge(gpt_df, coo_df, on=['Bio', 'SDoH factor','Bio_identifier', 'Bio_type', 'source', 'pmid', 'SDoH_subtype', 'SDoH_type'], how='inner')
            # create and Save the results to a CSV file, but don't overwrite the file, add the results to the existing file
            gpt_df.to_csv('result/gptresult'+gptmodel+'.csv', mode='a', header=False, index=False)
            coo_df.to_csv('result/cooresult'+gptmodel+'.csv', mode='a', header=False, index=False)
            result_df.to_csv('result/result'+gptmodel+'.csv', mode='a', header=False, index=False)
    
    print("Results saved to 'result.csv'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <data_path> <model>")
        sys.exit(1)
    data_path = sys.argv[1]
    model = sys.argv[2]
    main(data_path, model)