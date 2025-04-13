from llm_utils import llm, get_embedding
# from llm import llm
from scraping_utils import parse_page, scrape_lawyer_profile

import csv
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk



def passes_criterion_2(text, query: str):
    system_prompt = "Check if the query matches the lawyer's profile. Respond with 'match' if relevant, otherwise 'no match'."
    user_prompt = f"Query: {query}\nLawyer's profile: {text}"

    response = llm(system_prompt=system_prompt, user_prompt=user_prompt)
    return "match" if response.lower() == "match" else "no match"

def passes_criterion(text, query: str):
    """
    Evaluate if a lawyer passes a given criterion based on their profile.

    Args:
        lawyer_url (str): URL of the lawyer's profile
        query (str): Criterion to evaluate against

    Returns:
        bool: True if lawyer passes the criterion, False otherwise
    """
    # text = parse_page(lawyer_url)


    system_prompt = """
    You are checking if the provided query matches relevant information in the given text.

    Respond concisely with a short and precise snippet of relevant information from the text if it matches the query.
    If there is no match, respond with "No matching information".
    """.strip()

    user_prompt = f"""
    Here is the query: {query}
    Here is the lawyer's profile:: {text}
    """.strip()

    response = llm(system_prompt=system_prompt, user_prompt=user_prompt)
    
    # system_prompt = """
    # You are evaluating a lawyer whether they pass a given criterion.
    
    # Respond in the following format:
    # <thinking>...</thinking>, within which you include your detailed thought process.
    # <answer>...</answer>, within which you include your final answer. "Pass" or "Fail".
    # """.strip()
    
    # user_prompt = f"""
    # Here is the query: {query}
    # Here is the lawyer's profile: {text}
    # """.strip()
    
    # response = llm(system_prompt=system_prompt, user_prompt=user_prompt)
    # return response.split('<answer>')[1].split('</answer>')[0].strip() == 'Pass'
    return response


def extract_name_from_url(url):
    # Extract the lawyer's name from the URL
    return os.path.basename(url).replace('-', ' ').title()


# def clean_text(data):
#     # Remove newline characters, tabs, and excess spaces
#     cleaned_data = re.sub(r'\n+', ' ', data)  # Replace multiple newlines with a single space
#     cleaned_data = re.sub(r'\t+', ' ', cleaned_data)  # Replace multiple tabs with a single space
#     cleaned_data = re.sub(r'\s+', ' ', cleaned_data)  # Replace multiple spaces with a single space
#     cleaned_data = cleaned_data.strip()  # Remove leading/trailing spaces
#     return cleaned_data

def clean_text(data):
    """
    Clean the text data by removing newline characters, tabs, and excess spaces.
    This function will handle both string and list types within the data dictionary.

    Args:
        data (dict): The dictionary containing scraped data.

    Returns:
        dict: The cleaned dictionary with all values cleaned.
    """
    
    # Iterate through all keys and clean their respective values
    for key, value in data.items():
        if isinstance(value, str):
            # Clean the string value
            cleaned_value = re.sub(r'\n+', ' ', value)  # Replace multiple newlines with a single space
            cleaned_value = re.sub(r'\t+', ' ', cleaned_value)  # Replace multiple tabs with a single space
            cleaned_value = re.sub(r'\s+', ' ', cleaned_value)  # Replace multiple spaces with a single space
            cleaned_value = cleaned_value.strip()  # Remove leading/trailing spaces
            data[key] = cleaned_value
        elif isinstance(value, list):
            # If the value is a list, clean each string in the list
            data[key] = [re.sub(r'\n+', ' ', item) for item in value]  # Clean each string in the list
            data[key] = [re.sub(r'\t+', ' ', item) for item in data[key]]
            data[key] = [re.sub(r'\s+', ' ', item) for item in data[key]]
            data[key] = [item.strip() for item in data[key]]  # Remove leading/trailing spaces for each item in the list
    
    return data

def remove_website_boilerplate(lawyer_json_list):
    """
    Removes footers, legal disclaimers, and other website components from the 'data' field
    of each lawyer's JSON entry, while retaining relevant sections.

    Args:
        lawyer_json_list (list): A list of dictionaries, each containing 'name' and 'data' keys.

    Returns:
        list: A list of dictionaries with the cleaned 'data' field.
    """
    cleaned_lawyers = []

    for lawyer in lawyer_json_list:
        name = lawyer.get('name', 'Unknown')
        data = lawyer.get('data', '')

        # Remove footers, headers, and website navigation components using regular expressions
        # Define patterns to remove
        patterns_to_remove = [
            # Header navigation
            r"Skip to main content.*?See all results for \"\" Lawyers",
            # Footer navigation and legal disclaimer
            r"Back to top.*?Prior results do not guarantee a similar outcome\.",
            # Duplicate navigation elements
            r"Davis Polk Explore.*?Subscribe Receive insights from Davis Polk",
            # Download address card and print options
            r"Download address card Print this page",
            # Search bar
            r"Clear \| Search",
            # Any remaining navigation or footer elements
            r"Davis Polk.*?Contact Legal Privacy Notice.*?Cookie Settings Subscribe Receive insights from Davis Polk",
        ]

        # Apply all patterns to remove boilerplate
        for pattern in patterns_to_remove:
            data = re.sub(pattern, '', data, flags=re.DOTALL | re.IGNORECASE)

        # Normalize whitespace: replace multiple spaces/newlines with a single newline
        data = re.sub(r'\s{2,}', '\n', data).strip()

        # Append the cleaned data back to the list
        cleaned_lawyers.append({
            'name': name,
            'data': data
        })

    return cleaned_lawyers

def scrape_and_save_lawyer_data(input_file, output_file):
    with open(input_file, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            url = row[0]  # Assuming the URL is in the first column
            # lawyer_name = extract_name_from_url(url)
            lawyer_data = scrape_lawyer_profile(url)

            # If the page data is None (e.g., 403 error or failed fetch), skip it
            if lawyer_data is None:
                continue
            
            lawyer_data = clean_text(lawyer_data)
            
            # Create a dictionary with the lawyer's name and scraped data
            # lawyer_info = {
            #     'name': lawyer_name,
            #     'data': lawyer_data
            # }

            # lawyer_info = remove_website_boilerplate([lawyer_info])
            
            # Append the data to a JSON file
            # with open(output_file, 'a') as f_out:
            #     json.dump(lawyer_info[0], f_out)
            #     f_out.write('\n')  # Add a newline for readability

            # Directly save the lawyer's data without extra wrapping
            with open(output_file, 'a', encoding='utf-8') as f_out:
                json.dump(lawyer_data, f_out, ensure_ascii=False, indent=4)
                f_out.write('\n')  # Add a newline for readability
            
            print(f"Data for {lawyer_data['Name']} has been saved to {output_file}")


#  EMBEDDINGs


def extract_sections(lawyer_data):
    sections = {
        "Description": " ".join(lawyer_data.get("Description", [])),
        "Capabilities": " ".join(lawyer_data.get("Capabilities", [])),
        "Experience": " ".join(lawyer_data.get("Experience", [])),
        "News": " ".join(lawyer_data.get("News", [])),
        "Insights": " ".join(lawyer_data.get("Insights", [])),
        "Recognition": " ".join(lawyer_data.get("Recognition", []))
    }
    return {k: v for k, v in sections.items() if v.strip()}

def create_embeddings(lawyers_data):
    embeddings = {}
    for lawyer in lawyers_data:
        name = lawyer["Name"]
        sections = extract_sections(lawyer)
        section_texts = list(sections.values())
        section_embeddings = get_embedding(section_texts)
        embeddings[name] = dict(zip(sections.keys(), section_embeddings))
    return embeddings

def save_embeddings(embeddings, file_path):
    with open(file_path, 'w') as file:
        json.dump(embeddings, file, indent=4)





def find_top_similarity(lawyer_embeddings, query_embedding):
    top_score = -1
    top_section = ""
    for section, embedding in lawyer_embeddings.items():
        similarity = cosine_similarity(np.array(embedding), query_embedding)
        if similarity > top_score:
            top_score = similarity
            top_section = section
    return top_section, top_score

def process_lawyers(lawyer_embeddings, query_embedding):
    results = []
    for lawyer_name, sections in lawyer_embeddings.items():
        top_section, top_score = find_top_similarity(sections, query_embedding)
        results.append((lawyer_name, top_section, top_score))
    return sorted(results, key=lambda x: x[2], reverse=True)[:10]

def print_results(results):
    for lawyer_name, top_section, top_score in results:
        print(f"Lawyer: {lawyer_name}")
        print(f"Top section: {top_section}")
        print(f"Similarity score: {top_score:.4f}")
        print()

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# Function 2: Chunk Text
# def chunk_text(text, chunk_size=300, overlap=50):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = ' '.join(words[i:i + chunk_size])
#         chunks.append(chunk)
#     return chunks


# def chunk_text(data, chunk_size=500):
#     """
#     Splits the lawyer data into chunks of specified token or word count.
#     If sections are missing, it treats the entire text as one chunk.
    
#     Args:
#         data (str): The text to be chunked.
#         chunk_size (int): Maximum length of each chunk (in terms of tokens/words).
        
#     Returns:
#         list: A list of text chunks.
#     """
#     # Define a regex pattern to look for common section headers
#     section_splitter = r"(Experience|Insights|Education|Capabilities|Qualifications|Languages|Prior experience)"
    
#     # If sections are present, split based on the headers; otherwise, treat the entire text as one section
#     if re.search(section_splitter, data):  # Check if any section headers exist
#         sections = re.split(section_splitter, data)
#     else:
#         # If no specific section headers, treat the whole data as a single section
#         sections = [data]

#     # Filter out empty sections and strip leading/trailing whitespace
#     sections = [section.strip() for section in sections if section.strip()]
    
#     # Initialize chunks and a temporary string to accumulate text
#     chunks = []
#     current_chunk = ""
    
#     # Iterate over the sections, and build chunks
#     for section in sections:
#         # Check if adding this section to the current chunk exceeds the chunk size
#         if len(current_chunk.split()) + len(section.split()) > chunk_size:
#             # If chunk is too large, append the current chunk and start a new one
#             chunks.append(current_chunk.strip())
#             current_chunk = section
#         else:
#             # Otherwise, add the section to the current chunk
#             current_chunk += " " + section
    
#     # Add the last chunk if there is any leftover text
#     if current_chunk.strip():
#         chunks.append(current_chunk.strip())
    
#     return chunks



# # Function 4: Save Embeddings to File
# def save_embeddings(embeddings_dict, file_name='lawyer_embeddings.npz'):
#     np.savez(file_name, **embeddings_dict)
#     print(f"Embeddings have been saved to {file_name}")

# # Function 5: Main Function to Orchestrate the Process
# def process_and_save_embeddings(json_file_path, chunk_size=300, overlap=50):
#     data = load_data(json_file_path)
#     embeddings_dict = {}
    
#     for lawyer_info in data:
#         lawyer_name = lawyer_info['name']
#         scraped_text = lawyer_info['data']
#         chunks = chunk_text(scraped_text)
#         chunk_embeddings = get_embedding(chunks)
#         avg_embedding = np.mean(chunk_embeddings, axis=0)  # Average embedding for each lawyer
#         embeddings_dict[lawyer_name] = avg_embedding
    
#     save_embeddings(embeddings_dict)


# # Function to load embeddings from .npz file
# def load_embeddings(file_name):
#     data = np.load(file_name)
#     embeddings_dict = {key: value for key, value in data.items()}
#     print(f"Embeddings have been loaded from {file_name}")
#     return embeddings_dict

# # Function to normalize embeddings
# def normalize_embedding(embedding):
#     norm = np.linalg.norm(embedding)
#     return embedding / norm if norm > 0 else embedding

# # Function to compute cosine similarity with normalization
# def compute_cosine_similarity(query_embedding, lawyer_embedding):
#     query_embedding = normalize_embedding(query_embedding)
#     lawyer_embedding = normalize_embedding(lawyer_embedding)
#     return cosine_similarity([query_embedding], [lawyer_embedding])[0][0]


# # Main Function: User Search and Cosine Similarity Match with Threshold
# def search_lawyers_by_query_with_threshold(file_name, query, threshold):
#     # Load the pre-generated embeddings from the .npz file
#     embeddings_dict = load_embeddings(file_name)
    
#     # Convert the user's query into an embedding
#     query_chunks = chunk_text(query)  
#     query_embeddings = get_embedding(query_chunks)  
#     query_avg_embedding = np.mean(query_embeddings, axis=0) 
    
#     # List to store names of lawyers with cosine similarity above the threshold
#     matched_lawyers = []
    
#     # Iterate through the embeddings dictionary to find matching lawyers
#     for lawyer_name, lawyer_embedding in embeddings_dict.items():
#         similarity = compute_cosine_similarity(query_avg_embedding, lawyer_embedding)
#         print(lawyer_name, similarity)
        
#         # If the similarity score exceeds the threshold, add the lawyer to the list
#         if similarity >= threshold:
#             matched_lawyers.append((lawyer_name, similarity))
    
#     # Sort matched lawyers by similarity score in descending order
#     matched_lawyers.sort(key=lambda x: x[1], reverse=True)
    
#     if matched_lawyers:
#         print("Lawyers matching your query (sorted by similarity):")
#         for lawyer_name, similarity in matched_lawyers:
#             print(f"{lawyer_name} - Similarity: {similarity:.2f}")
#     else:
#         print("No lawyers found with sufficient match for your query.")


# Function 1: Load JSON Data
def load_data(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        # Split the content by '}' and remove empty strings
        json_strings = [s.strip() for s in content.split('}') if s.strip()]
        # Add the closing brace back and parse each object
        lawyers_data = [json.loads(s + '}') for s in json_strings]
    return lawyers_data


# Function to combine Description and Experience into a single string
def combine_description_and_experience(lawyer):
    description = lawyer.get('Description', 'No description available.')
    experience = lawyer.get('Experience', [])
    if isinstance(experience, str):
        experience = [experience]
    elif not isinstance(experience, list):
        experience = []
    
    combined_text = f"{description} Experience: {'; '.join(experience)}"
    return combined_text


# Function to generate and save name and embeddings
def generate_and_save_name_embeddings(file_path, output_file):
    # Load data from the JSON file
    lawyers_data = load_data(file_path)
    
    # Prepare data for embedding
    names = []
    texts_to_embed = []
    
    for lawyer in lawyers_data:
        name = lawyer.get('Name', 'Unknown')
        combined_text = combine_description_and_experience(lawyer)
        names.append(name)
        texts_to_embed.append(combined_text)
    
    # Generate embeddings for all texts at once
    embeddings = get_embedding(texts_to_embed)
    
    # Create dictionary with names and embeddings
    name_embedding_dict = dict(zip(names, embeddings))
    
    # Save the name and embeddings to the output file
    with open(output_file, 'w') as f:
        json.dump(name_embedding_dict, f, indent=4)
    
    print(f"Name and embeddings have been saved to {output_file}")
    return name_embedding_dict

# Calculate cosine similarity and sort results
def get_sorted_similarity_scores(query_embedding, lawyer_embeddings):
    similarities = []

    # Convert query_embedding to a numpy array
    query_embedding = np.array(query_embedding)

    for name, embedding in lawyer_embeddings.items():
        # similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        # similarities[name] = similarity
        embedding = np.array(embedding)
        # Calculate dot product
        dot_product = np.dot(query_embedding, embedding)
        
        # Calculate Euclidean distance
        euclidean_distance = np.linalg.norm(query_embedding - embedding)

        # Append results as a tuple (name, dot product, Euclidean distance)
        similarities.append((name, dot_product, euclidean_distance))

    # Sort by dot product in descending order and then by Euclidean distance in ascending order
    sorted_by_dot_product = sorted(similarities, key=lambda x: x[1], reverse=True)
    sorted_by_euclidean_distance = sorted(similarities, key=lambda x: x[2])

    return sorted_by_dot_product, sorted_by_euclidean_distance
    # return sorted(similarities.items(), key=lambda x: x[1], reverse=True)


# Main function to process query and print results
def process_query_and_print_results(query, embeddings_file, original_data_file, similarity_threshold=0.5):
    # Load embeddings and original data
    with open(embeddings_file, 'r') as f:
        lawyer_embeddings =  json.load(f)
    original_data = load_data(original_data_file)

    # Get query embedding
    query_embedding = get_embedding([query])[0]

    # Get sorted similarity scores
    sorted_by_dot_product, sorted_by_euclidean_distance = get_sorted_similarity_scores(query_embedding, lawyer_embeddings)

    print("\nSorted by Dot Product:")
    for name, dot_product, _ in sorted_by_dot_product:
        lawyer_data = next((lawyer for lawyer in original_data if lawyer['Name'] == name), None)
        if lawyer_data:
             #Pass the lawyer's data and query to passes_criterion function
            print(f"{name}: {dot_product:.4f}")
            match_result = passes_criterion(lawyer_data, query)
            
            # Pass the lawyer's data and query to passes_criterion function
            if match_result != "No matching information." :
                print(f"{name}: {dot_product:.4f} - {match_result}")
            else:
                print("No matches found.")
                break  # If criterion fails, stop processing further
        else:
            print(f"Data for {name} not found in original dataset.")

    print("\nSorted by Euclidean Distance:")
    for name, _, euclidean_distance in sorted_by_euclidean_distance:
        lawyer_data = next((lawyer for lawyer in original_data if lawyer['Name'] == name), None)
        if lawyer_data:
             #Pass the lawyer's data and query to passes_criterion function
            print(f"{name}: {euclidean_distance:.4f}")
            match_result = passes_criterion(lawyer_data, query)
            
            # Pass the lawyer's data and query to passes_criterion function
            if match_result != "No matching information." :
                print(f"{name}: {euclidean_distance:.4f} - {match_result}")
            else:
                print("No matches found.")
                break  # If criterion fails, stop processing further
        else:
            print(f"Data for {name} not found in original dataset.")

    # # Process results
    # for name, similarity in sorted_similarities:
    #     lawyer_data = next((lawyer for lawyer in original_data if lawyer['Name'] == name), None)
    #     if lawyer_data:
    #          #Pass the lawyer's data and query to passes_criterion function
    #         print(f"{name}: {similarity:.4f}")
    #         match_result = passes_criterion(lawyer_data, query)
            
    #         # Pass the lawyer's data and query to passes_criterion function
    #         if match_result != "No matching information." :
    #             print(f"{name}: {similarity:.4f} - {match_result}")
    #         else:
    #             print("No matches found.")
    #             break  # If criterion fails, stop processing further
    #     else:
    #         print(f"Data for {name} not found in original dataset.")




##             BM25 ALGO

def preprocess_data(lawyers_data):
    stop_words = set(stopwords.words('english'))
    lawyer_docs = []
    lawyer_names = []
    
    for lawyer in lawyers_data:
        name = lawyer['name']
        data = lawyer['data']
        tokens = word_tokenize(data.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        lawyer_docs.append(tokens)
        lawyer_names.append(name)
    
    return lawyer_docs, lawyer_names

def create_bm25_index(lawyer_docs):
    return BM25Okapi(lawyer_docs, k1=2.0, b=1.09)

def search_lawyers(query, bm25, lawyer_names, stop_words):
    query_tokens = word_tokenize(query.lower())
    query_tokens = [token for token in query_tokens if token.isalnum() and token not in stop_words]
    
    scores = bm25.get_scores(query_tokens)
    ranked_lawyers = sorted(zip(lawyer_names, scores), key=lambda x: x[1], reverse=True)
    
    return ranked_lawyers

# def print_results(matched_lawyers, lawyers_data, user_query):
    print("\nMatched lawyers:")
    if matched_lawyers:
        for name, score in matched_lawyers:
            if score > 0:  # Only show lawyers with a positive match score
                # Find the lawyer's data from the JSON file
                lawyer_entry = next((entry for entry in lawyers_data if entry["name"] == name), None)

                if lawyer_entry:
                    # Get the lawyer's data
                    data = lawyer_entry["data"]

                    # Pass the lawyer's data and query to passes_criterion function
                    match_result = passes_criterion(data, user_query)
                    
                    # Pass the lawyer's data and query to passes_criterion function
                    if match_result != "No matching information." :
                        print(f"{name}: {score:.4f} - {match_result}")
                    else:
                        print("No matches found.")
                        break  # If criterion fails, stop processing further
                
                # print(f"{name}: {score:.4f}")
    else:
        print("No matches found.")
    print()  # Add a blank line for readability


def main():
    """
    Takes in a string as a query and returns the list of lawyers.

    Args:
        query (str): The search query.

    Returns:
        list: A list of lawyers matching the query.
    """
    # TODO: Implement the search functionality

    # Load the CSV file containing lawyer URLs (adjust the file path as needed)
    # df = pd.read_csv('lawyers.csv', header=None, names=['profile_url'])

    # # List to store matching lawyer URLs
    # matching_lawyers = []

    # # Loop through each URL and check if it meets the criteria
    # for url in df['profile_url']:
    #     if passes_criterion(url, query):  # Call the function with the current URL and query
    #         matching_lawyers.append(url)  # Add the matching URL to the list

    # # Print the results
    # print("Matching Lawyers:")
    # for lawyer_url in matching_lawyers:
    #     print(lawyer_url)

    
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    file_path = 'scraped_data.json'
    lawyers_data = load_data(file_path)
    lawyer_docs, lawyer_names = preprocess_data(lawyers_data)
    bm25 = create_bm25_index(lawyer_docs)
    stop_words = set(stopwords.words('english'))

    while True:
        user_query = input("Enter your search query (or 'quit' to exit): ")
        
        if user_query.lower() == 'quit':
            print("Thank you for using the lawyer search. Goodbye!")
            break
        
        matched_lawyers = search_lawyers(user_query, bm25, lawyer_names, stop_words)
        print_results(matched_lawyers, lawyers_data, user_query)
    # print(parse_page())
    # return []

# Function to find the top experience and score for each lawyer
def find_top_experience(query_embed, lawyer_embeddings, lawyer_experience):
    top_experiences = []

    # Ensure the query embed is a 2D array
    query_embed_reshaped = np.reshape(query_embed, (1, -1))

    for emb, experiences in zip(lawyer_embeddings, lawyer_experience):
        # Convert emb to a numpy array and ensure it's 2D
        emb_array = np.array(emb)
        if emb_array.ndim == 1:
            emb_array = emb_array.reshape(1, -1)
        
        # Ensure emb_array is 2D if it consists of multiple vectors
        elif emb_array.ndim > 2:
            emb_array = emb_array.squeeze()  # Remove extra dimensions if needed

        # Calculate cosine similarity
        similarities = cosine_similarity(query_embed_reshaped, emb_array)[0]

        # Get the index of the most similar experience
        top_index = np.argmax(similarities)
        top_experience = experiences[top_index]
        top_score = similarities[top_index]

        # Append the result
        top_experiences.append((top_experience, top_score))

    return top_experiences


def evaluate_lawyers(lawyers_data, query):
    matched_lawyers = []
    
    for lawyer in lawyers_data:
        # Combine all sections into one text for evaluation
        combined_text = " ".join(lawyer.get("Description", []) + 
                                  lawyer.get("Capabilities", []) + 
                                  lawyer.get("Experience", []) + 
                                  lawyer.get("Education", []) + 
                                  lawyer.get("Insights", []) + 
                                  lawyer.get("Recognition", []))
        # print(combined_text)
        # Check if the lawyer passes the criterion
        result = passes_criterion(combined_text, query)
        
        if result == "match":
            matched_lawyers.append(lawyer["Name"])
    
    return matched_lawyers


def load_lawyer_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
        # Split multiple JSON objects and load them
        lawyers = []
        for line in data.strip().split('\n'):
            if line:  # Avoid empty lines
                lawyers.append(json.loads(line))
        return lawyers

if __name__ == '__main__':
    # user_query = input('Enter your search term: ')
    scraped_data = "scraped_data_2.json"
    # input_file = 'test2.csv'

    # scrape_and_save_lawyer_data(input_file,scraped_data)
    
    # Path to save the new JSON file with names and embeddings
    # output_file = 'lawyer_embeddings.json'

    # lawyers_data = load_data(scraped_data)
    # embeddings = create_embeddings(lawyers_data)
    # save_embeddings(embeddings, output_file)

    # print(f"Embeddings saved to {output_file}")

    # process_and_save_embeddings('scraped_data.json')
    # threshold = 0.7  # Set threshold to 0.7 (you can change this value)
    # search_lawyers_by_query_with_threshold('lawyer_embeddings.npz', user_query, threshold)
    # print(main(user_query))
    # data = load_data(scraped_data)
    # generate_and_save_name_embeddings(scraped_data, output_file)
    # main()
    # process_query_and_print_results(user_query,output_file,scraped_data)

    # def load_lawyer_embeddings(file_path):
    #     with open(file_path, 'r') as f:
    #         return json.load(f)

    # lawyer_embeddings = load_lawyer_embeddings('lawyer_embeddings.json')
    # query_embed = get_embedding(["worked on a case with tv network."])[0]
    # query_embedding = np.array(query_embed)
    # results = process_lawyers(lawyer_embeddings, query_embedding)
    # print("Top 10 Match: ")
    # print_results(results)
    lawyers_data = load_data(scraped_data)
    
    # Define your query
    query = "worked on a case with a TV network."  # Replace with your actual query
    
    # Evaluate lawyers against the query
    matched_lawyers = evaluate_lawyers(lawyers_data, query)
    
    # Print matched lawyers
    if matched_lawyers:
        print("Matched Lawyers:")
        for lawyer_name, snippet in matched_lawyers:
            print(f"{lawyer_name}: {snippet}")
    else:
        print("No lawyers matched the criterion.")

    
