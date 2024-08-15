import pdfplumber
import json
from transformers import GPT2Tokenizer
import requests
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_ACCESS_TOKEN")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# inference endpoint
API_URL = "https://bo6y25vl4oj219t5.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {hf_token}",
    "Content-Type": "application/json"
}

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# query inference endpoint
def query_inference_endpoint(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print("Response JSON:", response.json())
        return None
    return response.json()

# chunk text to stay within token limit
def chunk_text(text, chunk_size=800):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

# # detect section titles
# def detect_section_titles(text):
#     prompt = f"Extract section titles from the following text:\n\n{text}. THIS IS IMPORTANT: Section titles should not include the authors, affiliations, or acknowledgements. The section titles should be concise and not full sentences."
#     chunks = chunk_text(prompt)

#     generated_text = ""
#     for chunk in chunks:
#         response_json = query_inference_endpoint({"inputs": chunk, "parameters": {}})
#         if not response_json or 'generated_text' not in response_json[0]:
#             print("Error: 'generated_text' key not found in the response.")
#             print("Response JSON:", response_json)
#             continue
#         generated_text += response_json[0]['generated_text']

#     # regex to detect section titles
#     section_titles = re.findall(r'\n([A-Z][^\n]+)\n', generated_text)
#     # filter out irrelevant titles
#     filtered_titles = [title for title in section_titles if len(title.split()) > 2]
#     return filtered_titles

# # separate sections based on detected titles
# def separate_sections(text, section_titles):
#     section_patterns = [re.compile(rf'\b{re.escape(title)}\b', re.IGNORECASE) for title in section_titles]
#     section_texts = {title: "" for title in section_titles}

#     current_section = None
#     for line in text.split("\n"):
#         line = line.strip()
#         for i, pattern in enumerate(section_patterns):
#             if pattern.search(line):
#                 current_section = section_titles[i]
#                 break
#         if current_section:
#             section_texts[current_section] += line + "\n"
#     return section_texts

# generate comments for each section
def generate_comment(content):
    prompt = f"""
    You are a peer reviewer for a scientific paper. Your task is to provide detailed and constructive feedback on the paper. Please ensure your comments are professional, thorough, and helpful for the authors to improve their work.

    Content: {content}
    """

    chunks = chunk_text(prompt)

    generated_text = ""
    for chunk in chunks:
        response_json = query_inference_endpoint({"inputs": chunk, "parameters": {}})
        if not response_json or 'generated_text' not in response_json[0]:
            print("Error: 'generated_text' key not found in the response.")
            print("Response JSON:", response_json)
            continue
        generated_text += response_json[0]['generated_text']

    return generated_text.strip()

# revise paper based on criteria
def revise_paper(content, criteria):
    prompt = f"""
    You are a peer reviewer for a scientific paper. Your task is to revise the paper according to the provided criteria. Please ensure your revisions are professional, thorough, and helpful for the authors to improve their work.

    Criteria: {criteria}
    Content: {content}
    """

    chunks = chunk_text(prompt)

    generated_text = ""
    for chunk in chunks:
        response_json = query_inference_endpoint({"inputs": chunk, "parameters": {}})
        if not response_json or 'generated_text' not in response_json[0]:
            print("Error: 'generated_text' key not found in the response.")
            print("Response JSON:", response_json)
            continue
        generated_text += response_json[0]['generated_text']

    return generated_text.strip()

def load_journal_criteria(journal_name):
    with open(f'criteria/{journal_name}_criteria.json', 'r') as json_file:
        criteria = json.load(json_file)
    return criteria

def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main():
    pdf_path = "../data/sample_papers/co-2024.pdf"
    pdf_file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    text = extract_text_from_pdf(pdf_path)
    print("Extracted Text:")
    print(text[:1000])
    print("="*80)

    # Load criteria for the journal
    journal_name = input("Enter the journal name (e.g., 'nsf'): ")
    criteria = load_journal_criteria(journal_name)
    
    user_choice = input("Enter '1' to generate comments, '2' to revise paper according to journal criteria: ")

    if user_choice == '1':
        comment = generate_comment(text)
        comments_filename = f"{pdf_file_name}_comments.json"
        save_to_json({"comments": comment}, comments_filename)
        print(f"Comments saved to {comments_filename}")
    elif user_choice == '2':
        revised_text = revise_paper(text, criteria)
        revised_filename = f"{pdf_file_name}_revised.json"
        save_to_json({"revised_text": revised_text}, revised_filename)
        print(f"Revised paper exported as {revised_filename}")

if __name__ == "__main__":
    main()


