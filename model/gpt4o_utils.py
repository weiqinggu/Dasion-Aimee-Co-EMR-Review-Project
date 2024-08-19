import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()
    
def load_phq9_criteria(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# generate mental health report using GPT-4o
def generate_report(transcription_text, phq9_criteria):
    prompt = f"""
    I have a transcription of a patient's responses to a series of questions regarding mental health, and I also have the PHQ-9 criteria. 
    Please compare the patient's responses with the criteria and generate a report that includes:
    1. The total PHQ-9 score.
    2. The severity of depression based on the score.
    3. Any discrepancies or areas of concern between the patient's responses and the expected healthy responses.
    
    Here is the PHQ-9 criteria in JSON format:
    {json.dumps(phq9_criteria, indent=4)}

    Here is the transcription of the patient's responses:
    {transcription_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

def save_to_txt(report, filename):
    with open(filename, 'w') as file:
        file.write(report)
    
# main function
def main():
    transcription_path = "../transcription-depressed.txt"
    phq9_criteria_path = "../phq-9.json"
    
    file_name = os.path.splitext(os.path.basename(transcription_path))[0]
    
    transcription_text = read_txt_file(transcription_path)
    ph9_criteria = load_phq9_criteria(phq9_criteria_path)
    
    report = generate_report(transcription_text, ph9_criteria)
    
    output_filename = f"{file_name}_report.txt"
    save_to_txt(report, output_filename)
    print(f"Report saved to {output_filename}")

if __name__ == "__main__":
    main()
