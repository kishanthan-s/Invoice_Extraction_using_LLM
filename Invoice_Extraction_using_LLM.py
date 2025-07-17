!pip install PyMuPDF

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import fitz
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text
pdf_path = "/content/1XXXX0340_1223_AC_DET.pdf"  # Replace with your actual PDF path
pdf_text = extract_text_from_pdf(pdf_path)
#print(pdf_text)

template = f"""
You are an intelligent parser that extracts structured data from raw invoice text extracted from a PDF. The content may be jumbled, with fields not in a consistent order.

Your task is to extract **all identifiable fields and their values** in the format of clearly labeled key-value pairs.

Guidelines:
- Output must be in dictionary-style format: {{'Field Name': 'Value'}}
- Remove any currency symbols (e.g., Rs., $, €, etc.) from numeric values
- Infer the correct field names even if their labels are slightly different or scattered
- Avoid repeating values
- Include all fields you can recognize (e.g., Invoice Number, Invoice Date, Customer Name, Address, Mobile Number, Billing Period, Previous Due Amount, Charges, Taxes, VAT, Total Payable, Due Date, Loyalty Points, etc.)

Format:
{{
  'Field 1': 'Value 1',
  'Field 2': 'Value 2',
  ...
}}

Text to extract from:
<<<
{pdf_text}
>>>

Only return the structured output as a Python dictionary. Do not add explanations or commentary.
"""

!huggingface-cli login

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

#This will print the template and the output
inputs = tokenizer(template, return_tensors="pt", truncation=True).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0)

# Decode result
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

from transformers import logging
logging.set_verbosity_error()

# Tokenize input and get input token length
inputs = tokenizer(template, return_tensors="pt", truncation=True).to("cuda")
input_length = inputs['input_ids'].shape[1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)

# Slice to get only newly generated tokens (after the prompt)
generated_tokens = outputs[0][input_length:]

# Decode only the new tokens
result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(result)

import json
import re
import csv

import re
import csv

# Your generated raw text after decoding:
raw_text = result
print(raw_text)
# Extract from first '{' to last '}' (or end of last key-value pair)
start = raw_text.find('{')
end = raw_text.rfind('}')
if start == -1 or end == -1 or end <= start:
    print("No valid dictionary braces found in output.")
    data_dict = {}
else:
    dict_str = raw_text[start:end+1]

    # Remove trailing incomplete entries after the last complete key-value pair
    # We'll match key-value pairs like: 'key': 'value'
    pattern_kv = r"'([^']+)'\s*:\s*'([^']*)'"
    matches = re.findall(pattern_kv, dict_str)

    if not matches:
        print("No key-value pairs found.")
        data_dict = {}
    else:
        # Build dict from all matched pairs
        data_dict = dict(matches)

# Write to CSV if we have data
if data_dict:
    with open("output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Key", "Value"])
        for k, v in data_dict.items():
            writer.writerow([k, v])
    print("Data written to output.csv")
else:
    print("No valid data to write to CSV.")

