from fastapi import FastAPI, UploadFile, File
from google.cloud import documentai_v1beta3 as documentai
import os
from typing import Dict
import re

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./creds/grant01-joby.json"

app = FastAPI()
processor_name = "projects/332125695616/locations/us/processors/a6bceed480e9d614"

def process_document(processor_name: str, file_path: str) -> documentai.Document:
    client = documentai.DocumentProcessorServiceClient()
    
    # Read the file into memory
    with open(file_path, "rb") as f:
        document_content = f.read()

    # Configure the request
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(
            content=document_content,
            mime_type="application/pdf"
        )
    )

    result = client.process_document(request=request)
    return result.document

def extract_text_from_document(document: documentai.Document) -> str:
    text_content = ""
    for page in document.pages:
        for token in page.tokens:
            text_anchor = token.layout.text_anchor
            start_index = text_anchor.text_segments[0].start_index
            end_index = text_anchor.text_segments[0].end_index
            text_content += document.text[start_index:end_index] + " "
    return text_content.strip()

def extract_income_certificate_details(text: str) -> Dict:
    details = {}

    # Updated regex patterns
    name_pattern = re.compile(r'Name\s*:\s*([A-Za-z\s]+)')
    aadhaar_pattern = re.compile(r'Aadhaar\s*Number\s*:\s*([\d\s]{12})')
    dob_pattern = re.compile(r'Date\s*of\s*Birth\s*:\s*(\d{2}/\d{2}/\d{4})')

    name_match = name_pattern.search(text)
    aadhaar_match = aadhaar_pattern.search(text)
    dob_match = dob_pattern.search(text)

    if name_match:
        details['Name'] = name_match.group(1).strip()
    if aadhaar_match:
        details['Aadhaar_number'] = aadhaar_match.group(1).strip().replace(' ', '')
    if dob_match:
        details['Date_of_birth'] = dob_match.group(1).strip()

    return details

def extract_application_form_details(text: str) -> Dict:
    details = {}

    # Updated regex patterns
    applicant_name_pattern = re.compile(r'Applicant\s*Name\s*:\s*([A-Za-z\s]+)')
    father_husband_name_pattern = re.compile(r'Father\/Husband\s*Name\s*:\s*([A-Za-z\s]+)')
    dob_pattern = re.compile(r'Date\s*of\s*Birth\s*:\s*(\d{2}/\d{2}/\d{4})')
    aadhaar_pattern = re.compile(r'Aadhaar\s*Number\s*:\s*([\d\s]{12})')
    mobile_pattern = re.compile(r'Mobile\s*Number\s*:\s*([\d\s]{10})')
    ration_card_pattern = re.compile(r'Ration\s*Card\s*Number\s*:\s*([A-Za-z\d\s]+)')

    applicant_name_match = applicant_name_pattern.search(text)
    father_husband_name_match = father_husband_name_pattern.search(text)
    dob_match = dob_pattern.search(text)
    aadhaar_match = aadhaar_pattern.search(text)
    mobile_match = mobile_pattern.search(text)
    ration_card_match = ration_card_pattern.search(text)

    if applicant_name_match:
        details['Applicant_Name'] = applicant_name_match.group(1).strip()
    if father_husband_name_match:
        details['Father_Husband_Name'] = father_husband_name_match.group(1).strip()
    if dob_match:
        details['Date_of_birth'] = dob_match.group(1).strip()
    if aadhaar_match:
        details['Aadhaar_Number'] = aadhaar_match.group(1).strip().replace(' ', '')
    if mobile_match:
        details['Mobile_number'] = mobile_match.group(1).strip().replace(' ', '')
    if ration_card_match:
        details['Ration_card'] = ration_card_match.group(1).strip()

    return details

@app.post("/process-income-cert/")
async def process_income(application_form: UploadFile = File(...), aadhaar: UploadFile = File(...)):
    application_path = f"/tmp/{application_form.filename}"
    aadhaar_path = f"/tmp/{aadhaar.filename}"

    with open(application_path, "wb") as f:
        f.write(await application_form.read())
    
    with open(aadhaar_path, "wb") as f:
        f.write(await aadhaar.read())

    application_document = process_document(processor_name, file_path=application_path)
    aadhaar_document = process_document(processor_name, file_path=aadhaar_path)

    # Extract readable text from documents
    application_text = extract_text_from_document(application_document)
    aadhaar_text = extract_text_from_document(aadhaar_document)

    print("Extracted Text from Application Document:")
    print(application_text)

    print("Extracted Text from Aadhaar Document:")
    print(aadhaar_text)

    if application_text:
        application_data = extract_application_form_details(application_text)
        os.remove(application_path)
    else:
        return {"error": "Application data couldn't be parsed"}
    
    if aadhaar_text:
        aadhaar_data = extract_income_certificate_details(aadhaar_text)
        os.remove(aadhaar_path)
    else:
        return {"error": "Issue with the Aadhaar card. Please try again"}
    
    return {"application_data": application_data, "aadhaar_data": aadhaar_data}





# from fastapi import FastAPI, UploadFile, File
# from google.cloud import documentai_v1beta3 as documentai
# import os
# from typing import Dict
# import re

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./creds/grant01-joby.json"

# app = FastAPI()
# processor_name = "projects/332125695616/locations/us/processors/a6bceed480e9d614"

# def detect_language(text: str) -> str:
#     try:
#         return detect(text)
#     except:
#         return "unknown"

# def process_document(processor_name: str, file_path: str) -> documentai.Document:
#     client = documentai.DocumentProcessorServiceClient()
    
#     # Read the file into memory
#     with open(file_path, "rb") as f:
#         document_content = f.read()

#     # Configure the request
#     request = documentai.ProcessRequest(
#         name=processor_name,
#         raw_document=documentai.RawDocument(
#             content=document_content,
#             mime_type="application/pdf"
#         )
#     )

#     result = client.process_document(request=request)

#     # Extract text for language detection
#     document_text = result.document.text
#     language = detect_language(document_text)
#     print(f"Detected language: {language}")

#     return result.document

# def extract_income_certificate_details(text: str) -> Dict:
#     details = {}
#     print(f"Extracting income certificate details from text: {text[:100]}...")  # Debugging statement

#     # Regex patterns to extract the relevant information
#     name_pattern = re.compile(r'Name\s*:\s*([A-Za-z\s]+)', re.IGNORECASE)
#     aadhaar_pattern = re.compile(r'Aadhaar\s*Number\s*:\s*([\d\s]+)', re.IGNORECASE)
#     dob_pattern = re.compile(r'Date\s*of\s*Birth\s*:\s*(\d{2}/\d{2}/\d{4})', re.IGNORECASE)

#     name_match = name_pattern.search(text)
#     aadhaar_match = aadhaar_pattern.search(text)
#     dob_match = dob_pattern.search(text)

#     if name_match:
#         details['Name'] = name_match.group(1).strip()
#     else:
#         print("Name not found in the income certificate text.")

#     if aadhaar_match:
#         details['Aadhaar_number'] = aadhaar_match.group(1).strip().replace(' ', '')
#     else:
#         print("Aadhaar number not found in the income certificate text.")

#     if dob_match:
#         details['Date_of_birth'] = dob_match.group(1).strip()
#     else:
#         print("Date of birth not found in the income certificate text.")

#     return details

# def extract_application_form_details(text: str) -> Dict:
#     details = {}
#     print(f"Extracting application form details from text: {text[:100]}...")  # Debugging statement

#     # Regex patterns to extract the relevant information
#     applicant_name_pattern = re.compile(r'Applicant\s*Name\s*:\s*([A-Za-z\s]+)', re.IGNORECASE)
#     father_husband_name_pattern = re.compile(r'Father\/Husband\s*Name\s*:\s*([A-Za-z\s]+)', re.IGNORECASE)
#     dob_pattern = re.compile(r'Date\s*of\s*Birth\s*:\s*(\d{2}/\d{2}/\d{4})', re.IGNORECASE)
#     aadhaar_pattern = re.compile(r'Aadhar\s*Card\s*No\s*:\s*([\d\s]+)', re.IGNORECASE)
#     mobile_pattern = re.compile(r'Mobile\s*No\s*:\s*([\d\s]+)', re.IGNORECASE)
#     ration_card_pattern = re.compile(r'Ration\s*Card\s*No\s*:\s*([A-Za-z\d\s]+)', re.IGNORECASE)

#     applicant_name_match = applicant_name_pattern.search(text)
#     father_husband_name_match = father_husband_name_pattern.search(text)
#     dob_match = dob_pattern.search(text)
#     aadhaar_match = aadhaar_pattern.search(text)
#     mobile_match = mobile_pattern.search(text)
#     ration_card_match = ration_card_pattern.search(text)

#     if applicant_name_match:
#         details['Applicant_Name'] = applicant_name_match.group(1).strip()
#     else:
#         print("Applicant name not found in the application form text.")

#     if father_husband_name_match:
#         details['Father_Husband_Name'] = father_husband_name_match.group(1).strip()
#     else:
#         print("Father/Husband name not found in the application form text.")

#     if dob_match:
#         details['Date_of_birth'] = dob_match.group(1).strip()
#     else:
#         print("Date of birth not found in the application form text.")

#     if aadhaar_match:
#         details['Aadhaar_Number'] = aadhaar_match.group(1).strip().replace(' ', '')
#     else:
#         print("Aadhaar number not found in the application form text.")

#     if mobile_match:
#         details['Mobile_number'] = mobile_match.group(1).strip().replace(' ', '')
#     else:
#         print("Mobile number not found in the application form text.")

#     if ration_card_match:
#         details['Ration_card'] = ration_card_match.group(1).strip()
#     else:
#         print("Ration card number not found in the application form text.")

#     return details

# @app.post("/process-income-cert/")
# async def process_income(application_form: UploadFile = File(...), aadhaar: UploadFile = File(...)):
#     application_path = f"/tmp/{application_form.filename}"
#     aadhaar_path = f"/tmp/{aadhaar.filename}"

#     with open(application_path, "wb") as f:
#         f.write(await application_form.read())
    
#     with open(aadhaar_path, "wb") as f:
#         f.write(await aadhaar.read())

#     application_document = process_document(processor_name, file_path=application_path)
#     aadhaar_document = process_document(processor_name, file_path=aadhaar_path)

#     if application_document:
#         application_text = application_document.text
#         print(f"Extracted application document text: {application_text[:500]}...")  # Debugging statement
#         application_data = extract_application_form_details(application_text)
#         os.remove(application_path)
#     else:
#         return {"error": "Application data couldn't be parsed"}
    
#     if aadhaar_document:
#         aadhaar_text = aadhaar_document.text
#         print(f"Extracted Aadhaar document text: {aadhaar_text[:500]}...")  # Debugging statement
#         aadhaar_data = extract_income_certificate_details(aadhaar_text)
#         os.remove(aadhaar_path)
#     else:
#         return {"error": "Issue with the Aadhaar card. Please try again"}
    
#     return {"application_data": application_data, "aadhaar_data": aadhaar_data}



# from fastapi import FastAPI, UploadFile, File
# from google.cloud import documentai_v1beta3 as documentai
# import os
# from typing import Dict, List

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./creds/grant01-joby.json"

# app = FastAPI()
# processor_name = "projects/332125695616/locations/us/processors/a6bceed480e9d614"

# def process_document(processor_name: str, file_path: str) -> documentai.Document:
#     client = documentai.DocumentProcessorServiceClient()
    
#     # Read the file into memory
#     with open(file_path, "rb") as f:
#         document_content = f.read()

#     # Configure the request
#     request = documentai.ProcessRequest(
#         name=processor_name,
#         raw_document=documentai.RawDocument(
#             content=document_content,
#             mime_type="application/pdf"
#         )
#     )

#     result = client.process_document(request=request)

#     return result.document

# def extract_text_blocks(document: documentai.Document) -> str:
#     """Extract all text from the document's blocks."""
#     text_blocks = []
#     for block in document.blocks:
#         text = block.layout.text_anchor.text_segments[0].text
#         text_blocks.append(text)
#     return "\n".join(text_blocks)

# def extract_info_from_text(text: str, fields: Dict[str, str]) -> Dict:
#     """Extract specific fields from text using regex."""
#     import re

#     details = {}

#     # Regex patterns
#     for field_name, pattern in fields.items():
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             details[field_name] = match.group(1).strip()

#     return details

# @app.post("/process-income-cert/")
# async def process_income(application_form: UploadFile = File(...), aadhaar: UploadFile = File(...)):
#     application_path = f"/tmp/{application_form.filename}"
#     aadhaar_path = f"/tmp/{aadhaar.filename}"

#     with open(application_path, "wb") as f:
#         f.write(await application_form.read())
    
#     with open(aadhaar_path, "wb") as f:
#         f.write(await aadhaar.read())

#     application_document = process_document(processor_name, file_path=application_path)
#     aadhaar_document = process_document(processor_name, file_path=aadhaar_path)

#     if application_document:
#         # Extract text from the application document
#         application_text = extract_text_blocks(application_document)
#         # Define the fields and regex patterns for the application form
#         application_fields = {
#             "Applicant_Name": r'Applicant\s*Name\s*:\s*([A-Za-z\s]+)',
#             "Father_Husband_Name": r'Father/Husband\s*Name\s*:\s*([A-Za-z\s]+)',
#             "Date_of_birth": r'Date Of Birth\s*:\s*(\d{2}/\d{2}/\d{4})',
#             "Aadhaar_Number": r'Aadhar Card No\s*:\s*([\d\s]+)',
#             "Mobile_number": r'Mobile No\s*:\s*([\d\s]+)',
#             "Ration_card": r'Ration Card\s*:\s*([A-Za-z\d\s]+)'
#         }
#         application_data = extract_info_from_text(application_text, application_fields)
#         os.remove(application_path)
#     else:
#         return {"error": "Application data couldn't be parsed"}
    
#     if aadhaar_document:
#         # Extract text from the Aadhaar document
#         aadhaar_text = extract_text_blocks(aadhaar_document)
#         # Define the fields and regex patterns for the Aadhaar card
#         aadhaar_fields = {
#             "Name": r'Name\s*:\s*([A-Za-z\s]+)',
#             "Aadhaar_number": r'Aadhaar\s*Number\s*:\s*([\d\s]+)',
#             "Date_of_birth": r'Date of Birth\s*:\s*(\d{2}/\d{2}/\d{4})'
#         }
#         aadhaar_data = extract_info_from_text(aadhaar_text, aadhaar_fields)
#         os.remove(aadhaar_path)
#     else:
#         return {"error": "Issue with the Aadhaar card. Please try again"}
    
#     return {"application_data": application_data, "aadhaar_data": aadhaar_data}

