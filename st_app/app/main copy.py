from fastapi import FastAPI, UploadFile, File
from google.cloud import documentai_v1beta3 as documentai
import os
from typing import List, Dict
import ollama
from ast import literal_eval
import time 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/astrobalaji/Documents/stacknexus/grants/notebook/creds/grant01-joby.json"



app = FastAPI()
model_sel = "vicuna:7b"
# If you already have a Document AI Processor in your project, assign the full processor resource name here.
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

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    document = process_document(processor_name, file_path=file_path)

    if document:
        extracted_data: List[Dict] = []
        
        # Extract text from the document
        document_text = document.text

        # Split the text into chunks based on paragraphs
        document_chunks = document_text.split('\n\n')  # Assuming paragraphs are separated by double newlines

        for chunk_number, chunk_content in enumerate(document_chunks, start=1):
            extracted_data.append(
                {
                    "file_name": file.filename,
                    "file_type": os.path.splitext(file.filename)[1],
                    "chunk_number": chunk_number,
                    "content": chunk_content,
                }
            )

        combined_list = [t["content"] for t in extracted_data]
        combined_str = "\n".join(combined_list)

        # Remove the temporary file
        os.remove(file_path)

        return {"text": combined_str}
    else:
        return {"error": "Failed to process the document"}
    
def parse_aadhaar_info(extracted_text: str) -> dict:
    response = ollama.chat(model=model_sel, messages=[
        {
            'role': 'user',
            'content': f"""[Requirement] for the following content parsed from a scanned Aadhaar card document. The Aadhaar number is a 12 digit number with spaces in between. I want you to give me the following data in the following json structure. 
            [json_structure] {{"Name":---, "Aadhaar_number":---, "Date_of_birth":---}}
            ["content"]{extracted_text}
            """,
        },
    ], format="json")
    
    # Safely evaluate the response content to convert it to a dictionary
    print("here1")
    print(response["message"]["content"])
    print("here2")
    output = response["message"]["content"]
    aadhaar_info = literal_eval(output)


    return aadhaar_info

@app.post("/process-aadhaar/")
async def process_aadhaar(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    start_time = time.time()
    # Your code block
 

    document = process_document(processor_name, file_path=file_path)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")


    if document:
        extracted_text = document.text

        # Parse Aadhaar information
        aadhaar_info = parse_aadhaar_info(extracted_text)

        # Remove the temporary file
        os.remove(file_path)

        return aadhaar_info
    else:
        return {"error": "Failed to process the document"}

def parse_income_cert(extracted_text:str):
    prompt_template = """
        [Requirement] for the content that follows, which was extracted from an application form that was scanned. In addition to the applicant's name, which is a character with spaces between it, the date of birth is a variable character,  the mobile number with 10 digit number with spaces between it, the Adhaar number is a 12-digit number with spaces between it, and the ration card number is also a character. Please provide me with the following information in the JSON structure. 
        [json_structure] {{"Applicant Name":---, "Father_Husband_Name":---, "Date_of_birth":---  "Adhaar_Number":---  "Mobile_number":---  "Ration_card:---}}
        [content] {0}
    """
    response = ollama.chat(model=model_sel, messages=[
        {
            'role': 'user',
            'content': prompt_template.format(extracted_text),
        },
    ], format="json")
    # Safely evaluate the response content to convert it to a dictionary
    try:
        inc_info = literal_eval(response['message']['content'])
    except (SyntaxError, ValueError) as e:
        inc_info = {"error": "Failed to parse income information"}

    return inc_info 

@app.post("/process-income-cert/")
async def process_inc(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    document = process_document(processor_name, file_path=file_path)

    if document:
        extracted_text = document.text

        # Parse Aadhaar information
        inc_info = parse_income_cert(extracted_text)

        # Remove the temporary file
        os.remove(file_path)

        return inc_info
    else:
        return {"error": "Failed to process the document"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
