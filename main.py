from fastapi import FastAPI, UploadFile, File
from google.cloud import documentai_v1beta3 as documentai
import os
from typing import List, Dict
import ollama
from ast import literal_eval
import time 
import fitz  # PyMuPDF
import io
from PIL import Image
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from textblob import TextBlob
from langdetect import detect
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./creds/grant01-joby.json"
import json

with open('prompt_template.json', 'r') as file:
    prompts_superset = json.load(file)


app = FastAPI()
model_sel = "vicuna:7b"
# If you already have a Document AI Processor in your project, assign the full processor resource name here.
processor_name = "projects/332125695616/locations/us/processors/a6bceed480e9d614"


def detect_orientation_pdf(pdf_path):
    # Open the PDF file
    pdf = fitz.open(pdf_path)


    # Get the page
    page = pdf[0]

    # Convert the PDF page to an image
    pix = page.get_pixmap()
    img_data = pix.tobytes("ppm")
    img = Image.open(io.BytesIO(img_data))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, th = cv2.threshold(gray,
        127,  # threshold value
        255,  # maximum value assigned to pixel values exceeding the threshold
        cv2.THRESH_BINARY)   # constant

    print("pdf", pdf_path)
    custom_config = r'--dpi 300 --psm 0 -c min_characters_to_try=5'

    # Use Tesseract to detect orientation
    osd = pytesseract.image_to_osd(th, config=custom_config, output_type=Output.DICT)
    angle = osd['rotate']
    script = osd['script']

    # Check if the image is upside down
    if angle != 0:
        #print(f"Page {page_num + 1} is upside down. Detected script: {script}")
        # Rotate the image to correct it
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        # Convert the corrected image back to a PIL image and save
        corrected_img = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        corrected_img.save(pdf_path, "PDF", resolution=100.0)
    print("done")


def process_document(processor_name: str, file_path: str) -> documentai.Document:
    client = documentai.DocumentProcessorServiceClient()

    detect_orientation_pdf(file_path)


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

#@app.post("/process-pdf/")
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

def parse_docs(extracted_txt: str, doc_parent: str, doc_child: str) -> dict:
    lang = detect(extracted_txt)
    if lang == 'te':
        extracted_txt = transliterate(extracted_txt, sanscript.TELUGU, sanscript.HK)
    prompt_template = prompts_superset[doc_parent][doc_child]
    response = ollama.chat(model=model_sel, messages = [
        {
            'role':'user',
            'content': prompt_template+extracted_txt
      }
    ], format = "json")

    output = response["message"]["content"]
    final_out = literal_eval(output)
    return final_out

@app.post("/process-income-cert/")
async def process_income(application_form: UploadFile = File(...), aadhaar: UploadFile = File(...)):
    application_path = f"/tmp/{application_form.filename}"
    aadhaar_path = f"/tmp/{aadhaar.filename}"

    with open(application_path, "wb") as f:
        f.write(await application_form.read())
    
    with open(aadhaar_path, "wb") as f:
        f.write(await aadhaar.read())

    application_document = process_document(processor_name, file_path=application_path)
    aadhaar_document = process_document(processor_name, file_path = aadhaar_path)

    if application_document:
        application_data = parse_docs(application_document.text, "income_certificate", "application_form")
        os.remove(application_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Application data couldn't be parsed")
    
    if aadhaar_document:
        aadhaar_data = parse_docs(aadhaar_document.text, "income_certificate", "aadhaar_card")
        os.remove(aadhaar_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Issue with the Aadhaar card. Please try again")
    
    return {"application_data": application_data, "aadhaar_data": aadhaar_data}


@app.post("/process-community-dob-certificate/")
async def process_community_dob(study_certificate: UploadFile = File(...), application_form: UploadFile = File(...), aadhaar_card: UploadFile = File(...)):
    study_certificate_path = f"/tmp/{study_certificate.filename}"
    application_path = f"/tmp/{application_form.filename}"
    aadhaar_path = f"/tmp/{aadhaar_card.filename}"

    with open(study_certificate_path, "wb") as f:
        f.write(await study_certificate.read())

    with open(application_path, "wb") as f:
        f.write(await application_form.read())

    with open(aadhaar_path, "wb") as f:
        f.write(await aadhaar_card.read())

    study_certificate_document = process_document(processor_name, file_path=study_certificate_path)
    application_document = process_document(processor_name, file_path=application_path)
    aadhaar_document = process_document(processor_name, file_path=aadhaar_path)

    if study_certificate_document:
        study_certificate_data = parse_docs(study_certificate_document.text, "community_dob_certificate", "study_certificate")
        os.remove(study_certificate_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Study Certificate data couldn't be parsed")

    if application_document:
        application_data = parse_docs(application_document.text, "community_dob_certificate", "application_form")
        os.remove(application_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Application data couldn't be parsed")

    if aadhaar_document:
        aadhaar_data = parse_docs(aadhaar_document.text, "community_dob_certificate", "aadhaar_card")
        os.remove(aadhaar_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Issue with the Aadhaar card. Please try again")

    return {
        "study_certificate_data": study_certificate_data,
        "application_data": application_data,
        "aadhaar_data": aadhaar_data
  }

@app.post("/process-ebc-certificate/")
async def process_ebc(application_form: UploadFile = File(...), aadhaar_card: UploadFile = File(...)):
    application_path = f"/tmp/{application_form.filename}"
    aadhaar_path = f"/tmp/{aadhaar_card.filename}"

    with open(application_path, "wb") as f:
        f.write(await application_form.read())

    with open(aadhaar_path, "wb") as f:
        f.write(await aadhaar_card.read())

    application_document = process_document(processor_name, file_path=application_path)
    aadhaar_document = process_document(processor_name, file_path=aadhaar_path)

    if application_document:
        application_data = parse_docs(application_document.text, "ebc_certificate", "application_form")
        os.remove(application_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Application data couldn't be parsed")

    if aadhaar_document:
        aadhaar_data = parse_docs(aadhaar_document.text, "ebc_certificate", "aadhaar_card")
        os.remove(aadhaar_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Issue with the Aadhaar card. Please try again")

    return {
        "application_data": application_data,
        "aadhaar_data": aadhaar_data
  }

@app.post("/process-ewc-certificate/")
async def process_ewc(application_form: UploadFile = File(...), aadhaar_card: UploadFile = File(...)):
    application_path = f"/tmp/{application_form.filename}"
    aadhaar_path = f"/tmp/{aadhaar_card.filename}"

    with open(application_path, "wb") as f:
        f.write(await application_form.read())

    with open(aadhaar_path, "wb") as f:
        f.write(await aadhaar_card.read())

    application_document = process_document(processor_name, file_path=application_path)
    aadhaar_document = process_document(processor_name, file_path=aadhaar_path)

    if application_document:
        application_data = parse_docs(application_document.text, "economically_weaker_section", "application_form")
        os.remove(application_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Application data couldn't be parsed")

    if aadhaar_document:
        aadhaar_data = parse_docs(aadhaar_document.text, "economically_weaker_section", "aadhaar_card")
        os.remove(aadhaar_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Issue with the Aadhaar card. Please try again")


    return {
        "application_data": application_data,
        "aadhaar_data": aadhaar_data
  }
@app.post("/process-obc-certificate/")
async def process_obc(
    application_form: UploadFile = File(...),
    aadhaar_card: UploadFile = File(...),
    income_tax_return: UploadFile = File(...),
    property_particulars: UploadFile = File(...)
):
    application_path = f"/tmp/{application_form.filename}"
    aadhaar_path = f"/tmp/{aadhaar_card.filename}"
    income_tax_path = f"/tmp/{income_tax_return.filename}"
    property_path = f"/tmp/{property_particulars.filename}"

    with open(application_path, "wb") as f:
        f.write(await application_form.read())

    with open(aadhaar_path, "wb") as f:
        f.write(await aadhaar_card.read())

    with open(income_tax_path, "wb") as f:
        f.write(await income_tax_return.read())

    with open(property_path, "wb") as f:
        f.write(await property_particulars.read())

    application_document = process_document(processor_name, file_path=application_path)
    aadhaar_document = process_document(processor_name, file_path=aadhaar_path)
    income_tax_document = process_document(processor_name, file_path=income_tax_path)
    property_document = process_document(processor_name, file_path=property_path)

    if application_document:
        application_data = parse_docs(application_document.text, "obc_certificate", "application_form")
        os.remove(application_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Application data couldn't be parsed")

    if aadhaar_document:
        aadhaar_data = parse_docs(aadhaar_document.text, "obc_certificate", "aadhaar_card")
        os.remove(aadhaar_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Issue with the Aadhaar card. Please try again")

    if income_tax_document:
        income_tax_data = parse_docs(income_tax_document.text, "obc_certificate", "income_tax_return")
        os.remove(income_tax_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Issue with the Income Tax Return document. Please try again")

    if property_document:
        property_data = parse_docs(property_document.text, "obc_certificate", "property_particulars")
        os.remove(property_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Issue with the Property Particulars document. Please try again")

    return {
        "application_data": application_data,
        "aadhaar_data": aadhaar_data,
        "income_tax_data": income_tax_data,
        "property_data": property_data
  }
@app.post("/process-residence-certificate/")
async def process_residence_certificate(
    application_form: UploadFile = File(...),
    aadhaar_card: UploadFile = File(...)
):
    application_path = f"/tmp/{application_form.filename}"
    aadhaar_path = f"/tmp/{aadhaar_card.filename}"

    with open(application_path, "wb") as f:
        f.write(await application_form.read())

    with open(aadhaar_path, "wb") as f:
        f.write(await aadhaar_card.read())

    application_document = process_document(processor_name, file_path=application_path)
    aadhaar_document = process_document(processor_name, file_path=aadhaar_path)

    if application_document:
        application_data = parse_docs(application_document.text, "residence_certificate", "application_form")
        os.remove(application_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Application data couldn't be parsed")

    if aadhaar_document:
        aadhaar_data = parse_docs(aadhaar_document.text, "residence_certificate", "aadhaar_card")
        os.remove(aadhaar_path)
    else:
        raise HTTPException(status_code=422, detail="Unrecognized entity: Issue with the Aadhaar card. Please try again")

    return {
        "application_data": application_data,
        "aadhaar_data": aadhaar_data
  }
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
