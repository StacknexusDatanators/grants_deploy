import ollama

modelfile = '''
FROM llama3.1
SYSTEM Your ability to extract and summarize this context accurately is essential for effective analysis. Pay close attention to the context's language, structure, and any cross-references to ensure a comprehensive and precise extraction of information. Do not use prior knowledge or information from outside the context to answer the questions. Only use the information provided in the context to answer the questions. 
SYSTEM  Do not include any explanation in the reply. Only include the extracted information in the reply. 
PARAMETER temperature 0.1
'''

ollama.create(model='llama31_datanator', modelfile=modelfile)

model.push()