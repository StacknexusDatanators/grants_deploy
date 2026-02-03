import ollama

# Read the modelfile
with open('ollama_modfile', 'r') as f:
    modelfile_content = f.read()

print("Modelfile content:")
print(modelfile_content)
print("\n" + "="*50 + "\n")

try:
    # First, check if the base model exists
    print("Checking for base model llama3.1...")
    try:
        ollama.show('llama3.1')
        print("Base model llama3.1 found!")
    except:
        print("Base model llama3.1 not found. Pulling it first...")
        ollama.pull('llama3.1')
        print("Base model pulled successfully!")
    
    # Create the custom model with the modelfile content
    print("Creating custom model llama31_datanator...")
    
    # Use the correct API - pass modelfile as path or use stream parameter
    response = ollama.create(
        model='llama31_datanator',
        path='ollama_modfile'  # Pass the file path instead
    )
    
    print(f"Model created successfully!")
    print(f"Response: {response}")
    
except Exception as e:
    print(f"Error creating model: {e}")
    print("\nTrying alternative method...")
    
    try:
        # Alternative: use modelfile content directly with stream=False
        for chunk in ollama.create(model='llama31_datanator', modelfile=modelfile_content, stream=True):
            print(chunk.get('status', ''), end='', flush=True)
        print("\nModel created successfully using streaming method!")
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")
        print("\nPlease try creating the model manually:")
        print("1. Run: ollama create llama31_datanator -f ollama_modfile")

# Uncomment below if you want to push to Ollama registry
# ollama.push(model='llama31_datanator')