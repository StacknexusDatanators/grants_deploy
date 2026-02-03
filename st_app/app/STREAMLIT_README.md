# PDF Document Similarity Analyzer - Streamlit App

This Streamlit application allows you to upload multiple PDF documents, extract fields using the FastAPI backend, and calculate Jaccard similarity scores between all pairs of documents for each extracted field.

## Features

- 📁 **Multiple Document Upload**: Upload multiple sets of PDF documents
- 🔍 **Automated Field Extraction**: Uses the existing FastAPI endpoints to extract fields
- 📊 **Pairwise Comparison**: Calculates Jaccard similarity between all document pairs
- 📈 **Visual Analytics**: 
  - Interactive heatmaps for each field
  - Distribution charts
  - Detailed comparison tables
- 📥 **Export Results**: Download similarity analysis as CSV

## Prerequisites

1. **FastAPI Server Running**: The backend API must be running on `http://localhost:8000`
2. **Python Dependencies**: Install required packages

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Or install Streamlit and Plotly separately:
```bash
pip install streamlit plotly
```

## Usage

### Step 1: Start the FastAPI Server

First, ensure your FastAPI server is running:

```bash
python main.py
```

The API should be accessible at `http://localhost:8000`

### Step 2: Launch the Streamlit App

In a new terminal, run:

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Step 3: Use the Application

1. **Select Certificate Type**: Choose the type of certificate you want to process from the sidebar
2. **Set Number of Documents**: Enter how many document sets you want to compare (minimum 2)
3. **Upload Documents**: For each document set:
   - Provide a name (e.g., "Application_1", "Application_2")
   - Upload all required PDF files
4. **Process**: Click "Process and Compare Documents"
5. **View Results**: Explore the three tabs:
   - **Detailed Results**: Field-by-field comparison with color-coded similarities
   - **Visualizations**: Heatmaps and distribution charts
   - **Raw Data**: Extracted JSON data and download option

## Supported Certificate Types

- Income Certificate
- Community DOB Certificate
- EBC Certificate
- EWS Certificate
- OBC Certificate
- Residence Certificate

## How Jaccard Similarity Works

The Jaccard similarity coefficient measures similarity between two sets. For text fields:

1. Convert each string to a set of characters
2. Calculate intersection and union of these sets
3. Similarity = |intersection| / |union|

**Similarity Interpretation:**
- 🟢 **0.8 - 1.0**: High similarity (green)
- 🟡 **0.5 - 0.79**: Medium similarity (yellow)
- 🔴 **0.0 - 0.49**: Low similarity (red)

## Example Use Case

### Comparing Multiple Applications

If you have 3 applications for income certificates, you can:
1. Upload all 3 sets of documents (application form + aadhaar for each)
2. The app will extract fields like:
   - `applicant_name`
   - `father_husband_name`
   - `date_of_birth`
   - `aadhar_number`
   - etc.
3. For each field, it calculates similarity between:
   - Document 1 vs Document 2
   - Document 1 vs Document 3
   - Document 2 vs Document 3
4. Results show which applications have similar information

### Sample Output

```
Document 1: Application_A
Document 2: Application_B
Field: applicant_name
Value 1: "John Doe"
Value 2: "John D"
Jaccard Similarity: 0.75
```

## Configuration

To change the API URL, modify the `API_BASE_URL` in `streamlit_app.py`:

```python
API_BASE_URL = "http://localhost:8000"  # Change this if needed
```

## Troubleshooting

### Connection Error
- **Issue**: Cannot connect to API
- **Solution**: Ensure FastAPI server is running on port 8000

### Upload Error
- **Issue**: File upload fails
- **Solution**: Check file is a valid PDF and under 200MB

### No Results
- **Issue**: No similarity results shown
- **Solution**: Ensure at least 2 documents processed successfully with common fields

## Technical Details

### API Integration
The app makes POST requests to these endpoints:
- `/process-income-cert/`
- `/process-community-dob-certificate/`
- `/process-ebc-certificate/`
- `/process-ews-certificate/`
- `/process-obc-certificate/`
- `/process-residence-certificate/`

### Similarity Calculation
- Character-level Jaccard similarity
- Case-insensitive comparison
- Handles missing fields gracefully

### Performance
- Processes documents sequentially
- Shows progress bar during processing
- Caches results in session state

## Future Enhancements

Potential improvements:
- [ ] Batch processing with parallel API calls
- [ ] Custom similarity thresholds
- [ ] Additional similarity metrics (Levenshtein, Cosine)
- [ ] PDF preview before processing
- [ ] Field-level validation rules
- [ ] Historical comparison tracking

## Support

For issues or questions, refer to the main project documentation or contact the development team.
