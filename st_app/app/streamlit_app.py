import streamlit as st
import requests
from itertools import combinations
import pandas as pd
import json
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Certificate type configuration
CERTIFICATE_TYPES = {
    "Income Certificate": {
        "endpoint": "/process-income-cert/",
        "files": ["application_form", "aadhaar"]
    },
    "Community DOB Certificate": {
        "endpoint": "/process-community-dob-certificate/",
        "files": ["application_form", "aadhaar_card", "study_certificate"]
    },
    "EBC Certificate": {
        "endpoint": "/process-ebc-certificate/",
        "files": ["application_form", "aadhaar_card"]
    },
    "EWS Certificate": {
        "endpoint": "/process-ews-certificate/",
        "files": ["application_form", "aadhaar_card"]
    },
    "OBC Certificate": {
        "endpoint": "/process-obc-certificate/",
        "files": ["application_form", "aadhaar_card", "income_tax_return", "property_particulars"]
    },
    "Residence Certificate": {
        "endpoint": "/process-residence-certificate/",
        "files": ["application_form", "aadhaar_card"]
    }
}


def jaccard_similarity(str1: str, str2: str) -> float:
    """Calculate Jaccard similarity between two strings."""
    if not str1 or not str2:
        return 0.0
    
    set1 = set(str1.lower())
    set2 = set(str2.lower())
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def process_single_pdf_set(files: Dict, endpoint: str) -> Dict:
    """Process a single set of PDF files through the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            files=files
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def extract_all_fields(data: Dict) -> Dict[str, str]:
    """Extract all field values from API response."""
    fields = {}
    
    for key, value in data.items():
        if isinstance(value, dict):
            for field_name, field_value in value.items():
                # Create a unique key combining the source and field name
                unique_key = f"{key}_{field_name}"
                fields[unique_key] = str(field_value)
        elif key != "name_score":  # Skip similarity scores from API
            fields[key] = str(value)
    
    return fields


def calculate_pairwise_similarities(documents_data: List[Dict], doc_names: List[str]) -> pd.DataFrame:
    """Calculate Jaccard similarity for each field across all document pairs."""
    
    # Extract fields from all documents
    all_fields = []
    for doc_data in documents_data:
        all_fields.append(extract_all_fields(doc_data))
    
    # Get all unique field names
    all_field_names = set()
    for fields in all_fields:
        all_field_names.update(fields.keys())
    
    # Calculate similarities for each field
    results = []
    
    # Compare all pairs of documents
    for (i, doc1), (j, doc2) in combinations(enumerate(all_fields), 2):
        doc1_name = doc_names[i]
        doc2_name = doc_names[j]
        
        for field_name in all_field_names:
            val1 = doc1.get(field_name, "")
            val2 = doc2.get(field_name, "")
            
            if val1 and val2:  # Only calculate if both documents have this field
                similarity = jaccard_similarity(val1, val2)
                results.append({
                    "Document 1": doc1_name,
                    "Document 2": doc2_name,
                    "Field": field_name,
                    "Value 1": val1,
                    "Value 2": val2,
                    "Jaccard Similarity": round(similarity, 4)
                })
    
    return pd.DataFrame(results)


def create_similarity_heatmap(df: pd.DataFrame, field_name: str):
    """Create a heatmap for a specific field's similarities."""
    # Filter data for the specific field
    field_data = df[df["Field"] == field_name]
    
    if field_data.empty:
        return None
    
    # Get unique document names
    docs = sorted(set(field_data["Document 1"].tolist() + field_data["Document 2"].tolist()))
    
    # Create similarity matrix
    matrix = [[1.0 if i == j else 0.0 for j in range(len(docs))] for i in range(len(docs))]
    
    for _, row in field_data.iterrows():
        i = docs.index(row["Document 1"])
        j = docs.index(row["Document 2"])
        sim = row["Jaccard Similarity"]
        matrix[i][j] = sim
        matrix[j][i] = sim
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=docs,
        y=docs,
        colorscale='RdYlGn',
        text=[[f"{val:.3f}" for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Similarity")
    ))
    
    fig.update_layout(
        title=f"Jaccard Similarity Heatmap - {field_name}",
        xaxis_title="Documents",
        yaxis_title="Documents",
        height=500
    )
    
    return fig


def main():
    st.set_page_config(page_title="PDF Document Similarity Analyzer", layout="wide")
    
    st.title("📄 PDF Document Similarity Analyzer")
    st.markdown("Upload multiple PDF documents to extract fields and calculate Jaccard similarity scores between them.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        certificate_type = st.selectbox(
            "Select Certificate Type",
            options=list(CERTIFICATE_TYPES.keys())
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This app processes PDF documents through the API and calculates 
        Jaccard similarity scores for each extracted field across all document pairs.
        """)
    
    # Main content
    st.header(f"Process {certificate_type}")
    
    required_files = CERTIFICATE_TYPES[certificate_type]["files"]
    endpoint = CERTIFICATE_TYPES[certificate_type]["endpoint"]
    
    # ========================================
    # SECTION 1: UPLOAD ALL DOCUMENTS
    # ========================================
    st.subheader("📁 Upload Documents")
    st.markdown("Upload all PDF files for comparison. Each file should be a complete document set.")
    
    # Create columns for different document types
    st.markdown("#### Required Documents for Each Application:")
    file_info = st.info(f"**Required files:** {', '.join([f.replace('_', ' ').title() for f in required_files])}")
    
    # Upload multiple complete document sets
    uploaded_files = st.file_uploader(
        "Upload PDF Documents (one complete set per application)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload all your PDF documents here. Each document should contain all required forms/cards."
    )
    
    st.markdown("---")
    
    # Process button
    if st.button("🔍 Process and Compare Documents", type="primary") and uploaded_files:
        # Validate minimum files
        if len(uploaded_files) < 2:
            st.error("Please upload at least 2 PDF documents for comparison.")
        else:
            # For income certificate, we need pairs of files (application + aadhaar)
            # Assuming files are uploaded in pairs or we process each file independently
            
            st.info(f"Processing {len(uploaded_files)} documents...")
            
            # ========================================
            # SECTION 2: PROCESS EACH DOCUMENT
            # ========================================
            st.subheader("⚙️ Processing Documents")
            
            documents_data = []
            doc_names = []
            
            progress_bar = st.progress(0)
            result_containers = []
            
            for i in range(len(uploaded_files)):
                result_containers.append(st.empty())
            
            # Process each uploaded file
            for i, uploaded_file in enumerate(uploaded_files):
                doc_name = uploaded_file.name.replace('.pdf', '')
                doc_names.append(doc_name)
                
                # For simplicity, treating each PDF as a single document
                # You may need to adjust based on your specific requirements
                api_files = {
                    required_files[0]: (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        'application/pdf'
                    )
                }
                
                # Add dummy files for required fields if needed
                # This is a simplified version - adjust based on your API requirements
                if len(required_files) > 1:
                    for file_type in required_files[1:]:
                        # Skip optional files
                        if file_type in ["study_certificate", "income_tax_return", "property_particulars"]:
                            continue
                        # For now, using the same file for all required types
                        # You may want to enhance this logic
                        api_files[file_type] = (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            'application/pdf'
                        )
                
                try:
                    result = process_single_pdf_set(api_files, endpoint)
                    
                    if result:
                        documents_data.append(result)
                        result_containers[i].success(f"✅ Processed {doc_name}")
                    else:
                        result_containers[i].error(f"❌ Failed to process {doc_name}")
                except Exception as e:
                    result_containers[i].error(f"❌ Error processing {doc_name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.markdown("---")
            
            # ========================================
            # SECTION 3: INDIVIDUAL DOCUMENT RESULTS
            # ========================================
            if len(documents_data) >= 1:
                st.subheader("📋 Extracted Data from Each Document")
                
                for i, (doc_data, doc_name) in enumerate(zip(documents_data, doc_names)):
                    with st.expander(f"📄 {doc_name} - Extracted Fields", expanded=False):
                        # Display extracted data in a formatted way
                        for key, value in doc_data.items():
                            if isinstance(value, dict):
                                st.markdown(f"**{key.replace('_', ' ').title()}:**")
                                df = pd.DataFrame([value]).T
                                df.columns = ['Value']
                                st.dataframe(df, use_container_width=True)
                            elif key != "name_score":
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        # Show raw JSON
                        with st.expander("View Raw JSON"):
                            st.json(doc_data)
            
            st.markdown("---")
            
            # ========================================
            # SECTION 4: COMPARISON & SIMILARITY ANALYSIS
            # ========================================
            # Calculate similarities
            if len(documents_data) >= 2:
                st.subheader("📊 Similarity Analysis - All Comparisons")
                
                similarity_df = calculate_pairwise_similarities(documents_data, doc_names)
                
                if not similarity_df.empty:
                    # Display overall statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Comparisons", len(similarity_df))
                    with col2:
                        avg_similarity = similarity_df["Jaccard Similarity"].mean()
                        st.metric("Average Similarity", f"{avg_similarity:.3f}")
                    with col3:
                        unique_fields = similarity_df["Field"].nunique()
                        st.metric("Fields Compared", unique_fields)
                    
                    # Tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📋 Detailed Results", "📈 Visualizations", "💾 Download Data"])
                    
                    with tab1:
                        st.markdown("### Field-by-Field Comparison")
                        
                        # Filter by field
                        selected_field = st.selectbox(
                            "Select Field to View",
                            options=["All Fields"] + sorted(similarity_df["Field"].unique().tolist())
                        )
                        
                        if selected_field == "All Fields":
                            display_df = similarity_df
                        else:
                            display_df = similarity_df[similarity_df["Field"] == selected_field]
                        
                        # Color code based on similarity
                        def highlight_similarity(val):
                            if isinstance(val, (int, float)):
                                if val >= 0.8:
                                    return 'background-color: #90EE90'
                                elif val >= 0.5:
                                    return 'background-color: #FFD700'
                                else:
                                    return 'background-color: #FFB6C6'
                            return ''
                        
                        styled_df = display_df.style.applymap(
                            highlight_similarity,
                            subset=['Jaccard Similarity']
                        )
                        
                        st.dataframe(styled_df, use_container_width=True)
                    
                    with tab2:
                        st.markdown("### Similarity Heatmaps")
                        
                        # Create heatmap for each field
                        fields = sorted(similarity_df["Field"].unique())
                        
                        for field in fields:
                            fig = create_similarity_heatmap(similarity_df, field)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Overall similarity distribution
                        st.markdown("### Similarity Score Distribution")
                        fig_hist = px.histogram(
                            similarity_df,
                            x="Jaccard Similarity",
                            nbins=20,
                            title="Distribution of Jaccard Similarity Scores",
                            labels={"Jaccard Similarity": "Similarity Score"}
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with tab3:
                        st.markdown("### Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv = similarity_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Similarity Results (CSV)",
                                data=csv,
                                file_name="similarity_results.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Export all extracted data as JSON
                            all_data = {
                                doc_names[i]: documents_data[i] 
                                for i in range(len(documents_data))
                            }
                            json_data = json.dumps(all_data, indent=2)
                            st.download_button(
                                label="📥 Download Extracted Data (JSON)",
                                data=json_data,
                                file_name="extracted_data.json",
                                mime="application/json"
                            )
                else:
                    st.warning("No common fields found across documents for comparison.")
            else:
                st.warning("Need at least 2 successfully processed documents for comparison.")


if __name__ == "__main__":
    main()
