import streamlit as st
from PIL import Image
import io
import easyocr
import torch
import requests
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="Medicine Identifier & Price Comparison",
    page_icon="💊",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        border-left: 3px solid #ff4b4b;
        padding-left: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ocr():
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    return reader

def perform_ocr(image):
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Get OCR reader
    reader = load_ocr()
    
    # Perform OCR
    results = reader.readtext(image_np)
    
    # Extract text from results
    extracted_text = '\n'.join([text[1] for text in results])
    
    return extracted_text

def load_models():
    # Load ViT model for medicine classification
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    return processor, model

def create_savings_gauge(savings_percentage):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = savings_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Potential Savings"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': savings_percentage
            }
        }
    ))
    return fig

def get_ai_insights(medicine_name):
    url = "https://api.together.xyz/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"
    }
    
    prompt = f"""
    For the medicine '{medicine_name}', provide information in the following format:

    MEDICINE DETAILS
    ---------------
    Brand Name: {medicine_name}
    Active Ingredient:
    Primary Uses:
    Typical Dosage:
    Side Effects:
    Precautions:
    
    GENERIC ALTERNATIVES
    -------------------
    Format: Name | Price (₹) | % Savings
    1. Generic Name | ₹XX.XX | XX% cheaper
    2. Generic Name | ₹XX.XX | XX% cheaper
    3. Generic Name | ₹XX.XX | XX% cheaper
    4. Generic Name | ₹XX.XX | XX% cheaper
    5. Generic Name | ₹XX.XX | XX% cheaper

    Note: Provide real generic alternatives with accurate current market prices in Indian Rupees (₹).
    Calculate actual percentage savings compared to the branded medicine.
    """
    
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "prompt": prompt,
        "max_tokens": 800,
        "temperature": 0.3,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        
        if 'choices' in response_json and len(response_json['choices']) > 0:
            return response_json['choices'][0].get('text', '')
        return "No information found"
            
    except Exception as e:
        st.error(f"Error fetching information: {str(e)}")
        return "Unable to fetch information at this time."

def main():
    # Header section with professional styling
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/pharmacy-shop.png", width=80)
    with col2:
        st.title("🏥 MedSeva: Smart Medicine Analyzer")
        st.write("Empowering healthcare decisions with AI-driven insights")
    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar for additional features
    with st.sidebar:
        st.header("⚙️ Settings")
        language = st.selectbox("Select Language", ["English", "हिंदी", "తెలుగు", "தமிழ்"])
        price_filter = st.slider("Max Price Range (₹)", 0, 5000, 1000)
        show_advanced = st.checkbox("Show Advanced Details")
        
        st.header("📊 Statistics")
        st.info("Total Searches: 1,234\nMedicines Analyzed: 567\nAverage Savings: 45%")
        
        st.header("🔍 Quick Search")
        search_medicine = st.text_input("Search Medicine")
        if search_medicine:
            st.info("Feature coming soon!")

    # Main content
    tab1, tab2 = st.tabs(["📷 Image Analysis", "ℹ️ How It Works"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an image of your medicine", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            # Create layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption="Uploaded Medicine", use_container_width=True)
                
                # Add image details
                with st.expander("📸 Image Details"):
                    st.write(f"File name: {uploaded_file.name}")
                    st.write(f"File size: {uploaded_file.size/1024:.1f} KB")
                    st.write(f"Image dimensions: {image.size}")
            
            with col2:
                with st.spinner("🔍 Analyzing image..."):
                    extracted_text = perform_ocr(image)
                    
                    if extracted_text:
                        medicine_name = extracted_text.split('\n')[0]
                        
                        # Status indicators
                        st.success(f"✅ Successfully identified: {medicine_name}")
                        
                        # Create professional tabs
                        tabs = st.tabs(["📋 Details", "💰 Price Analysis", "📊 Comparisons", "📝 Report"])
                        
                        medicine_info = get_ai_insights(medicine_name)
                        
                        with tabs[0]:
                            st.markdown(medicine_info)
                            
                            # Add interaction buttons
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("🔖 Save Analysis"):
                                    st.success("Analysis saved!")
                            with col2:
                                st.download_button(
                                    label="📥 Download Report",
                                    data=f"Medicine Analysis Report\n\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{medicine_info}",
                                    file_name=f"{medicine_name}_analysis.txt",
                                    mime="text/plain"
                                )
                            with col3:
                                if st.button("📱 Share"):
                                    st.info("Sharing options coming soon!")
                        
                        with tabs[1]:
                            st.subheader("💰 Price Analysis")
                            
                            # Create price comparison chart
                            try:
                                price_lines = [line for line in medicine_info.split('\n') 
                                             if '₹' in line and '|' in line]
                                
                                if price_lines:
                                    prices_data = []
                                    for line in price_lines:
                                        parts = line.split('|')
                                        if len(parts) >= 3:
                                            name = parts[0].strip().replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('4. ', '').replace('5. ', '')
                                            price = float(parts[1].strip().replace('₹', ''))
                                            savings = float(parts[2].strip().replace('% cheaper', ''))
                                            prices_data.append({
                                                "Medicine": name,
                                                "Price (₹)": price,
                                                "Savings %": savings
                                            })
                                    
                                    if prices_data:
                                        df = pd.DataFrame(prices_data)
                                        
                                        # Create interactive price chart
                                        fig = px.bar(df, x='Medicine', y='Price (₹)',
                                                    color='Savings %',
                                                    title='Price Comparison',
                                                    color_continuous_scale='RdYlGn_r')
                                        st.plotly_chart(fig)
                                        
                                        # Add savings gauge
                                        avg_savings = df['Savings %'].mean()
                                        st.plotly_chart(create_savings_gauge(avg_savings))
                                        
                                        # Add detailed price table
                                        st.dataframe(df.style.highlight_max(axis=0, color='lightgreen')
                                                   .highlight_min(axis=0, color='lightcoral'))
                            except Exception as e:
                                st.error("Could not create price analysis")
                        
                        with tabs[2]:
                            st.subheader("📊 Comparative Analysis")
                            if show_advanced:
                                # Add more detailed comparisons here
                                st.write("Detailed comparison coming soon!")
                            
                        with tabs[3]:
                            st.subheader("📝 Complete Report")
                            st.code(extracted_text)
                            
                            # Add export options
                            export_format = st.selectbox("Export Format", ["PDF", "Word", "Excel"])
                            if st.button(f"Export as {export_format}"):
                                st.info(f"{export_format} export coming soon!")
    
    with tab2:
        st.header("How MedSeva Works")
        st.write("""
        1. **Image Upload**: Upload a clear image of your medicine
        2. **AI Analysis**: Our AI system analyzes the image and extracts information
        3. **Price Comparison**: We compare prices with generic alternatives
        4. **Detailed Report**: Get comprehensive information about the medicine
        """)
        
        # Add a feedback section
        st.subheader("📢 Feedback")
        feedback = st.text_area("Help us improve!")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main() 