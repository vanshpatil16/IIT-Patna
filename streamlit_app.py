import streamlit as st
import requests
import json
import time
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="CitationEdge",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        background-color: #f8f9ff;
    }
    
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        color: #856404;
        font-weight: 500;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        color: #155724;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        color: #721c24;
    }
    
    .report-ready-notification {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        animation: slideDown 0.5s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .report-notification-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .report-notification-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 15px;
    }
    
    .download-button-container {
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📄 CitationEdge</h1>', unsafe_allow_html=True)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False
if 'show_report_notification' not in st.session_state:
    st.session_state.show_report_notification = False
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

# Function to handle report download from integrated response
def handle_report_download(analysis_result, filename):
    """Handle report download from analysis response"""
    try:
        if analysis_result.get("report_available") and "report_pdf" in analysis_result:
            # Decode base64 PDF data
            import base64
            pdf_bytes = base64.b64decode(analysis_result["report_pdf"])
            
            # Create download button
            report_filename = f"citation_report_{filename.replace('.pdf', '')}.pdf"
            
            st.download_button(
                label="📥 Download Report Now",
                data=pdf_bytes,
                file_name=report_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
            
            # Show report info
            report_size_mb = analysis_result.get("report_size_bytes", 0) / (1024 * 1024)
            st.info(f"📄 Report size: {report_size_mb:.2f} MB")
            return True
        else:
            st.warning("⚠️ Report was not generated during analysis. Please try again.")
            return False
            
    except Exception as e:
        st.error(f"❌ Error preparing report download: {str(e)}")
        return False

# Create columns for layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a single PDF file for analysis"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Store uploaded filename
        st.session_state.uploaded_filename = uploaded_file.name
        
        # Display file info
        st.markdown('<div class="success-message">', unsafe_allow_html=True)
        st.write(f"✅ **File uploaded:** {uploaded_file.name}")
        st.write(f"📊 **File size:** {uploaded_file.size / 1024:.1f} KB")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze button
        st.markdown("### 🚀 Start Analysis")
        
        # Replace your existing API call section with this improved version:

        if st.button("🔍 Analyze Paper", type="primary", use_container_width=True):
            st.session_state.is_analyzing = True
            st.session_state.analysis_result = None
            st.session_state.show_report_notification = False
            
            # Show loading widget
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            
            def update_progress():
                """Update progress periodically for long operations"""
                elapsed = time.time() - start_time
                if elapsed < 300:  # First 5 minutes
                    progress = min(20 + (elapsed / 300) * 30, 50)
                elif elapsed < 1800:  # 5-30 minutes
                    progress = min(50 + ((elapsed - 300) / 1500) * 30, 80)
                else:  # 30+ minutes
                    progress = min(80 + ((elapsed - 1800) / 1800) * 15, 95)
                
                progress_bar.progress(int(progress))
                
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                status_text.text(f"🔄 Analysis in progress... ({minutes}m {seconds}s elapsed)")
            
            try:
                status_text.text("🔄 Preparing analysis request...")
                progress_bar.progress(10)
                
                # Prepare the API request
                files = {'pdf_file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
                data = {
                    'generate_report': True
                }
                
                status_text.text("🚀 Sending request to backend...")
                progress_bar.progress(20)
                
                # Make API request with better error handling
                response = requests.post(
                    "http://localhost:8000/citation-edge-paper",
                    files=files,
                    data=data,
                    timeout=3600,  # 1 hour (60 minutes * 60 seconds)
                    stream=False  # Don't stream for now to avoid partial responses
                )
                
                progress_bar.progress(80)
                status_text.text("📊 Processing response...")
                
                # Check response status
                if response.status_code == 200:
                    try:
                        result = response.json()
                        st.session_state.analysis_result = result
                        st.session_state.is_analyzing = False
                        st.session_state.show_report_notification = True
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis completed!")
                        
                        st.success("✅ Analysis completed successfully!")
                        time.sleep(1)  # Brief pause before rerun
                        st.rerun()
                        
                    except json.JSONDecodeError as json_err:
                        st.session_state.is_analyzing = False
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Invalid response format from server. Please try again.")
                        st.error(f"Debug info: {str(json_err)}")
                        
                elif response.status_code == 413:
                    st.session_state.is_analyzing = False
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ File too large. Please try a smaller PDF.")
                    
                elif response.status_code == 400:
                    st.session_state.is_analyzing = False
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Invalid file format. Please upload a valid PDF.")
                    
                elif response.status_code == 500:
                    st.session_state.is_analyzing = False
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Server error during analysis. Please try again.")
                    # Show response text for debugging
                    try:
                        error_detail = response.json().get('detail', 'Unknown server error')
                        st.error(f"Server error: {error_detail}")
                    except:
                        st.error(f"Server response: {response.text[:500]}...")
                        
                else:
                    st.session_state.is_analyzing = False
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Unexpected response (Status: {response.status_code})")
                    st.error(f"Response: {response.text[:200]}...")
                    
            except requests.exceptions.Timeout:
                st.session_state.is_analyzing = False
                progress_bar.empty()
                status_text.empty()
                st.error("❌ Request timed out after 1 hour. The analysis is taking too long.")
                st.error("This might happen with very large documents or complex analysis.")
                st.info("💡 Try using Quick Analysis mode or a smaller document.")
                
            except requests.exceptions.ConnectionError:
                st.session_state.is_analyzing = False
                progress_bar.empty()
                status_text.empty()
                st.error("❌ Cannot connect to the analysis server.")
                st.error("Please check if the backend server is running on http://localhost:8000")
                
            except requests.exceptions.RequestException as req_err:
                st.session_state.is_analyzing = False
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Network error: {str(req_err)}")
                st.error("Please check your connection and try again.")
                
            except MemoryError:
                st.session_state.is_analyzing = False
                progress_bar.empty()
                status_text.empty()
                st.error("❌ Out of memory. The response is too large to process.")
                st.info("💡 Try disabling report generation or using a smaller document.")
                
            except Exception as e:
                st.session_state.is_analyzing = False
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Unexpected error: {str(e)}")
                st.error("Please check the console for more details and try again.")
                # Log the full error for debugging
                st.exception(e)

# Show report ready notification
if st.session_state.show_report_notification and st.session_state.analysis_result:
    st.markdown("""
    <div class="report-ready-notification">
        <div class="report-notification-title">🎉 Your Analysis Report is Ready!</div>
        <div class="report-notification-subtitle">
            Analysis completed successfully. Your detailed citation report has been generated and is ready for download.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Download report button
    with st.container():
        col_download1, col_download2, col_download3 = st.columns([1, 2, 1])
        with col_download2:
            if handle_report_download(st.session_state.analysis_result, st.session_state.uploaded_filename):
                # Success message for report generation
                st.success("📄 Report ready for download!")
            
            # Add a dismiss button for the notification
            if st.button("✖️ Dismiss Notification", use_container_width=True):
                st.session_state.show_report_notification = False
                st.rerun()

# Display results if available
if st.session_state.analysis_result and not st.session_state.is_analyzing:
    st.markdown("---")
    st.markdown("### 📋 Analysis Results")
    
    # Create an expandable section for results
    with st.expander("🔍 View Detailed Results", expanded=False):
        # Display JSON response in a formatted way
        result = st.session_state.analysis_result
        
        if isinstance(result, dict):
            # Display analysis data if it's nested
            analysis_data = result.get("analysis", result)
            
            # Try to display in a structured format
            for key, value in analysis_data.items():
                if key == "_timing_info":
                    continue  # Skip timing info in main display
                elif key.lower() in ['summary', 'conclusion', 'abstract']:
                    st.markdown(f"**{key.title()}:**")
                    st.markdown(value)
                    st.markdown("---")
                elif isinstance(value, (list, dict)):
                    st.markdown(f"**{key.title()}:**")
                    st.json(value)
                else:
                    st.markdown(f"**{key.title()}:** {value}")
        else:
            # Display as plain text if not a dict
            st.text(str(result))
    
    # Download JSON results button
    st.markdown("### 💾 Download Raw Data")
    col_json1, col_json2, col_json3 = st.columns([1, 2, 1])
    with col_json2:
        # Prepare JSON data (analysis part only, not the full response)
        json_data = st.session_state.analysis_result.get("analysis", st.session_state.analysis_result)
        json_string = json.dumps(json_data, indent=2)
        st.download_button(
            label="📥 Download JSON Results",
            data=json_string,
            file_name=f"analysis_results_{st.session_state.uploaded_filename.replace('.pdf', '') if st.session_state.uploaded_filename else 'results'}.json",
            mime="application/json",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        📄 PDF Analysis Tool | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

# Instructions sidebar
with st.sidebar:
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Upload** a PDF file
    2. **Choose** analysis type:
       - Quick: Fast analysis (5 arguments)
       - Deep: Comprehensive analysis (15-30 mins)
    3. **Click** Analyze Paper
    4. **Wait** for results
    5. **Download** citation report when ready
    6. **Download** raw JSON data if needed
    """)
    
    st.markdown("### ℹ️ Tips")
    st.markdown("""
    - Ensure your PDF is readable
    - Deep analysis provides more insights
    - Citation report includes formatted analysis
    - Raw JSON contains all extracted data
    - Check your internet connection
    """)
    
    st.markdown("### 📄 Report Features")
    st.markdown("""
    - Professional PDF format
    - Citation gap analysis
    - Argumentation structure
    - Literary quality score
    - Keyword extraction results
    """)