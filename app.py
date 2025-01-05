import streamlit as st

def main():
    st.set_page_config(page_title="Summarizer Home", page_icon=":house:")
 
    st.sidebar.header("Choose a Summarizer")
    
    # Buttons for navigation
    summarizer_option = st.sidebar.radio("Select a summarizer", ["Home", "PDF Summarizer", "CSV Summarizer", "Website Summarizer"])

    if summarizer_option == "Home":
        show_home_page()
    
    elif summarizer_option == "PDF Summarizer":
        pdf_app()

    elif summarizer_option == "CSV Summarizer":
        csv_app()

    elif summarizer_option == "Website Summarizer":
        website_app()

def show_home_page():
    st.title("Summarizer Home Page :house:")
    st.write("""
    Welcome to the Summarizer Home Page! You can navigate to the individual summarizers using the sidebar.
    """)
  

def pdf_app():  
    import pdf  
    pdf.main()  

def csv_app():
    import csv1 
    csv1.main()  

def website_app():
    import website 
    website.main()

if __name__ == "__main__":
    main()
