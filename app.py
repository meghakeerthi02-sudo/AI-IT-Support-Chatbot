import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='AI IT Support', page_icon='🤖')
data = pd.read_csv(Path(__file__).with_name('it_support.csv'))
vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
vectors = vectorizer.fit_transform(data['query'])

def get_response(query):
    scores = cosine_similarity(vectorizer.transform([query]), vectors)[0]
    idx = scores.argmax()
    if scores[idx] < 0.30:
        return 'I could not identify your issue. Please provide more details or contact IT support.'
    row = data.iloc[idx]
    return f"**Category:** {row['category']}\n\n**Solution:** {row['solution']}"

st.title('🤖 AI IT Support Chatbot')
st.caption('Troubleshooting assistant • NLP and TF-IDF')
st.sidebar.info('Do not enter passwords, OTPs or confidential information.')
if 'messages' not in st.session_state:
    st.session_state.messages = []
if st.sidebar.button('Clear chat'):
    st.session_state.messages = []
    st.rerun()
with st.chat_message('assistant'):
    st.write('Hello! Describe your technical issue.')
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
if query := st.chat_input('Describe your IT issue...'):
    st.session_state.messages.append({'role': 'user', 'content': query})
    with st.chat_message('user'):
        st.write(query)
    response = get_response(query)
    with st.chat_message('assistant'):
        st.markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
