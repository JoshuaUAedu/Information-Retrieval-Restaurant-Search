import streamlit as st

from py.data_loader import (
    load_data, build_index, load_model,
    build_doc_embeddings, build_vocab_embeddings
)
from py.search import bm25_search, semantic_search, hybrid_search


@st.cache_resource
def init():
    documents, docs = load_data()
    tokens, invert_index, idf, bm25 = build_index(documents)
    model_s = load_model()
    doc_embeds = build_doc_embeddings(model_s, documents)
    vocab, vocab_embeds = build_vocab_embeddings(model_s, invert_index)
    return docs, bm25, model_s, doc_embeds, vocab_embeds, vocab


def format_results(doc_ids, docs_df):
    formatted = []
    seen = set() #set for unique only
    rank = 1
    for doc_id in doc_ids:
        row = docs_df.iloc[doc_id]
        title = row['name']
        if title in seen: #prevent adding duplicate restaurants
            continue
        seen.add(title)
        formatted.append({
            'rank': rank,
            'title': title,
            'info': f'{row["categories"]} | Rating: {row["stars"]}',
            'doc_id': doc_id,
        })
        rank += 1
    return formatted


#PAGE SETUP
st.set_page_config(page_title='Restaurant Search (AZ)', layout='wide')
st.title('Restaurant Search (AZ) IR')

with st.spinner('Loading models and indexes...'):
    docs, bm25, model_s, doc_embeds, vocab_embeds, vocab = init()

#SESSION STATE
if 'results' not in st.session_state:
    st.session_state.results = []
if 'page' not in st.session_state:
    st.session_state.page = 0

#CONTROLS
model_choice = st.radio(
    'Select Model:',
    ['BM25', 'Semantic', 'Hybrid (Best)'],
    index=2,
    horizontal=True,
)
query = st.text_input('Enter your query')

if st.button('Search'):
    st.session_state.page = 0

    if model_choice == 'BM25':
        raw = bm25_search(query, bm25)
    elif model_choice == 'Semantic':
        raw = semantic_search(query, model_s, doc_embeds, vocab_embeds, vocab)
    else:
        raw = hybrid_search(query, bm25, model_s, doc_embeds, vocab_embeds, vocab)

    st.session_state.results = format_results(raw, docs)

#PAGE 
RESULTS_PER_PAGE = 10
MAX_RESULTS = 50

results = st.session_state.results[:MAX_RESULTS]
start = st.session_state.page * RESULTS_PER_PAGE
end = start + RESULTS_PER_PAGE
page_results = results[start:end]

#DISPLAY
st.subheader('Results')

if page_results:
    for doc in page_results:
        st.markdown(f'**{doc["rank"]}. {doc["title"]}**')
        st.write(doc['info'])
        st.caption(f'Doc ID: {doc["doc_id"]}')
        st.divider()
else:
    st.write('No results yet. Enter a query and press Search.')

#PAGE CONTROLS
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button('<- Previous') and st.session_state.page > 0:
        st.session_state.page -= 1
        st.rerun()

with col3:
    if st.button('Next ->') and end < len(results):
        st.session_state.page += 1
        st.rerun()

total_pages = max(1, min(len(results), MAX_RESULTS) // RESULTS_PER_PAGE)
st.caption(f'Page {st.session_state.page + 1} of {total_pages}')
